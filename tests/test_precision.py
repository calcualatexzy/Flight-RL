from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
from scipy.interpolate import BPoly
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree

from FlightEnv.evaluation import TrackingEvaluation, selection_key, tracking_metrics
from FlightEnv.retiming import cruise_phase, retime_reference
from FlightEnv.tracking_env import TrackingEnv
from FlightEnv.trajectory_dataset import FLIGHT_FAMILIES, generate_path, sample_bank
from main import parse_args


def test_cruise_has_short_smooth_ramps_and_constant_middle_speed():
    u = np.linspace(0, 1, 10001)
    s, v, a = cruise_phase(u)
    np.testing.assert_allclose(s[[0, -1]], [0, 1])
    np.testing.assert_allclose(v[[0, -1]], 0)
    np.testing.assert_allclose(a[[0, -1]], 0)
    np.testing.assert_allclose(np.gradient(s, u)[2:-2], v[2:-2], atol=2e-6)
    np.testing.assert_allclose(np.gradient(v, u)[2:-2], a[2:-2], atol=.003)
    assert np.ptp(v[(u >= .15) & (u <= .85)]) < 1e-10


@pytest.mark.parametrize('family', FLIGHT_FAMILIES)
def test_retiming_preserves_geometry_and_derivatives_within_dynamic_limits(family):
    arrays = generate_path(family, 72, 'validation')[:3]
    # Exercise the float32 precision of the on-disk dataset.
    arrays = [a.astype(np.float32).astype(float) for a in arrays]
    position, velocity, acceleration, meta = retime_reference(*arrays, .02,
        time_profile='cruise', speed_scale=1.3)
    np.testing.assert_allclose(position[[0, -1]], arrays[0][[0, -1]], atol=1e-6)
    np.testing.assert_allclose(velocity[[0, -1]], 0, atol=1e-7)
    np.testing.assert_allclose(acceleration[[0, -1]], 0, atol=1e-7)
    temporal = BPoly.from_derivatives(np.arange(501)*.02, list(zip(*arrays)))
    distance, _ = cKDTree(temporal(np.linspace(0, 10, 10001))).query(position)
    assert distance.max() < .005
    np.testing.assert_allclose(np.gradient(position, .02, axis=0)[2:-2], velocity[2:-2], atol=.008)
    # At sharp bends, a 50 Hz finite difference averages the changing
    # acceleration. Check its integral rather than demanding point equality.
    integrated_velocity = cumulative_trapezoid(acceleration, dx=.02, axis=0, initial=0)
    np.testing.assert_allclose(integrated_velocity, velocity, atol=.025)
    assert meta['reference_peak_speed'] <= 5.
    assert meta['reference_peak_acceleration'] <= 8.
    assert meta['reference_peak_jerk'] <= 40.
    assert meta['reference_peak_thrust_ratio'] <= 1.5
    assert acceleration[:, 2].min() + 9.8 >= .45*9.8 - 1e-6
    assert meta['effective_speed_scale'] <= 1.3 + 1e-9
    assert meta['reference_duration'] == pytest.approx((len(position)-1)*.02)


def test_quintic_speed_scaling_changes_velocity_and_acceleration_consistently():
    pos, vel, acc, _ = generate_path('line', 9)
    fast_pos, fast_vel, fast_acc, meta = retime_reference(pos, vel, acc, .02, speed_scale=1.25)
    assert not meta['speed_limited']
    np.testing.assert_allclose(fast_pos[::4], pos[::5], atol=1e-9)
    np.testing.assert_allclose(fast_vel[::4], vel[::5]*1.25, atol=1e-9)
    np.testing.assert_allclose(fast_acc[::4], acc[::5]*1.25**2, atol=1e-9)


def test_original_reference_is_unchanged_and_reset_does_not_compound_speed(tracking_dataset):
    with TrackingEnv(dataset_dir=tracking_dataset, difficulty=1., dataset_split='validation',
                     is_domain_randomization=False, trajectory_family='line') as env:
        options = dict(trajectory_id=4, time_profile='quintic', speed_scale=1.)
        env.reset(seed=42, options=options)
        original = sample_bank(np.random.default_rng(0), 10., .02, trajectory_id=4,
                               split='validation', directory=tracking_dataset)
        np.testing.assert_array_equal(env.state_goal[:, [0, 2, 4]], original[0])
        np.testing.assert_array_equal(env.reference_acceleration, original[2])
        _, info = env.reset(seed=42, options=dict(options, speed_scale=1.25))
        first = env.state_goal.copy()
        env.reset(seed=42, options=dict(options, speed_scale=1.25))
        np.testing.assert_array_equal(first, env.state_goal)
        assert env.max_episode_steps == 400
        assert env._episode_len_sec == 8.
        env.reset(seed=42, options=options)
        assert env.max_episode_steps == 500 and env._episode_len_sec == 10.


def test_precision_reward_increases_pressure_at_10_and_20_cm(tracking_dataset):
    drops = {}
    for profile in ('legacy', 'precision'):
        with TrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False,
                         reward_profile=profile) as env:
            target = env.state_goal[0, [0, 2, 4]]
            values = []
            for error in (0., .1, .2, 2.):
                p.resetBasePositionAndOrientation(env.quadrotor.my_quadrotor, target+[error, 0, 0],
                                                  [0, 0, 0, 1], physicsClientId=env.PYB_CLIENT)
                env.quadrotor._update_and_store_kinematic_information()
                env._get_reward(np.zeros(4))
                values.append(env.reward_terms['position'])
            assert all(a > b for a, b in zip(values, values[1:])) and values[-1] > 0
            drops[profile] = np.array(values[0])-values[1:3]
    assert np.all(drops['precision'] > 4*drops['legacy'])


def test_failed_episode_cannot_pass_precision_or_beat_completed_episode():
    failed = tracking_metrics([0., 0.], False)
    completed = tracking_metrics([.2, .2], True)
    accurate = tracking_metrics([.08, .09], True)
    spike = tracking_metrics([.31]+[0.]*99, True)
    assert not failed['tracking_success']
    assert completed['legacy_tracking_success'] and not completed['tracking_success']
    assert accurate['tracking_success'] and not spike['tracking_success']
    assert selection_key([completed]) > selection_key([failed])
    assert selection_key([accurate]) > selection_key([completed])


def test_speed_curriculum_requires_two_passing_validations(tracking_dataset, tmp_path):
    config = dict(trajectory_family='mixed', dataset_dir=tracking_dataset, speed_min=.9, speed_max=1.1)
    calls = []
    with TrackingEnv(dataset_dir=tracking_dataset, dataset_split='validation') as env:
        callback = TrackingEvaluation(env, config, tmp_path/'validation', 100, 16, 10042, None,
                                      speed_scales=[1., 1.15, 1.3], time_profiles=['quintic', 'cruise'],
                                      curriculum_max_speed=1.3)
        callback.model = SimpleNamespace(flight_env_config=config,
            get_env=lambda: SimpleNamespace(env_method=lambda *args: calls.append(args)))
        groups = {f'{p}_{v:g}x': dict(completion_rate=1., success_rate=.75, mean_rmse=.09)
                  for p in callback.time_profiles for v in callback.speed_scales}
        callback._advance_curriculum(groups)
        assert not calls
        # Even passing faster probes cannot stand in for an unmeasured current
        # boundary. Only the additional current-speed probe unlocks promotion.
        callback._advance_curriculum(groups)
        assert not calls
        for profile in callback.time_profiles:
            groups[f'{profile}_1.1x'] = dict(completion_rate=1., success_rate=.75, mean_rmse=.09)
        callback._advance_curriculum(groups)
        groups['cruise_1.1x']['completion_rate'] = .5
        callback._advance_curriculum(groups)
        assert not calls
        groups['cruise_1.1x']['completion_rate'] = 1.
        callback._advance_curriculum(groups)
        callback._advance_curriculum(groups)
        assert calls == [('set_speed_range', .9, 1.2)]
        assert config['speed_max'] == 1.2


@pytest.mark.parametrize('flags', [ ['--speed-min', '0'], ['--speed-min', '2', '--speed-max', '1'],
    ['--ent-coef', '-1'], ['--target-kl', 'nan'], ['--eval-speed-scales', '1', '1'],
    ['--curriculum-max-speed', '1.3', '--eval-freq', '0']])
def test_precision_cli_rejects_invalid_settings(flags):
    with pytest.raises(SystemExit):
        parse_args(['train', *flags])


def test_tight_bend_does_not_force_slower_cruise_than_feasible_original():
    arrays = generate_path('slalom', 72, 'validation')[:3]
    old = retime_reference(*arrays, .02, speed_scale=1.3, time_profile='quintic')
    new = retime_reference(*arrays, .02, speed_scale=1.3, time_profile='cruise')
    assert new[3]['reference_duration'] <= old[3]['reference_duration']
    assert new[3]['cruise_fallback']
    assert new[3]['effective_time_profile'] == 'quintic_fallback'

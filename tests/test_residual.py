from types import SimpleNamespace
import json

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from FlightEnv.residual_env import ResidualTrackingEnv, geometric_action
from FlightEnv.tracking_env import TrackingEnv
from FlightEnv.evaluation import TrackingEvaluation, tracking_metrics


def test_residual_api_and_seed_reproducibility(tracking_dataset):
    with ResidualTrackingEnv(dataset_dir=tracking_dataset, difficulty=1.) as env:
        check_env(env)
        obs, _ = env.reset(seed=5)
        assert obs.shape == (67,) and env.observation_space.contains(obs)
        first = [env.step(np.zeros(4))[0] for _ in range(12)]
        env.reset(seed=5)
        second = [env.step(np.zeros(4))[0] for _ in range(12)]
        np.testing.assert_array_equal(first, second)


def test_zero_residual_matches_explicit_controller_in_same_physics(tracking_dataset):
    config = dict(dataset_dir=tracking_dataset, difficulty=1., trajectory_family='figure8',
                  dataset_split='validation', reward_profile='precision')
    with TrackingEnv(**config) as direct, ResidualTrackingEnv(**config) as hybrid:
        direct.reset(seed=41)
        hybrid.reset(seed=41)
        for _ in range(150):
            action, _ = geometric_action(direct)
            direct.step(action)
            hybrid.step(np.zeros(4))
            np.testing.assert_allclose(hybrid.quadrotor.pos, direct.quadrotor.pos, atol=1e-10)
            np.testing.assert_allclose(hybrid.last_control, direct.last_control, atol=1e-10)


def test_residual_changes_real_motor_commands_and_observation_history(tracking_dataset):
    with ResidualTrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False) as env:
        env.reset(seed=1)
        nominal, _ = geometric_action(env)
        residual = np.array([.5, .4, -.3, .2])
        expected = np.clip(nominal+env.residual_scale*residual, -1, 1)
        obs, reward, _, _, info = env.step(residual)
        np.testing.assert_allclose(env.last_control, expected)
        np.testing.assert_allclose(obs[55:59], expected, atol=1e-7)
        np.testing.assert_allclose(obs[-4:], residual, atol=1e-7)
        assert env.quadrotor.vel[2] > .005
        assert np.linalg.norm(env.quadrotor.ang_vel) > .005
        assert info['control_mode'] == 'geometric_feedforward_plus_rl_residual'
        assert info['reward_residual_effort'] < 0
        assert reward == pytest.approx(sum(v for k, v in info.items() if k.startswith('reward_')))


def test_zero_residual_hovers_and_preserves_terminal_semantics(tracking_dataset):
    with ResidualTrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False,
                             speed_min=1.25, speed_max=1.25) as env:
        env.reset(seed=1)
        start = np.array(env.quadrotor.pos)
        for _ in range(env.max_episode_steps):
            _, _, terminated, truncated, _ = env.step(np.zeros(4))
        assert truncated and not terminated and env.max_episode_steps == 400
        np.testing.assert_allclose(env.quadrotor.pos, start, atol=1e-7)


def test_validation_plateau_stops_and_keeps_initial_baseline(tracking_dataset, tmp_path):
    saved = []
    def save(model, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test checkpoint')
        saved.append(str(path))
    config = dict(trajectory_family='mixed', dataset_dir=tracking_dataset, profile='residual')
    with ResidualTrackingEnv(dataset_dir=tracking_dataset, dataset_split='validation') as env:
        callback = TrackingEvaluation(env, config, tmp_path/'validation', 1, 8, 10042, save, patience=2)
        callback.model = SimpleNamespace(flight_env_config=config, logger=SimpleNamespace(record=lambda *a: None))
        def scenario(profile, speed):
            return [dict(tracking_metrics([.05], True), effective_speed_scale=speed,
                         speed_limited=False, cruise_fallback=False)]*8
        callback._evaluate_scenario = scenario
        callback.evaluate(initial=True)
        assert (tmp_path/'initial_model.zip').is_file()
        assert (tmp_path/'baseline.json').is_file()
        callback.num_timesteps = 100
        assert callback._on_step()
        callback.num_timesteps = 200
        assert not callback._on_step()
        assert len(saved) == 2  # initial and best, never overwrite best with a tie/regression
        assert json.loads((tmp_path/'early_stop.json').read_text())['timesteps'] == 200
        assert json.loads((tmp_path/'best/validation.json').read_text())['timesteps'] == 0


def test_resume_preserves_best_and_rejects_changed_validation_suite(tracking_dataset, tmp_path):
    config = dict(trajectory_family='mixed', dataset_dir=tracking_dataset, profile='residual', speed_max=1.3)
    with ResidualTrackingEnv(dataset_dir=tracking_dataset, dataset_split='validation') as env:
        callback = TrackingEvaluation(env, config, tmp_path/'validation', 1, 8, 10042, None)
        (tmp_path/'best').mkdir()
        (tmp_path/'best/validation.json').write_text(json.dumps(dict(
            suite_signature=callback.suite_signature, selection_key=[8, 8, -.03])))
        restored = TrackingEvaluation(env, dict(config, speed_max=1.4), tmp_path/'validation', 1, 8, 10042, None)
        assert restored.best_key == (8, 8, -.03)
        with pytest.raises(ValueError, match='fresh run directory'):
            TrackingEvaluation(env, config, tmp_path/'validation', 1, 8, 10043, None)

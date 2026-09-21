import numpy as np
import pybullet as p
import pytest
from stable_baselines3.common.env_checker import check_env

from FlightEnv.env import FlightEnv
from FlightEnv.quadrotor import Physics
from FlightEnv.trajectories import LIBRARY, library_members, sample_trajectory


def test_sb3_contract_and_zero_history():
    with FlightEnv(mode='hover', last_horizon=0) as env:
        check_env(env)
        obs, _ = env.reset(seed=3)
        assert obs.shape == (72,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)


@pytest.mark.parametrize('hard_reset', [False, True])
def test_seed_reproduces_state_physics_trajectory_and_steps(hard_reset):
    with FlightEnv(hard_reset=hard_reset) as env:
        def rollout():
            initial, _ = env.reset(seed=17)
            goals = env.state_goal.copy()
            mass = env.quadrotor.MASS
            states = [env.step(np.full(4, .05))[0] for _ in range(8)]
            return initial, goals, mass, states
        first = rollout()
        second = rollout()
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
        changed, _ = env.reset(seed=18)
        assert not np.allclose(first[0], changed)


def test_motor_commands_change_acceleration_and_attitude():
    with FlightEnv(mode='hover', is_domain_randomization=False) as env:
        states = []
        for action in (np.zeros(4), np.ones(4), np.array([1., -1., -1., 1.])):
            env.reset(seed=0)
            for _ in range(10):
                env.step(action)
            states.append(env._state.copy())
        assert abs(states[0][4] - .5) < 1e-6
        assert states[1][5] > states[0][5] + .5
        assert abs(states[2][7]) > .1


def test_history_reset_and_correct_previous_state():
    with FlightEnv(mode='hover', last_horizon=2, is_domain_randomization=False) as env:
        initial, _ = env.reset(seed=1)
        state = env._state.copy()
        obs, *_ = env.step(np.ones(4))
        np.testing.assert_allclose(obs[12:24], state)
        reset, _ = env.reset(seed=1)
        np.testing.assert_array_equal(reset, initial)


def test_time_limit_truncates_but_collision_terminates():
    with FlightEnv(mode='hover', episode_len_sec=.05, is_domain_randomization=False) as env:
        for _ in range(10):
            obs, reward, terminated, truncated, _ = env.step(np.zeros(4))
        assert truncated and not terminated
        with pytest.raises(RuntimeError, match='reset'):
            env.step(np.zeros(4))
    with FlightEnv(mode='hover', is_domain_randomization=False) as env:
        for _ in range(1000):
            obs, reward, terminated, truncated, info = env.step(-np.ones(4))
            if terminated or truncated:
                break
        assert terminated and not truncated and info['collision']
        assert reward < 0
        assert env.observation_space.contains(obs)


def test_multiple_clients_keep_their_own_dynamics_and_disconnect():
    with FlightEnv(is_domain_randomization=False) as first:
        first_mass = p.getDynamicsInfo(first.quadrotor.my_quadrotor, -1,
                                      physicsClientId=first.PYB_CLIENT)[0]
        second = FlightEnv()
        client = second.PYB_CLIENT
        second.reset(seed=23)
        actual_mass = p.getDynamicsInfo(second.quadrotor.my_quadrotor, -1,
                                       physicsClientId=client)[0]
        assert actual_mass == pytest.approx(second.quadrotor.MASS)
        assert p.getDynamicsInfo(first.quadrotor.my_quadrotor, -1,
                                physicsClientId=first.PYB_CLIENT)[0] == first_mass
        assert np.all(np.linalg.eigvalsh(second.quadrotor.J) > 0)
        np.testing.assert_allclose(second.quadrotor.J @ second.quadrotor.J_INV, np.eye(3), atol=1e-12)
        second.close()
        second.close()
        assert not p.isConnected(client)
        assert p.isConnected(first.PYB_CLIENT)


def test_zip_library_and_assets_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(library_members(str(LIBRARY))) == 400
    with FlightEnv() as env:
        assert env.state_goal.shape == (1001, 12)
        assert np.isfinite(env.state_goal).all()
        image = env.render()
        assert image.shape == (360, 480, 3) and image.dtype == np.uint8
    rng = np.random.default_rng(3)
    a = sample_trajectory(5., .005, rng)
    b = sample_trajectory(10., .01, np.random.default_rng(3))
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1] / 2, b[1])
    np.testing.assert_allclose(a[2] / 4, b[2])


@pytest.mark.parametrize('kwargs', [dict(ctrl_freq=201), dict(last_horizon=-1),
                                   dict(episode_len_sec=0), dict(physics=Physics.DYN)])
def test_invalid_configs_fail_before_connecting(kwargs):
    with pytest.raises(ValueError):
        FlightEnv(**kwargs)


def test_nonfinite_or_wrong_shape_actions_are_rejected():
    with FlightEnv(mode='hover') as env:
        for action in ([np.nan, 0, 0, 0], [0, 0, 0], [[0, 0, 0, 0]]):
            with pytest.raises(ValueError):
                env.step(action)


def test_drag_dissipates_energy_in_rotated_body_frame(monkeypatch):
    with FlightEnv(mode='hover', is_domain_randomization=False) as env:
        q = env.quadrotor
        q.quat = p.getQuaternionFromEuler([.3, -.2, 1.5])
        q.vel = np.array([2., -1., .5])
        forces = []
        monkeypatch.setattr(p, 'applyExternalForce', lambda *a, **kw: forces.append(kw))
        q._drag(np.full(4, q.HOVER_RPM))
        body_force = np.asarray(forces[0]['forceObj'])
        rotation = np.asarray(p.getMatrixFromQuaternion(q.quat)).reshape(3, 3)
        assert np.dot(rotation @ body_force, q.vel) < 0
        expected = -q.DRAG_COEFF * (4 * 2*np.pi*q.HOVER_RPM/60) * (rotation.T @ q.vel)
        np.testing.assert_allclose(body_force, expected)


def test_preview_spacing_and_recording(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with FlightEnv(goal_stride=20, record=True) as env:
        obs, _ = env.reset(seed=0)
        np.testing.assert_allclose(obs[24:36], env.state_goal[20], rtol=1e-6)
        for _ in range(20):
            env.step(np.zeros(4))
        frames = list(env.ONBOARD_IMG_PATH.glob('*.png'))
        assert len(frames) == 1

import numpy as np
import pybullet as p
from stable_baselines3.common.env_checker import check_env

from FlightEnv.tracking_env import TrackingEnv


def test_tracking_api_seed_and_initial_target_alignment(tracking_dataset):
    with TrackingEnv(dataset_dir=tracking_dataset, trajectory_family='figure8', difficulty=1., dataset_split='test') as env:
        check_env(env)
        obs, info = env.reset(seed=17)
        assert obs.shape == (59,) and env.observation_space.contains(obs)
        assert info['position_error'] < .18
        first = [env.step(np.zeros(4))[0] for _ in range(5)]
        env.reset(seed=17)
        second = [env.step(np.zeros(4))[0] for _ in range(5)]
        np.testing.assert_array_equal(first, second)
        assert info['trajectory_family'] == 'figure8' and info['dataset_split'] == 'test'


def test_wrench_mixer_hover_and_independent_moments(tracking_dataset):
    with TrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False) as env:
        env.reset(seed=5)
        height = env.quadrotor.pos[2]
        for _ in range(20):
            env.step(np.zeros(4))
        assert abs(env.quadrotor.pos[2] - height) < 1e-7
        for axis in range(3):
            env.reset(seed=5)
            action = np.zeros(4)
            action[axis+1] = .5
            env.step(action)
            omega = env._state[9:12]
            assert omega[axis] > .03
            assert np.linalg.norm(np.delete(omega, axis)) < .001


def test_far_tracking_error_has_nonzero_informative_reward(tracking_dataset):
    with TrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False) as env:
        values = []
        start = env.state_goal[0, [0, 2, 4]]
        for distance in (0., 2., 4.):
            p.resetBasePositionAndOrientation(env.quadrotor.my_quadrotor, start+[distance,0,0],
                                              [0,0,0,1],physicsClientId=env.PYB_CLIENT)
            env.quadrotor._update_and_store_kinematic_information()
            env._get_reward(np.zeros(4))
            values.append(env.reward_terms['position'])
        assert values[0] > values[1] > values[2] > .02


def test_relative_position_features_are_translation_invariant(tracking_dataset):
    with TrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False) as env:
        initial, _ = env.reset(seed=8)
        offset = np.array([2., -3., 0.])
        env.state_goal[:, [0,2,4]] += offset
        p.resetBasePositionAndOrientation(env.quadrotor.my_quadrotor, np.array(env.quadrotor.pos)+offset,
                                          env.quadrotor.quat,physicsClientId=env.PYB_CLIENT)
        env.quadrotor._update_and_store_kinematic_information()
        np.testing.assert_allclose(initial, env._get_observation(), atol=1e-7)


def test_reward_breakdown_includes_failure_penalty(tracking_dataset):
    with TrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False) as env:
        _, reward, _, _, info = env.step(np.zeros(4))
        np.testing.assert_allclose(reward, sum(v for k, v in info.items() if k.startswith('reward_')))
        p.resetBasePositionAndOrientation(env.quadrotor.my_quadrotor, [16., 0., 1.],
                                          [0, 0, 0, 1], physicsClientId=env.PYB_CLIENT)
        _, reward, terminated, _, info = env.step(np.zeros(4))
        assert terminated and info['reward_failure'] == -30.
        np.testing.assert_allclose(reward, sum(v for k, v in info.items() if k.startswith('reward_')))


def test_heading_reward_penalizes_spin_without_requiring_level_attitude(tracking_dataset):
    with TrackingEnv(dataset_dir=tracking_dataset, mode='hover', is_domain_randomization=False,
                     reward_heading_weight=.1, reward_yaw_rate_weight=.02) as env:
        env._get_reward(np.zeros(4))
        assert env.reward_terms['heading'] == .1
        assert env.reward_terms['yaw_rate'] == 0
        p.resetBasePositionAndOrientation(env.quadrotor.my_quadrotor, env.quadrotor.pos,
                                          p.getQuaternionFromEuler([.3, .2, np.pi/2]),
                                          physicsClientId=env.PYB_CLIENT)
        p.resetBaseVelocity(env.quadrotor.my_quadrotor, angularVelocity=[0., 0., 2.],
                            physicsClientId=env.PYB_CLIENT)
        env.quadrotor._update_and_store_kinematic_information()
        env._get_reward(np.zeros(4))
        assert abs(env.reward_terms['heading']) < 1e-9
        assert env.reward_terms['yaw_rate'] < -.05

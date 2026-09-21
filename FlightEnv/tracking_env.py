"""V3 tracking task: relative observations, dense rewards and mixed controls."""
import gymnasium as gym
import numpy as np
import pybullet as p

from FlightEnv.env import FlightEnv
from FlightEnv.trajectory_dataset import FAMILIES, SPLITS, sample_bank


class TrackingEnv(FlightEnv):
    def __init__(self, *, difficulty=.15, trajectory_family='mixed', dataset_split='train',
                 dataset_dir=None, reward_heading_weight=0., reward_yaw_rate_weight=0.,
                 reward_profile='legacy', time_profile='quintic', speed_min=1., speed_max=1., **kwargs):
        if not 0 <= difficulty <= 1:
            raise ValueError('difficulty must be in [0, 1]')
        if trajectory_family not in ('mixed', *FAMILIES) or dataset_split not in SPLITS:
            raise ValueError('Invalid trajectory family or dataset split')
        if not np.isfinite([reward_heading_weight, reward_yaw_rate_weight]).all() or min(reward_heading_weight, reward_yaw_rate_weight) < 0:
            raise ValueError('Heading/yaw-rate reward weights must be finite and nonnegative')
        self.reward_heading_weight = float(reward_heading_weight)
        self.reward_yaw_rate_weight = float(reward_yaw_rate_weight)
        if reward_profile not in ('legacy', 'precision'):
            raise ValueError('reward_profile must be legacy or precision')
        if time_profile not in ('quintic', 'cruise', 'mixed'):
            raise ValueError('time_profile must be quintic, cruise or mixed')
        self.reward_profile = reward_profile
        self.time_profile = time_profile
        self.set_speed_range(speed_min, speed_max)
        self.difficulty = float(difficulty)
        self.trajectory_family = trajectory_family
        self.dataset_split = dataset_split
        self.dataset_dir = dataset_dir
        self.last_control = np.zeros(4)
        self.previous_control = np.zeros(4)
        self.reward_terms = {}
        self.trajectory_metadata = {}
        kwargs.setdefault('ctrl_freq', 50)
        kwargs.setdefault('pybullet_freq', 200)
        kwargs.setdefault('goal_stride', 5)
        kwargs.setdefault('last_horizon', 0)
        kwargs.setdefault('episode_len_sec', 10.)
        self.nominal_episode_seconds = kwargs['episode_len_sec']
        super().__init__(**kwargs)
        self.observation_dim = 29 + 6*self._goal_horizon
        self.observation_space = gym.spaces.Box(-10., 10., (self.observation_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.last_control = np.zeros(4)
        self.previous_control = np.zeros(4)
        self.reward_terms = {}
        self._reset_options = options or {}
        return super().reset(seed=seed, options=options)

    def _make_reference(self):
        family = 'hover' if self._mode == 'hover' else self.trajectory_family
        family = self._reset_options.get('family', family)
        pos, vel, acc, meta = sample_bank(self.np_random, self.nominal_episode_seconds, self._time_step,
                                          family=family, difficulty=self.difficulty,
                                          split=self.dataset_split, directory=self.dataset_dir,
                                          trajectory_id=self._reset_options.get('trajectory_id'))
        profile = self._reset_options.get('time_profile', self.time_profile)
        if profile == 'mixed':
            profile = str(self.np_random.choice(('quintic', 'cruise'), p=(.4, .6)))
        speed = self._reset_options.get('speed_scale')
        if speed is None:
            speed = (self.speed_min if self.speed_min == self.speed_max else
                     float(self.np_random.uniform(self.speed_min, self.speed_max)))
        if profile not in ('quintic', 'cruise') or not np.isfinite(speed) or speed <= 0:
            raise ValueError('Invalid time profile or speed scale')
        # Keep original V3 references exactly when evaluating original speed.
        if profile != 'quintic' or speed != 1.:
            from FlightEnv.retiming import retime_reference
            pos, vel, acc, timing = retime_reference(pos, vel, acc, self._time_step,
                                                     speed_scale=speed, time_profile=profile)
        else:
            timing = dict(time_profile=profile, effective_time_profile=profile, cruise_fallback=False, requested_speed_scale=1., effective_speed_scale=1.,
                          reference_duration=self.nominal_episode_seconds, speed_limited=False,
                          reference_mean_speed=float(np.mean(np.linalg.norm(vel, axis=1))),
                          reference_peak_speed=float(np.linalg.norm(vel, axis=1).max()))
        meta.update(timing)
        self.max_episode_steps = len(pos)-1
        self._episode_len_sec = self.max_episode_steps*self._time_step
        self.reference_acceleration = acc
        self.trajectory_metadata = meta
        return pos, vel, acc

    def set_speed_range(self, minimum, maximum):
        if not np.isfinite([minimum, maximum]).all() or not 0 < minimum <= maximum:
            raise ValueError('speed range must be finite and 0 < minimum <= maximum')
        self.speed_min, self.speed_max = float(minimum), float(maximum)

    def _reset_initial_pose(self):
        # Start around the first reference point; preserve the seeded disturbance.
        offset = np.asarray(self.quadrotor.pos) - [0., 0., .5]
        position = self.state_goal[0, [0, 2, 4]] + offset
        p.resetBasePositionAndOrientation(self.quadrotor.my_quadrotor, position, self.quadrotor.quat,
                                          physicsClientId=self.PYB_CLIENT)
        self.quadrotor._update_and_store_kinematic_information()
        q = self.quadrotor
        rotation = np.asarray(p.getMatrixFromQuaternion(q.quat)).reshape(3, 3)
        arms = np.array([p.getLinkState(q.my_quadrotor, i, computeForwardKinematics=True,
                                        physicsClientId=self.PYB_CLIENT)[0] for i in range(4)])
        arms = (arms-np.asarray(q.pos)) @ rotation
        allocation = np.vstack((np.ones(4), arms[:, 1], -arms[:, 0],
                                 np.array([-1., 1., -1., 1.]) * q.KM/q.KF))
        self.mixer_inverse = np.linalg.inv(allocation)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('action must contain four finite commands')
        self.previous_control = self.last_control.copy()
        self.last_control = np.clip(action, -1, 1)
        return super().step(action)

    def _action_to_thrust(self, action):
        # Policy controls collective thrust and body moments. This is a static
        # mixer, not a hidden PID controller: stabilization is learned by PPO.
        q = self.quadrotor
        collective = q.MASS * self.GRAVITY_ACC * (1 + .7*action[0])
        torque = q.J.diagonal() * np.array([12., 12., 8.]) * action[1:]
        thrust = self.mixer_inverse @ np.r_[collective, torque]
        return np.clip(thrust, self.action_bounds[0], self.action_bounds[1])

    def _get_observation(self):
        q = self.quadrotor
        rotation = np.asarray(p.getMatrixFromQuaternion(q.quat)).reshape(3, 3)
        position, velocity = np.asarray(q.pos), np.asarray(q.vel)
        idx = self._env_step_counter
        goal = self.state_goal[idx]
        future = np.minimum(idx + self._goal_stride*(1 + np.arange(self._goal_horizon)),
                            len(self.state_goal)-1)
        # Row vectors @ rotation transform world vectors into body coordinates.
        future_position = (self.state_goal[future][:, [0, 2, 4]] - position) @ rotation / 2
        future_velocity = (self.state_goal[future][:, [1, 3, 5]] - velocity) @ rotation / 3
        obs = np.concatenate((rotation.ravel(), velocity @ rotation / 3,
                              np.asarray(q.ang_vel) @ rotation / 5, [position[2]/3],
                              (goal[[0, 2, 4]]-position) @ rotation / 2,
                              (goal[[1, 3, 5]]-velocity) @ rotation / 3,
                              self.reference_acceleration[idx] @ rotation / 6,
                              np.column_stack((future_position, future_velocity)).ravel(),
                              self.last_control))
        return np.clip(obs, -10, 10).astype(np.float32)

    def _get_reward(self, action):
        q = self.quadrotor
        idx = self._env_step_counter
        goal = self.state_goal[idx]
        ep = np.linalg.norm(np.asarray(q.pos) - goal[[0, 2, 4]])
        ev = np.linalg.norm(np.asarray(q.vel) - goal[[1, 3, 5]])
        rotation = np.asarray(p.getMatrixFromQuaternion(q.quat)).reshape(3, 3)
        desired_up = self.reference_acceleration[idx] + [0., 0., self.GRAVITY_ACC]
        desired_up /= np.linalg.norm(desired_up)
        self.reward_terms = {
            'position': 1.4 / (1 + (ep/.6)**2),
            'velocity': .4 / (1 + (ev/.8)**2),
            'alignment': .1 * float(rotation[:, 2] @ desired_up),
            'survival': .1,
            'heading': self.reward_heading_weight * float(np.cos(q.rpy[2])),
            'yaw_rate': -self.reward_yaw_rate_weight * float((np.asarray(q.ang_vel) @ rotation[:, 2])**2),
            'angular_rate': -.002 * float(np.sum(np.asarray(q.ang_vel)**2)),
            'smoothness': -.02 * float(np.sum((self.last_control-self.previous_control)**2)),
            'effort': -.003 * float(np.sum(action**2)),
        }
        if self.reward_profile == 'precision':
            # A wide recovery gradient plus a sharper 5--10 cm target; maintain
            # a similar total reward scale for warm-starting the value function.
            self.reward_terms['position'] = (.25/(1+(ep/.6)**2)
                                             + .35/(1+(ep/.25)**2)
                                             + .8/(1+(ep/.1)**2))
            self.reward_terms['velocity'] = .5/(1+(ev/.4)**2)
            self.reward_terms['smoothness'] *= .25
        return sum(self.reward_terms.values())

    def _get_info(self):
        info = super()._get_info()
        info.update(self.trajectory_metadata)
        info.update({f'reward_{key}': value for key, value in self.reward_terms.items()})
        info['reward_failure'] = -30. if info['collision'] or info['out_of_bounds'] else 0.
        idx = self._env_step_counter
        reference_velocity = self.state_goal[idx, [1, 3, 5]]
        info['reference_speed'] = float(np.linalg.norm(reference_velocity))
        info['actual_speed'] = float(np.linalg.norm(self.quadrotor.vel))
        error = np.asarray(self.quadrotor.pos)-self.state_goal[idx, [0, 2, 4]]
        tangent = reference_velocity / max(info['reference_speed'], 1e-9)
        info['along_track_error'] = float(error @ tangent)
        info['cross_track_error'] = float(np.linalg.norm(error-info['along_track_error']*tangent))
        return info

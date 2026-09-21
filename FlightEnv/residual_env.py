"""Explicit geometric feedforward control with a bounded learned correction.

This is a hybrid controller, not a direct-actuation pure-RL policy. The reference
position/velocity/acceleration and its one-step preview are planned quantities;
no future measured vehicle state is used. Physics and motor limits are unchanged.
"""
import gymnasium as gym
import numpy as np
import pybullet as p

from FlightEnv.tracking_env import TrackingEnv


def _vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def geometric_action(env):
    """Return a normalized collective/moment command and saturation diagnostics."""
    q = env.quadrotor
    idx = env._env_step_counter
    rotation = np.asarray(p.getMatrixFromQuaternion(q.quat)).reshape(3, 3)
    omega = rotation.T @ np.asarray(q.ang_vel)
    position_error = env.state_goal[idx, [0, 2, 4]] - q.pos
    velocity_error = env.state_goal[idx, [1, 3, 5]] - q.vel
    acceleration = env.reference_acceleration[idx]
    force = acceleration + 8*position_error + 5*velocity_error + [0., 0., env.GRAVITY_ACC]
    # Avoid an inverted attitude request during a large recovery disturbance.
    force[2] = max(force[2], 1.)
    norm = np.linalg.norm(force)
    up = force/norm
    heading = np.array([1., 0., 0.])
    side_raw = np.cross(up, heading)
    side = side_raw/np.linalg.norm(side_raw)
    forward = np.cross(side, up)
    desired_rotation = np.column_stack((forward, side, up))
    attitude_error = _vee((desired_rotation.T@rotation-rotation.T@desired_rotation)/2)

    next_idx = min(idx+1, len(env.state_goal)-1)
    jerk = (env.reference_acceleration[next_idx]-acceleration)/env._time_step
    # Acceleration predicted from the preceding *applied* collective command.
    predicted_acceleration = rotation[:, 2]*env.GRAVITY_ACC*(1+.7*env.last_control[0]) - [0., 0., env.GRAVITY_ACC]
    force_dot = jerk + 8*velocity_error + 5*(acceleration-predicted_acceleration)
    up_dot = (np.eye(3)-np.outer(up, up))@force_dot/norm
    side_dot = ((np.eye(3)-np.outer(side, side))@np.cross(up_dot, heading)
                / np.linalg.norm(side_raw))
    forward_dot = np.cross(side_dot, up)+np.cross(side, up_dot)
    rotation_dot = np.column_stack((forward_dot, side_dot, up_dot))
    desired_omega = rotation.T@desired_rotation@_vee(desired_rotation.T@rotation_dot)
    angular_acceleration = (-100*attitude_error - 20*(omega-desired_omega)
                            + np.cross(omega, q.J@omega)/np.diag(q.J))
    raw = np.r_[(force@rotation[:, 2]/env.GRAVITY_ACC-1)/.7,
                angular_acceleration/[12., 12., 8.]]
    return np.clip(raw, -1., 1.), bool(np.any(np.abs(raw) > 1.))


class ResidualTrackingEnv(TrackingEnv):
    """PPO adds bounded corrections to the named geometric control baseline.

    Observation: the original 59 values, nominal command (4), and last residual
    (4). The original last-control feature always contains the applied command.
    """
    def __init__(self, *, residual_scale=(.10, .20, .20, .15), **kwargs):
        scale = np.asarray(residual_scale, dtype=float)
        if scale.shape != (4,) or not np.isfinite(scale).all() or np.any(scale <= 0) or np.any(scale > 1):
            raise ValueError('residual_scale must contain four finite values in (0, 1]')
        self.residual_scale = scale
        self.last_residual = np.zeros(4)
        self.nominal_control = np.zeros(4)
        self.nominal_saturated = False
        self.combined_saturated = False
        kwargs.setdefault('reward_profile', 'precision')
        kwargs.setdefault('reward_heading_weight', .1)
        kwargs.setdefault('reward_yaw_rate_weight', .02)
        super().__init__(**kwargs)
        self.observation_dim += 8
        self.observation_space = gym.spaces.Box(-10., 10., (self.observation_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.last_residual = np.zeros(4)
        self.nominal_control = np.zeros(4)
        self.nominal_saturated = False
        self.combined_saturated = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('action must contain four finite residual commands')
        self.last_residual = np.clip(action, -1., 1.)
        # Observation construction cached this command at the current state.
        nominal = self.nominal_control
        combined = nominal + self.residual_scale*self.last_residual
        self.combined_saturated = bool(np.any(np.abs(combined) > 1.))
        # TrackingEnv receives physical normalized commands for its history,
        # smoothness reward, mixer, and actual PyBullet motor actuation.
        return super().step(np.clip(combined, -1., 1.))

    def _get_observation(self):
        original = super()._get_observation()
        self.nominal_control, self.nominal_saturated = geometric_action(self)
        return np.r_[original, self.nominal_control, self.last_residual].astype(np.float32)

    def _get_reward(self, action):
        super()._get_reward(action)
        self.reward_terms['residual_effort'] = -.01*float(np.sum(self.last_residual**2))
        return sum(self.reward_terms.values())

    def _get_info(self):
        info = super()._get_info()
        info.update(control_mode='geometric_feedforward_plus_rl_residual',
                    residual_norm=float(np.linalg.norm(self.last_residual)),
                    nominal_saturated=self.nominal_saturated,
                    combined_saturated=self.combined_saturated)
        return info

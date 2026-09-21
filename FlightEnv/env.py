"""Gymnasium environment for direct motor control of a quadrotor."""
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from FlightEnv.quadrotor import Physics, Quadrotor
from FlightEnv.trajectories import sample_trajectory


class FlightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, render=False, mode="trajectory", verbose=False,
                 hard_reset=False, record=False,
                 reward_state_pos_xy_weight=1, reward_state_pos_z_weight=1,
                 reward_state_vel_weight=0.02, reward_state_attitude_weight=0.5,
                 reward_state_ang_vel_weight=0.01, reward_action_weight=0.001,
                 reward_exponential=True, is_domain_randomization=True,
                 goal_horizon=5, last_horizon=1, episode_len_sec=5, goal_stride=1,
                 ctrl_freq=200, pybullet_freq=600, fov_img_freq=10,
                 physics=Physics.PYB_DRAG, drone_model="250",
                 flight_urdf_root=None, render_mode=None):
        super().__init__()
        if ctrl_freq <= 0 or pybullet_freq < ctrl_freq or pybullet_freq % ctrl_freq:
            raise ValueError("pybullet_freq must be a positive integer multiple of ctrl_freq")
        if episode_len_sec <= 0 or not np.isclose(episode_len_sec * ctrl_freq,
                                                 round(episode_len_sec * ctrl_freq)):
            raise ValueError("episode_len_sec * ctrl_freq must be a positive integer")
        if round(episode_len_sec * ctrl_freq) < 1:
            raise ValueError("An episode must contain at least one control step")
        if any(not isinstance(h, int) or h < 0 for h in (goal_horizon, last_horizon)):
            raise ValueError("Observation horizons must be nonnegative integers")
        if not isinstance(goal_stride, int) or goal_stride < 1:
            raise ValueError("goal_stride must be a positive integer")
        self._goal_stride = goal_stride
        mode = mode.lower()
        if mode not in ("trajectory", "hover"):
            raise ValueError("mode must be 'trajectory' or 'hover'")
        self.render_mode = render_mode or ("human" if render else None)
        if self.render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")
        self._physics = Physics(physics)
        if self._physics not in (Physics.PYB, Physics.PYB_DRAG):
            raise ValueError("Only PYB and PYB_DRAG physics are implemented")
        if fov_img_freq <= 0 or fov_img_freq > ctrl_freq:
            raise ValueError("fov_img_freq must be in (0, ctrl_freq]")
        root = Path(flight_urdf_root) if flight_urdf_root else Path(__file__).parent / "assets"
        self._urdf_path = str((root / f"{drone_model}.urdf").resolve())
        if not Path(self._urdf_path).is_file():
            raise FileNotFoundError(self._urdf_path)
        self.GRAVITY_ACC = 9.8
        self.GROUND_PLANE_Z = -0.05
        self.RENDER_HEIGHT, self.RENDER_WIDTH = 360, 480
        self.metadata = dict(self.metadata, render_fps=ctrl_freq)
        self._verbose = verbose
        self._hard_reset = hard_reset
        self._is_domain_randomization = is_domain_randomization
        self._mode = mode
        self._goal_horizon, self._last_horizon = goal_horizon, last_horizon
        self._episode_len_sec = episode_len_sec
        self._time_step, self._pybullet_time_step = 1. / ctrl_freq, 1. / pybullet_freq
        self._pybullet_steps_per_ctrl = int(pybullet_freq // ctrl_freq)
        self.max_episode_steps = int(round(episode_len_sec * ctrl_freq))
        self.action_dim, self.state_dim = 4, 12
        self.observation_dim = self.state_dim * (1 + goal_horizon + last_horizon)
        # Zero action is nominal hover; policy outputs are independent of motor units.
        self.action_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
        # Terminal observations may exceed flight limits. Limits belong in termination.
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                (self.observation_dim,), dtype=np.float32)
        self._reward_state_Q = np.diag([
            reward_state_pos_xy_weight, reward_state_vel_weight,
            reward_state_pos_xy_weight, reward_state_vel_weight,
            reward_state_pos_z_weight, reward_state_vel_weight,
            reward_state_attitude_weight, reward_state_attitude_weight,
            reward_state_attitude_weight,
            reward_state_ang_vel_weight, reward_state_ang_vel_weight,
            reward_state_ang_vel_weight,
        ])
        self._reward_action_weight = reward_action_weight
        self._reward_exponential = reward_exponential
        self._is_record = record
        self._record_interval = max(1, round(ctrl_freq / fov_img_freq))
        if record:
            self.ONBOARD_IMG_PATH = Path("recording_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            self.ONBOARD_IMG_PATH.mkdir(parents=True, exist_ok=True)
        self._viewer = None
        if self.render_mode == "human":
            self.PYB_CLIENT = p.connect(p.GUI, options="--width=1280 --height=800 "
                                       "--background_color_red=0.035 --background_color_green=0.055 "
                                       "--background_color_blue=0.09")
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)
        if self.PYB_CLIENT < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.quadrotor = None
        self._episode_index = -1
        try:
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.PYB_CLIENT < 0:
            raise RuntimeError("Cannot reset a closed environment")
        if self._viewer is not None:
            self._viewer.clear()
        if self._hard_reset or self.quadrotor is None:
            p.resetSimulation(physicsClientId=self.PYB_CLIENT)
            p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
            p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
            p.setTimeStep(self._pybullet_time_step, physicsClientId=self.PYB_CLIENT)
            self._ground_id = p.loadURDF(str(Path(pybullet_data.getDataPath()) / "plane.urdf"),
                                         [0, 0, self.GROUND_PLANE_Z], physicsClientId=self.PYB_CLIENT)
            self.quadrotor = Quadrotor(self.PYB_CLIENT, self._urdf_path,
                                       self._pybullet_steps_per_ctrl,
                                       is_domain_randomization=self._is_domain_randomization,
                                       physics_type=self._physics, verbose=self._verbose,
                                       rng=self.np_random)
        else:
            self.quadrotor.rng = self.np_random
            self.quadrotor.reset(reload_urdf=False)
        pos, vel, _ = self._make_reference()
        self.state_goal = np.zeros((len(pos), self.state_dim))
        self.state_goal[:, [0, 2, 4]] = pos
        self.state_goal[:, [1, 3, 5]] = vel
        self.action_goal = np.full(4, self.quadrotor.MASS * self.GRAVITY_ACC / 4)
        self.action_bounds = np.array(self._set_action())
        if not np.all((self.action_goal > self.action_bounds[0]) &
                      (self.action_goal < self.action_bounds[1])):
            raise ValueError("Motor thrust limits do not allow hovering")
        self._env_step_counter = 0
        self._out_of_bounds = self._collision = self._episode_done = False
        self._episode_index += 1
        self._reset_initial_pose()
        self._update_state()
        self._last_state_queue = np.tile(self._state, (self._last_horizon, 1))
        if self.render_mode == "human":
            if self._viewer is None:
                from FlightEnv.visualization import TrajectoryViewer
                self._viewer = TrajectoryViewer(self.PYB_CLIENT)
            self._viewer.reset(pos, self.quadrotor.pos, self._ground_id, self._episode_len_sec)
        return self._get_observation(), self._get_info()

    def _reset_initial_pose(self):
        pass

    def _make_reference(self):
        return sample_trajectory(self._episode_len_sec, self._time_step,
                                 self.np_random, self._mode)

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("Episode ended; call reset() before step()")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must contain four finite motor commands")
        action = np.clip(action, -1., 1.)
        if self._last_horizon:
            self._last_state_queue = np.roll(self._last_state_queue, -1, axis=0)
            self._last_state_queue[-1] = self._state
        thrust = self._action_to_thrust(action)
        self.quadrotor.step(self.quadrotor.thrust2rpm(thrust))
        self._env_step_counter += 1
        self._update_state()
        obs = self._get_observation()
        self._out_of_bounds = bool(np.any(np.abs(self._state[[0, 2]]) > 15)
                                   or self._state[4] < self.GROUND_PLANE_Z
                                   or self._state[4] > 15
                                   or np.any(np.abs(self._state[6:8]) > np.deg2rad(85)))
        self._collision = bool(p.getContactPoints(self.quadrotor.my_quadrotor, self._ground_id,
                                                  physicsClientId=self.PYB_CLIENT))
        terminated = self._out_of_bounds or self._collision
        truncated = bool(self._env_step_counter >= self.max_episode_steps and not terminated)
        self._episode_done = terminated or truncated
        reward = self._get_reward(action) - (30. if terminated else 0.)
        if self._viewer is not None:
            self._viewer.update(self.quadrotor.pos,
                                self.state_goal[self._env_step_counter, [0, 2, 4]],
                                self._env_step_counter * self._time_step,
                                self._get_info(), self._episode_done)
        if self._is_record and self._env_step_counter % self._record_interval == 0:
            frame = self.render()
            Image.fromarray(frame).save(self.ONBOARD_IMG_PATH /
                                       f"episode_{self._episode_index:04d}_{self._env_step_counter:06d}.png")
        return obs, float(reward), terminated, truncated, self._get_info()

    def _action_to_thrust(self, action):
        span = np.where(action >= 0., self.action_bounds[1] - self.action_goal,
                        self.action_goal - self.action_bounds[0])
        return self.action_goal + action * span

    def _preprocess_action(self, action):
        return self.quadrotor.thrust2rpm(self._action_to_thrust(np.clip(action, -1., 1.)))

    def _set_action(self):
        q = self.quadrotor
        low = q.KF * (q.PWM2RPM_SCALE * q.MIN_PWM + q.PWM2RPM_CONST)**2
        high = q.KF * (q.PWM2RPM_SCALE * q.MAX_PWM + q.PWM2RPM_CONST)**2
        return np.full(4, low), np.full(4, high)

    def _update_state(self):
        q = self.quadrotor
        rotation = np.asarray(p.getMatrixFromQuaternion(q.quat)).reshape(3, 3)
        self._state = np.concatenate(([q.pos[0], q.vel[0], q.pos[1], q.vel[1],
                                       q.pos[2], q.vel[2]], q.rpy, rotation.T @ q.ang_vel))

    def _get_observation(self):
        indices = np.minimum(self._env_step_counter + self._goal_stride * (1 + np.arange(self._goal_horizon)),
                             len(self.state_goal) - 1)
        return np.concatenate((self._last_state_queue.ravel(), self._state,
                               self.state_goal[indices].ravel())).astype(np.float32)

    def _get_reward(self, action):
        error = self._state - self.state_goal[self._env_step_counter]
        error[6:9] = (error[6:9] + np.pi) % (2 * np.pi) - np.pi
        cost = error @ self._reward_state_Q @ error + self._reward_action_weight * np.sum(action**2)
        # A bounded positive survival reward avoids incentivizing early crashes to
        # escape accumulating large negative tracking costs.
        return float(np.exp(-cost) if self._reward_exponential else -cost)

    def _get_info(self):
        goal = self.state_goal[self._env_step_counter]
        return {"out_of_bounds": self._out_of_bounds, "collision": self._collision,
                "position_error": float(np.linalg.norm(self._state[[0, 2, 4]] - goal[[0, 2, 4]])),
                "velocity_error": float(np.linalg.norm(self._state[[1, 3, 5]] - goal[[1, 3, 5]])),
                "elapsed_seconds": self._env_step_counter * self._time_step}

    def render(self):
        view = p.computeViewMatrixFromYawPitchRoll(self.quadrotor.pos, 1.5, 35, -25, 0, 2)
        projection = p.computeProjectionMatrixFOV(60, self.RENDER_WIDTH / self.RENDER_HEIGHT, 0.1, 100)
        _, _, rgba, _, _ = p.getCameraImage(self.RENDER_WIDTH, self.RENDER_HEIGHT,
                                            viewMatrix=view, projectionMatrix=projection,
                                            renderer=p.ER_TINY_RENDERER,
                                            physicsClientId=self.PYB_CLIENT)
        return np.asarray(rgba, dtype=np.uint8).reshape(self.RENDER_HEIGHT, self.RENDER_WIDTH, 4)[:, :, :3]

    def close(self):
        if getattr(self, "PYB_CLIENT", -1) >= 0:
            if p.isConnected(self.PYB_CLIENT):
                p.disconnect(self.PYB_CLIENT)
            self.PYB_CLIENT = -1

import pybullet
import pybullet_data

import gymnasium as gym
from gymnasium.utils import seeding

from enum import Enum
import os
import numpy as np
import xml.etree.ElementTree as etxml
import math
from copy import deepcopy
import time

from FlightEnv.quadrotor import Quadrotor
from FlightEnv.gen_traj import generate_trajectory
from FlightEnv.gen_line import generate_line
from FlightEnv.optim_gen_traj import load_trajectory

class Physics(str, Enum):
    '''Physics implementations enumeration class.'''

    PYB = 'pyb'  # Base PyBullet physics update.
    DYN = 'dyn'  # Update with an explicit model of the dynamics.
    PYB_GND = 'pyb_gnd'  # PyBullet physics update with ground effect.
    PYB_DRAG = 'pyb_drag'  # PyBullet physics update with drag.
    PYB_DW = 'pyb_dw'  # PyBullet physics update with downwash.
    PYB_GND_DRAG_DW = 'pyb_gnd_drag_dw'  # PyBullet physics update with ground effect, drag, and downwash.


class FlightEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}
    
    def __init__(self, render=False, mode="Hover",
                 verbose=False, 
                 hard_reset=True,
                 # reward_state_weight should be a diagonal matrix
                 reward_state_pos_weight=1.0,
                 reward_state_vel_weight=0.01,
                 reward_state_attitude_weight=0.5,
                 reward_state_ang_vel_weight=0.01,
                 reward_action_weight=0.0001,
                 reward_constraint_pos_radius=0.5,
                 reward_constraint_pos_penalty=1.0,
                 reward_exponential=True,
                 goal_horizon=50,
                 episode_len_sec=10,
                 ctrl_freq = 50,
                 pybullet_freq = 240,
                 physics: Physics = Physics.PYB,
                 drone_model ='cf2x',
                 flight_urdf_root="FlightEnv/assets"):
        super(FlightEnv, self).__init__()
        # Constants.
        self.GRAVITY_ACC = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.GROUND_PLANE_Z = -0.05

        self.RENDER_HEIGHT = 360
        self.RENDER_WIDTH = 480
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30

        self._verbose = verbose
        self._urdf_path = os.path.join(flight_urdf_root, drone_model + '.urdf')

        self._env_step_counter = 0
        self._is_render = render
        self._mode = mode
        self._physics = Physics(physics)
        self._time_step = 1. / ctrl_freq
        self._pybullet_time_step = 1. / pybullet_freq

        if self._is_render:
            self.PYB_CLIENT = pybullet.connect(pybullet.GUI)
        else:
            self.PYB_CLIENT = pybullet.connect(pybullet.DIRECT)

        self._hard_reset = True

        # Set the goal horizon.
        self._goal_horizon = goal_horizon
        self._episode_len_sec = episode_len_sec

        self.action_dim = 4
        self.state_dim = 12
        self.observation_dim = 12 * (1 + self._goal_horizon)

        seed = self.seed()
        self.reset(seed=seed)

        self._reward_state_Q = np.diag([
            reward_state_pos_weight, reward_state_pos_weight, reward_state_pos_weight,
            reward_state_vel_weight, reward_state_vel_weight, reward_state_vel_weight,
            reward_state_attitude_weight, reward_state_attitude_weight, reward_state_attitude_weight,
            reward_state_ang_vel_weight, reward_state_ang_vel_weight, reward_state_ang_vel_weight
        ])
        self.reward_constraint_pos_radius = reward_constraint_pos_radius
        self.reward_constraint_pos_penalty = reward_constraint_pos_penalty

        self._reward_action_weight = reward_action_weight
        self._reward_exponential = reward_exponential

        # Action space: 4 motors thrusts 
        action_low, action_high = self._set_action()
        self.action_bounds = np.array([action_low, action_high])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # State space: 12 states (no goal horizon)
        # Observation space: 12 states * horizon = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}        
        self._state = np.zeros(self.state_dim)
        self._pos_threshold = [15, 15, self.GROUND_PLANE_Z+15]
        
        observation_low, observation_high = self._set_observation()
        self._state_space_low = observation_low
        self._state_space_high = observation_high

        if self._goal_horizon > 0:
            mul = 1 + self._goal_horizon
            observation_low = np.concatenate([observation_low] * mul)
            observation_high = np.concatenate([observation_high] * mul)

        self.observation_space = gym.spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        self._observation = np.zeros(self.observation_dim)
        self._norm_observation = np.zeros(self.observation_dim)

        self._hard_reset = hard_reset


    def _generate_trajectory(self, episode_len_sec, sample_time):  
        """
        TODO: Generate a trajectory for the quadrotor to follow,
        add velocity and acceleration bounds.
        """ 
        # return generate_trajectory(episode_len_sec=episode_len_sec, sample_time=sample_time)
        return generate_line(episode_len_sec=episode_len_sec, sample_time=sample_time)
        # average_speed = 0.4
        # pos, vel, _ = load_trajectory(episode_len_sec=episode_len_sec, sample_time=sample_time, average_speed=average_speed)
        # return pos, vel

    def reset(self, seed=None, options=None):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0, physicsClientId=self.PYB_CLIENT)        
        if self._hard_reset:
            pybullet.resetSimulation(physicsClientId=self.PYB_CLIENT)
            pybullet.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
            pybullet.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
            pybullet.setTimeStep(self._pybullet_time_step, physicsClientId=self.PYB_CLIENT)
            self._ground_id = pybullet.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath(), [0, 0, self.GROUND_PLANE_Z]
                                                , physicsClientId=self.PYB_CLIENT)

            self.quadrotor = Quadrotor(pybullet_client=self.PYB_CLIENT, urdf_path=self._urdf_path
                                       , time_step=self._time_step, verbose=self._verbose)
            
            
            self.quadrotor.load_model_param()


        self.quadrotor.reset(reload_urdf=False)

        pybullet.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0],
                                            physicsClientId=self.PYB_CLIENT)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1, physicsClientId=self.PYB_CLIENT)  
        
        # Set goal state and action
        pos_ref, vel_ref = self._generate_trajectory(self._episode_len_sec, self._time_step)
        self.state_goal = np.vstack([
            pos_ref[:, 0],
            vel_ref[:, 0],
            pos_ref[:, 1],
            vel_ref[:, 1],
            pos_ref[:, 2],
            vel_ref[:, 2],
            np.zeros(pos_ref.shape[0]),
            np.zeros(pos_ref.shape[0]),
            np.zeros(pos_ref.shape[0]),
            np.zeros(vel_ref.shape[0]),
            np.zeros(vel_ref.shape[0]),
            np.zeros(vel_ref.shape[0])
        ]).transpose()
        self.action_goal = np.ones(self.action_dim) * self.quadrotor.MASS * self.GRAVITY_ACC / self.action_dim
        
        self._env_step_counter = 0
        self._out_of_bounds = False

        if self._hard_reset:
            if self._is_render:
                # User debug draw failed
                self._debug_line()
                

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
        action: A list of motor rpms.

        Returns:
          observations: 
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.
        """
        if self._env_step_counter == 0:
            action = self.action_goal
        raw_action = action
        rpm = self._preprocess_action(action)
        self.quadrotor.step(rpm)
        self._env_step_counter += 1
        # print(self._env_step_counter)
        obs = self._get_observation()
        reward = self._get_reward(raw_action)
        terminated = self._get_ternimated()
        info = self._get_info()
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.quadrotor.get_base_position()
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.PYB_CLIENT)
        proj_matrix = pybullet.computeProjectionMatrixFOV(fov=60,
                                                          aspect=float(self.RENDER_WIDTH) /
                                                          self.RENDER_HEIGHT,
                                                          nearVal=0.1,
                                                          farVal=100.0,
                                                          physicsClientId=self.PYB_CLIENT)
        (_, _, px, _, _) = pybullet.getCameraImage(width=self.RENDER_WIDTH,
                                                    height=self.RENDER_HEIGHT,
                                                    viewMatrix=view_matrix,
                                                    projectionMatrix=proj_matrix,
                                                    renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                                                    physicsClientId=self.PYB_CLIENT)
        
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        pass

    def _debug_line(self):
        for i in range(self.state_goal.shape[0]-1):
                    pybullet.addUserDebugLine([self.state_goal[i, 0], self.state_goal[i, 2], self.state_goal[i, 4]],
                                            [self.state_goal[i+1, 0], self.state_goal[i+1, 2], self.state_goal[i+1, 4]],
                                            lineColorRGB=[1, 0, 0], lineWidth=1,
                                            physicsClientId=self.PYB_CLIENT)

    def _get_observation(self):
        R_wb = np.array(pybullet.getMatrixFromQuaternion(self.quadrotor.quat)).reshape(3, 3)
        R_bw = R_wb.T
        ang_vel_b = R_bw @ self.quadrotor.ang_vel
        self._state = np.hstack([self.quadrotor.pos[0], self.quadrotor.vel[0],
                                       self.quadrotor.pos[1], self.quadrotor.vel[1],
                                       self.quadrotor.pos[2], self.quadrotor.vel[2],
                                       self.quadrotor.rpy, ang_vel_b]).reshape((self.state_dim,))
        # extend observation with horizon
        obs = deepcopy(self._state)
        if self._goal_horizon > 0:
            wp_idx = [
                min(self._env_step_counter + 1 + i, self.state_goal.shape[0] - 1)
                for i in range(self._goal_horizon)
            ]
            goal_state = self.state_goal[wp_idx].flatten()
            obs = np.concatenate([obs, goal_state])

        self._observation = obs
        return obs

    def _get_reward(self, raw_action):
        action = np.asarray(raw_action)
        action_error = action - self.action_goal
        wp_idx = min(self._env_step_counter, self.state_goal.shape[0] - 1)
        state_error = self._state - self.state_goal[wp_idx]
        dist = np.sum(state_error @ self._reward_state_Q @ state_error) + self._reward_action_weight * np.sum(action_error**2)
        reward = -dist
        # constraint penalty
        # pos constraint: radius penalty
        pos = np.array([self._state[0], self._state[2], self._state[4]])
        goal_pos = np.array([self.state_goal[wp_idx, 0], self.state_goal[wp_idx, 2], self.state_goal[wp_idx, 4]])
        last_goal_pos = np.array([self.state_goal[wp_idx - 1, 0], self.state_goal[wp_idx - 1, 2], self.state_goal[wp_idx - 1, 4]])
        adj_pos = np.linalg.norm(last_goal_pos - goal_pos)
        self.reward_constraint_pos_radius = max(self.reward_constraint_pos_radius, adj_pos * self._goal_horizon * 2)
        if np.linalg.norm(pos - goal_pos) > self.reward_constraint_pos_radius:
            reward -= self.reward_constraint_pos_penalty
            
        if self._reward_exponential:
            reward = np.exp(reward)
        return reward

    def _get_info(self):
        info = {}
        info['out_of_bounds'] = self._out_of_bounds 

        return info
    
    def _get_ternimated(self):
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
        terminate = np.logical_or(self._state < self._state_space_low
                                  , self._state > self._state_space_high)
        terminated = np.any(np.logical_and(terminate, mask))
        if terminated:
            self._out_of_bounds = True
        if self._env_step_counter >= self._episode_len_sec / self._time_step:
            if not terminated:
                self._out_of_bounds = False
            terminated = True
        return terminated
    
    def seed(self, seed=None):
        """
        Waited to add disturbances
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _preprocess_action(self, action):
        """
        Preprocess the action to fit the quadrotor's action space.
        """
        # TODO: apply disturbance in action 

        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        rpm = self.quadrotor.thrust2rpm(action)

        return rpm
        

    def _set_action(self):
        """
        Return action bounds.
        """
        action_low = self.quadrotor.KF * (self.quadrotor.PWM2RPM_SCALE * self.quadrotor.MIN_PWM + self.quadrotor.PWM2RPM_CONST)**2
        action_high = self.quadrotor.KF * (self.quadrotor.PWM2RPM_SCALE * self.quadrotor.MAX_PWM + self.quadrotor.PWM2RPM_CONST)**2
        return np.full(self.action_dim, action_low, np.float32), np.full(self.action_dim, action_high, np.float32)

    def _set_observation(self):
        x_threshold = self._pos_threshold[0]
        y_threshold = self._pos_threshold[1]
        z_threshold = self._pos_threshold[2]
        phi_threshold_radians = 85 * math.pi / 180
        theta_threshold_radians = 85 * math.pi / 180
        psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.

        observation_low = np.array([
                -x_threshold, -np.finfo(np.float32).max,
                -y_threshold, -np.finfo(np.float32).max,
                self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
                -phi_threshold_radians, -theta_threshold_radians, -psi_threshold_radians,
                -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max
            ])
        
        observation_high = np.array([
                x_threshold, np.finfo(np.float32).max,
                y_threshold, np.finfo(np.float32).max,
                z_threshold, np.finfo(np.float32).max,
                phi_threshold_radians, theta_threshold_radians, psi_threshold_radians,
                np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
            ])
        
        
        return observation_low, observation_high

    
    
        
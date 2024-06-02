
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
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

from FlightEnv.quadrotor import Quadrotor, Physics
from FlightEnv.gen_traj import generate_trajectory
from FlightEnv.gen_line import generate_line, generate_uniform_line, generate_hover
from FlightEnv.optim_gen_traj import load_trajectory

class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0     # Red, green, blue (and alpha)
    DEP = 1     # Depth
    SEG = 2     # Segmentation by object id
    BW = 3      # Black and white

class FlightEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}
    
    def __init__(self, render=False, mode="Hover",
                 verbose=False, 
                 hard_reset=True,
                 record=False,
                 # reward_state_weight should be a diagonal matrix
                 reward_state_pos_xy_weight=1,
                 reward_state_pos_z_weight=1,
                 reward_state_vel_weight=0.02,
                 reward_state_attitude_weight=0.5,
                 reward_state_ang_vel_weight=0.01,
                 reward_action_thrust_weight=1,
                 reward_action_rpy_weight=0.001,
                 reward_exponential=False,
                 reward_last_action=10,
                 is_domain_randomization=True,
                 goal_horizon=5,
                 last_horizon=1,
                 episode_len_sec=5,
                 ctrl_freq = 200,
                 pybullet_freq = 600,
                 fov_img_freq = 10,
                 physics: Physics = Physics.PYB_DRAG,
                #  drone_model ='cf2x',
                 drone_model ='250',
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
        self._is_record = record
        self._mode = mode
        self._physics = Physics(physics)
        self._time_step = 1. / ctrl_freq
        self._pybullet_time_step = 1. / pybullet_freq
        self._pybullet_steps_per_ctrl = int(pybullet_freq / ctrl_freq)
        self._fov_img_freq = fov_img_freq

        self._is_domain_randomization = is_domain_randomization

        if self._is_render:
            self.PYB_CLIENT = pybullet.connect(pybullet.GUI)
        else:
            self.PYB_CLIENT = pybullet.connect(pybullet.DIRECT)

        if self._is_record:
            self.ONBOARD_IMG_PATH = "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
            os.makedirs(self.ONBOARD_IMG_PATH, exist_ok=True)

        self._hard_reset = True

        # Set the goal horizon.
        self._goal_horizon = goal_horizon
        self._last_horizon = last_horizon
        self._episode_len_sec = episode_len_sec

        # self.action_dim = 4
        #### euler action space ####
        self.action_dim = 4
        #### euler action space ####

        self.state_dim = 12
        self.observation_dim = 12 * (1 + self._goal_horizon)

        if self._last_horizon > 0:
            self._last_state_queue = np.zeros((self._last_horizon, self.state_dim))

        self._depth_img = None
        self._rgb_img = None
        self._seg_img = None

        seed = self.seed()
        self.reset(seed=seed)

        self._reward_state_Q = np.diag([
            reward_state_pos_xy_weight, reward_state_vel_weight, 
            reward_state_pos_xy_weight, reward_state_vel_weight, 
            reward_state_pos_z_weight, reward_state_vel_weight,
            reward_state_attitude_weight, reward_state_attitude_weight, reward_state_attitude_weight,
            reward_state_ang_vel_weight, reward_state_ang_vel_weight, reward_state_ang_vel_weight
        ])

        self._reward_action_Q = np.diag([
            reward_action_thrust_weight,
            reward_action_rpy_weight,
            reward_action_rpy_weight,
            reward_action_rpy_weight
        ])
        self._reward_exponential = reward_exponential

        action_low, action_high = self._set_action()
        self.action_bounds = np.array([action_low, action_high])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self._last_action = np.array([self.quadrotor.MASS * self.GRAVITY_ACC, 0, 0, 0])
        self._reward_last_action = reward_last_action

        # State space: 12 states (no goal horizon)
        # Observation space: 12 states * horizon = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}        
        self._state = np.zeros(self.state_dim)
        self._pos_threshold = [15, 15, self.GROUND_PLANE_Z+15]
        self._vel_threshold = [5, 5, 5]
        
        observation_low, observation_high = self._set_observation()
        self._state_space_low = observation_low
        self._state_space_high = observation_high

        mul = 1 + self._goal_horizon + self._last_horizon
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
        # return generate_line(episode_len_sec=episode_len_sec, sample_time=sample_time)
        # return generate_uniform_line(episode_len_sec=episode_len_sec, sample_time=sample_time)
        # return generate_hover(episode_len_sec=episode_len_sec, sample_time=sample_time)
        average_speed = 0.4
        pos, vel, acc = load_trajectory(episode_len_sec=episode_len_sec, sample_time=sample_time, average_speed=average_speed, num_files=1000)
        return pos, vel, acc

    def reset(self, seed=None, options=None):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0, physicsClientId=self.PYB_CLIENT)        
        if self._hard_reset:
            pybullet.resetSimulation(physicsClientId=self.PYB_CLIENT)
            pybullet.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
            pybullet.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
            pybullet.setTimeStep(self._pybullet_time_step, physicsClientId=self.PYB_CLIENT)
            self._ground_id = pybullet.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath(), [0, 0, self.GROUND_PLANE_Z]
                                                , physicsClientId=self.PYB_CLIENT)

            self.quadrotor = Quadrotor(pybullet_client=self.PYB_CLIENT, urdf_path=self._urdf_path,
                                       pybullet_steps_per_ctrl=self._pybullet_steps_per_ctrl, 
                                       is_domain_randomization=self._is_domain_randomization, 
                                       physics_type=Physics.PYB_DRAG, verbose=self._verbose)

        self.quadrotor.reset(reload_urdf=False)

        pybullet.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0],
                                            physicsClientId=self.PYB_CLIENT)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1, physicsClientId=self.PYB_CLIENT)  
        
        # Set goal state and action
        pos_ref, vel_ref, acc_ref = self._generate_trajectory(self._episode_len_sec, self._time_step)
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
        # self.action_goal = np.ones(self.action_dim) * self.quadrotor.MASS * self.GRAVITY_ACC / self.action_dim
        #### euler action space ####
        self.action_goal = np.array([self.quadrotor.MASS * self.GRAVITY_ACC, 0, 0, 0])
        #### euler action space ####
        
        self._env_step_counter = 0
        self._out_of_bounds = False

        if self._hard_reset:
            if self._is_render:
                # self._debug_line()
                self._debug_state = np.array([]).reshape(0, self.state_dim)
                self._debug_reward = []
                

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
        raw_action = action
        # action = np.ones(self.action_dim) * self.quadrotor.MASS * self.GRAVITY_ACC / self.action_dim
        
        #### euler action space ####
        # action = np.array([0., -1.4835298, 0.19656561, 0.27414256])
        # action = np.array([1.1 * self.quadrotor.MASS * self.GRAVITY_ACC, 0., 0., 0.])
        #### euler action space ####
        # print(f"mg: {self.quadrotor.MASS * self.GRAVITY_ACC}")
        # print(f"action: {action}")
        # print(f"last action: {self._last_action}")
        rpm = self._preprocess_action(action)
        # print(f"rpm: {rpm}")
        self.quadrotor.step(rpm)
        self._env_step_counter += 1
        # print(self._env_step_counter)
        obs = self._get_observation()
        reward = self._get_reward(raw_action)
        terminated = self._get_ternimated()
        if self._out_of_bounds:
            reward -= 30
        info = self._get_info()
        truncated = False
        if self._is_render:
            self._debug_state = np.append(self._debug_state, self._state)
            self._debug_reward.append(reward)

        # self._rgb_img, self._depth_img, self._seg_img = self._get_fovimgs()
        if self._is_record:
            self._exportImage(img_type=ImageType.DEP, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self._depth_img,
                                    path=self.ONBOARD_IMG_PATH,
                                    frame_num=int(self._env_step_counter/self._fov_img_freq)
                                    )
        # print("reward: ", reward)
        self._last_action = action
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
        # plot the goal trajectory in matplotlib

    def _get_fovimgs(self, segmentation=False):
        rot_mat = np.array(pybullet.getMatrixFromQuaternion(self.quadrotor.quat)).reshape(3, 3)
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(self.quadrotor.pos)
        DRONE_CAM_VIEW = pybullet.computeViewMatrix(cameraEyePosition=self.quadrotor.pos+np.array([0, 0, self.quadrotor.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.PYB_CLIENT
                                             )
        DRONE_CAM_PRO =  pybullet.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.quadrotor.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else pybullet.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = pybullet.getCameraImage(width=self.RENDER_WIDTH,
                                                 height=self.RENDER_HEIGHT,
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.PYB_CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg
    
    def _exportImage(self,
                     img_type: ImageType,
                     img_input,
                     path: str,
                     frame_num: int=0
                     ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'), 'RGBA')).save(os.path.join(path,"frame_"+str(frame_num)+".png"))
        elif img_type == ImageType.DEP:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.SEG:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype('uint8')
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(os.path.join(path,"frame_"+str(frame_num)+".png"))

    def _get_observation(self):
        self._state = np.hstack([self.quadrotor.pos[0], self.quadrotor.vel[0],
                                       self.quadrotor.pos[1], self.quadrotor.vel[1],
                                       self.quadrotor.pos[2], self.quadrotor.vel[2],
                                       self.quadrotor.rpy, self.quadrotor.rpy_vel]).reshape((self.state_dim,))
        # print(np.array(self.quadrotor.rpy)*180/np.pi)
        # extend observation with horizon
        obs = deepcopy(self._state)
        if self._goal_horizon > 0:
            wp_idx = [
                min(self._env_step_counter + 1 + i, self.state_goal.shape[0] - 1)
                for i in range(self._goal_horizon)
            ]
            goal_state = self.state_goal[wp_idx].flatten()
            obs = np.concatenate([obs, goal_state])

        if self._last_horizon > 0:
            last_state = self._last_state_queue[-self._last_horizon:].flatten()
            obs = np.concatenate([last_state, obs])

        self._last_state_queue = np.roll(self._last_state_queue, -1, axis=0)
        self._last_state_queue[-1] = self._state

        return obs

    def _get_reward(self, raw_action):
        action = np.asarray(raw_action)
        action_error = action - self.action_goal
        wp_idx = min(self._env_step_counter, self.state_goal.shape[0] - 1)
        state_error = self._state - self.state_goal[wp_idx]
        # print("step: ", self._env_step_counter)
        # print("state: ", self._state)
        # print("goal: ", self.state_goal[wp_idx])
        dist = np.sum(state_error @ self._reward_state_Q @ state_error) + np.sum(action_error @ self._reward_action_Q @ action_error)
        reward = -dist

        # last action reward
        if self._env_step_counter > 0:
            action_diff = raw_action - self._last_action
            reward -= self._reward_last_action * np.sum(action_diff**2)

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
        # if self._env_step_counter >= 100:
            if not terminated:
                self._out_of_bounds = False
            terminated = True
        if terminated and self._is_render:

            #### euler action space ####
            # plot action with env step
            self.quadrotor.plot_euler_and_vel()
            #### euler action space ####

            self._debug_fig, (ax, axv) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

            # Plotting on the first subplot (ax)
            ax.plot(self.state_goal[:, 0], self.state_goal[:, 2], self.state_goal[:, 4], label='Trajectory')
            state_traj = np.array(self._debug_state).reshape(-1, self.state_dim)
            ax.plot(state_traj[:, 0], state_traj[:, 2], state_traj[:, 4], label='State Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            # Plotting on the second subplot (axv)
            # plot with (timestamp, vX, vY)
            axv.plot(np.arange(len(self.state_goal)), self.state_goal[:, 1], self.state_goal[:, 3], label='Velocity')
            axv.plot(np.arange(len(state_traj)), state_traj[:, 1], state_traj[:, 3], label='State Velocity')
            axv.set_xlabel('step')
            axv.set_ylabel('vX')
            axv.set_zlabel('vY')
            axv.legend()

            plt.show()


            # plot reward with env step
            plt.plot(self._debug_reward)
            plt.xlabel('Env Step')
            plt.ylabel('Reward')
            plt.show()

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

        #### euler action space ####
        action = self.quadrotor.euler_step(action)
        #### euler action space ####

        rpm = self.quadrotor.thrust2rpm(action)

        return rpm
        

    def _set_action(self):
        """
        Return action bounds.
        """
        # action_low = self.quadrotor.KF * (self.quadrotor.PWM2RPM_SCALE * self.quadrotor.MIN_PWM + self.quadrotor.PWM2RPM_CONST)**2
        # action_high = self.quadrotor.KF * (self.quadrotor.PWM2RPM_SCALE * self.quadrotor.MAX_PWM + self.quadrotor.PWM2RPM_CONST)**2
        # return np.full(self.action_dim, action_low, np.float32), np.full(self.action_dim, action_high, np.float32)

        #### euler action space ####
        total_thrust_threshold = self.quadrotor.MASS * self.GRAVITY_ACC * 5
        phi_threshold_radians = 85 * math.pi / 180
        theta_threshold_radians = 85 * math.pi / 180
        psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.
        action_low = np.array([0, 
                               -phi_threshold_radians, -theta_threshold_radians, -psi_threshold_radians])
        action_high = np.array([25., 
                                phi_threshold_radians, theta_threshold_radians, psi_threshold_radians])
        return action_low, action_high
        #### euler action space ####




    def _set_observation(self):
        x_threshold = self._pos_threshold[0]
        y_threshold = self._pos_threshold[1]
        z_threshold = self._pos_threshold[2]
        vx_threshold = self._vel_threshold[0]
        vy_threshold = self._vel_threshold[1]
        vz_threshold = self._vel_threshold[2]
        phi_threshold_radians = 85 * math.pi / 180
        theta_threshold_radians = 85 * math.pi / 180
        psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.

        observation_low = np.array([
                -x_threshold, -vx_threshold,
                -y_threshold, -vy_threshold,
                self.GROUND_PLANE_Z, -vz_threshold,
                -phi_threshold_radians, -theta_threshold_radians, -psi_threshold_radians,
                -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max
            ])
        
        observation_high = np.array([
                x_threshold, vx_threshold,
                y_threshold, vy_threshold,
                z_threshold, vz_threshold,
                phi_threshold_radians, theta_threshold_radians, psi_threshold_radians,
                np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
            ])
        
        
        return observation_low, observation_high

    
    
        
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

from FlightEnv.quadrotor import Quadrotor

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
                 goal_horizon=0,
                 freq = 50,
                 physics: Physics = Physics.PYB,
                 drone_model ='cf2x',
                 flight_urdf_root="FlightEnv/assets"):
        super(FlightEnv, self).__init__()
        # Constants.
        self.GRAVITY_ACC = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.GROUND_PLANE_Z = -0.05

        self._verbose = verbose
        self._urdf_path = os.path.join(flight_urdf_root, drone_model + '.urdf')

        self._env_step_counter = 0
        self._render = render
        self._mode = mode
        self._time_step = 1. / freq

        if self._render:
            self.PYB_CLIENT = pybullet.connect(pybullet.GUI)
        else:
            self.PYB_CLIENT = pybullet.connect(pybullet.DIRECT)

        self._hard_reset = True

        self.seed()
        self.reset()

        self._physics = Physics(physics)
        self.quadrotor.load_model_param()

        # Action space: 4 motors rpm
        self.action_dim = 4
        action_low, action_high = self._set_action()
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation space: 12 states = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}        
        observation_low, observation_high = self._set_observation()
        self.observation_space = gym.spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        self.observation_dim = self.observation_space.shape[0]
        self._observation = np.zeros(self.observation_dim)
        self._norm_observation = np.zeros(self.observation_dim)

        # State space: goal_horizon * 12 states = {observation, goal_horizon1, goal_horizon2, ...} 
        state_low = observation_low
        state_high = observation_high
        if goal_horizon > 0:
            mul = goal_horizon + 1
            state_low = np.concatenate([observation_low] * mul)
            state_high = np.concatenate([observation_high] * mul)
        
        self.state_space = gym.spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        self.state_dim = self.state_space.shape[0]

        self._hard_reset = hard_reset

    def reset(self):        
        if self._hard_reset:
            pybullet.resetSimulation(physicsClientId=self.PYB_CLIENT)
            pybullet.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
            pybullet.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
            pybullet.setTimeStep(self._time_step, physicsClientId=self.PYB_CLIENT)
            self._ground_id = pybullet.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath(), [0, 0, self.GROUND_PLANE_Z]
                                                , physicsClientId=self.PYB_CLIENT)

            self.quadrotor = Quadrotor(pybullet_client=self.PYB_CLIENT, urdf_path=self._urdf_path
                                       , time_step=self._time_step, verbose=self._verbose)

    def step(self, action):
        pass

    def _get_observation(self):
        R_wb = pybullet.getMatrixFromQuaternion(self.quadrotor.quat).reshape(3, 3)
        R_bw = R_wb.T
        ang_vel_b = R_bw @ self.quadrotor.ang_vel
        self._observation = np.hstack([self.quadrotor.pos[0], self.quadrotor.vel[0],
                                       self.quadrotor.pos[1], self.quadrotor.vel[1],
                                       self.quadrotor.pos[2], self.quadrotor.vel[2],
                                       self.quadrotor.rpy, ang_vel_b]).reshape((self.observation_dim,))
        
        obs = deepcopy(self._observation)
        # needed to add horizon to get self.state

    def seed(self, seed=None):
        """
        Waited to add disturbances
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _set_action(self):
        """
        Return action bounds.
        """
        action_low = self.KF * (self.quadrotor.PWM2RPM_SCALE * self.quadrotor.MIN_PWM + self.quadrotor.PWM2RPM_CONST)**2
        action_high = self.KF * (self.quadrotor.PWM2RPM_SCALE * self.quadrotor.MAX_PWM + self.quadrotor.PWM2RPM_CONST)**2
        return np.full(self.action_dim, action_low, np.float32), np.full(self.action_dim, action_high, np.float32)

    def _set_observation(self):
        x_threshold = 2
        y_threshold = 2
        z_threshold = 2
        phi_threshold_radians = 85 * math.pi / 180
        theta_threshold_radians = 85 * math.pi / 180
        psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.

        observation_low = np.array([
                -x_threshold, -np.finfo(np.float32).max,
                -y_threshold, -np.finfo(np.float32).max,
                self.quadrotor.GROUND_PLANE_Z, -np.finfo(np.float32).max,
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

    
    
        
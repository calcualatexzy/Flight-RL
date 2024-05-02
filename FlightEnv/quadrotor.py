from enum import Enum
import os
import numpy as np
import xml.etree.ElementTree as etxml
import math
import pybullet

class Quadrotor:
    def __init__(self, pybullet_client, urdf_path, pybullet_steps_per_ctrl, verbose=False):
        self.PYB_CLIENT = pybullet_client
        self._urdf_path = urdf_path
        self._pybullet_steps_per_ctrl = pybullet_steps_per_ctrl

        self._verbose = verbose
        
        self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                 'phi', 'theta', 'psi', 'p', 'q', 'r']
        self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                                'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        
        self.INIT_XYZ = np.array([0, 0, 0.5])
        self.INIT_VEL = np.array([0, 0, 0])
        self.INIT_ANG_VEL = np.array([0, 0, 0])
        self.INIT_RPY = np.array([0, 0, 0])
        
        self.pos = np.zeros(3)
        self.quat = np.zeros(4)
        self.rpy = np.zeros(3)
        self.vel = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.last_clipped_action = np.zeros(4)

        self._step_counter = 0

        self.reset(reload_urdf=True)

    def reset(self, reload_urdf=True):
        self.load_model_param()
        if reload_urdf:
            self.my_quadrotor = pybullet.loadURDF(self._urdf_path, self.INIT_XYZ, 
                                                  pybullet.getQuaternionFromEuler(self.INIT_RPY), 
                                                  physicsClientId=self.PYB_CLIENT)

            pybullet.changeDynamics(self.my_quadrotor, -1, linearDamping=0, angularDamping=0, mass=self.MASS, localInertiaDiagonal=self.J.diagonal())

            self._update_and_store_kinematic_information()
        else:
            pybullet.resetBasePositionAndOrientation(self.my_quadrotor, self.INIT_XYZ, 
                                                     pybullet.getQuaternionFromEuler(self.INIT_RPY), 
                                                     physicsClientId=self.PYB_CLIENT)
            pybullet.resetBaseVelocity(self.my_quadrotor, self.INIT_VEL, self.INIT_ANG_VEL, 
                                      physicsClientId=self.PYB_CLIENT)

            self._update_and_store_kinematic_information()

        self._step_counter = 0
        self.last_clipped_action = np.zeros(4)

    def step(self, action):
        for _ in range(self._pybullet_steps_per_ctrl):
            self._physics(action)
            pybullet.stepSimulation(physicsClientId=self.PYB_CLIENT)
        self._update_and_store_kinematic_information()
        self._step_counter += 1

    def _physics(self, rpm):
        '''Base PyBullet physics implementation.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.
        '''
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            pybullet.applyExternalForce(self.my_quadrotor,
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=pybullet.LINK_FRAME,
                                 physicsClientId=self.PYB_CLIENT)
        pybullet.applyExternalTorque(self.my_quadrotor,
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=pybullet.LINK_FRAME,
                              physicsClientId=self.PYB_CLIENT)

    def _update_and_store_kinematic_information(self):
        self.pos, self.quat = pybullet.getBasePositionAndOrientation(self.my_quadrotor, physicsClientId=self.PYB_CLIENT)

        self.rpy = pybullet.getEulerFromQuaternion(self.quat)

        self.vel, self.ang_vel = pybullet.getBaseVelocity(self.my_quadrotor, physicsClientId=self.PYB_CLIENT)
    
    def get_base_position(self):
        pos, _ = pybullet.getBasePositionAndOrientation(self.my_quadrotor, physicsClientId=self.PYB_CLIENT)
        return pos
    
    def thrust2rpm(self, thrust):
        thrust = np.clip(thrust, np.zeros_like(thrust), None) 
        return np.sqrt(thrust / self.KF)

    def load_model_param(self):
        # Constants.
        self.GRAVITY_ACC = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        # Load the drone properties from the .urdf file.
        self.MASS, \
            self.L, \
            self.THRUST2WEIGHT_RATIO, \
            self.J, \
            self.J_INV, \
            self.KF, \
            self.KM, \
            self.COLLISION_H, \
            self.COLLISION_R, \
            self.COLLISION_Z_OFFSET, \
            self.MAX_SPEED_KMH, \
            self.GND_EFF_COEFF, \
            self.PROP_RADIUS, \
            self.DRAG_COEFF, \
            self.DW_COEFF_1, \
            self.DW_COEFF_2, \
            self.DW_COEFF_3, \
            self.PWM2RPM_SCALE, \
            self.PWM2RPM_CONST, \
            self.MIN_PWM, \
            self.MAX_PWM = self._parse_urdf_parameters(self._urdf_path)
        print("扰动前m", self.MASS)
        print("扰动前J", self.J)
        # 添加随机扰动
        self.MASS += np.random.uniform(0.005, 0.005)
        self.J += np.random.uniform(-0.000005, 0.000005, size=self.J.shape)
        print("扰动后m", self.MASS)
        print("扰动后J", self.J)
        if self._verbose:
            print(
                '[INFO] BaseAviary.__init__() loaded parameters from the drone\'s .urdf: \
                \n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f}, \
                \n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f}, \
                \n[INFO] gnd_eff_coeff {:f}, prop_radius {:f}, \
                \n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f}, \
                \n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f} \
                \n[INFO] pwm2rpm_scale {:f}, pwm2rpm_const {:f}, min_pwm {:f}, max_pwm {:f}'
                .format(self.MASS, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2],
                        self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                        self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS,
                        self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1,
                        self.DW_COEFF_2, self.DW_COEFF_3, self.PWM2RPM_SCALE,
                        self.PWM2RPM_CONST, self.MIN_PWM, self.MAX_PWM))
            
        # Compute constants.
        self.GRAVITY = self.GRAVITY_ACC * self.MASS
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)

    def _parse_urdf_parameters(self, file_name):
        '''Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        '''
        URDF_TREE = etxml.parse(file_name).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        PWM2RPM_SCALE = float(URDF_TREE[0].attrib['pwm2rpm_scale'])
        PWM2RPM_CONST = float(URDF_TREE[0].attrib['pwm2rpm_const'])
        MIN_PWM = float(URDF_TREE[0].attrib['pwm_min'])
        MAX_PWM = float(URDF_TREE[0].attrib['pwm_max'])
        
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, \
            PWM2RPM_SCALE, PWM2RPM_CONST, MIN_PWM, MAX_PWM
    
    
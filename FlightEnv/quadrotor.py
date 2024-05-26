from enum import Enum
import os
import numpy as np
import xml.etree.ElementTree as etxml
import math
import pybullet
import matplotlib.pyplot as plt

class Physics(str, Enum):
    '''Physics implementations enumeration class.'''

    PYB = 'pyb'  # Base PyBullet physics update.
    DYN = 'dyn'  # Update with an explicit model of the dynamics.
    PYB_GND = 'pyb_gnd'  # PyBullet physics update with ground effect.
    PYB_DRAG = 'pyb_drag'  # PyBullet physics update with drag.
    PYB_DW = 'pyb_dw'  # PyBullet physics update with downwash.
    PYB_GND_DRAG_DW = 'pyb_gnd_drag_dw'  # PyBullet physics update with ground effect, drag, and downwash.
class Quadrotor:
    def __init__(self, pybullet_client, urdf_path, pybullet_steps_per_ctrl, is_domain_randomization=False, 
                 physics_type=Physics.PYB,
                 verbose=False):
        self.PYB_CLIENT = pybullet_client
        self._urdf_path = urdf_path
        self._pybullet_steps_per_ctrl = pybullet_steps_per_ctrl

        self._verbose = verbose
        
        self._physics_type = physics_type

        self._is_domain_randomization = is_domain_randomization
        
        self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                 'phi', 'theta', 'psi', 'p', 'q', 'r']
        self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                                'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        
        self.INIT_XYZ = np.array([0., 0., 0.5])
        self.INIT_VEL = np.array([0., 0., 0.])
        self.INIT_ANG_VEL = np.array([0., 0., 0.])
        self.INIT_RPY = np.array([0., 0., 0.])

        if self._is_domain_randomization:
            self.INIT_XYZ += np.random.uniform(-0.1, 0.1, size=3)
            self.INIT_VEL += np.random.uniform(-0.1, 0.1, size=3)
            self.INIT_ANG_VEL += np.random.uniform(-0.1, 0.1, size=3)
            self.INIT_RPY += np.random.uniform(-0.1, 0.1, size=3)

        self.pos = np.zeros(3)
        self.quat = np.zeros(4)
        self.rpy = np.zeros(3)
        self.vel = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.rpy_vel = np.zeros(3)
        self.last_action = np.zeros(4)

        #### euler action space ####
        # roll, pitch, yaw in DEGREE
        # self.euler_Kp = np.array([8.0, 8.0, 4.0])
        self.euler_Kp = np.array([10.0, 10.0, 4.0])

        self.euler_vel_Kp = np.array([0.35, 0.35, -0.2]) * 180
        self.euler_vel_Kd = np.array([0.004, 0.004, -0.002]) * 10
        # self.euler_vel_Ki = np.array([0.1, 0.1, -0.1])
        self.euler_vel_Ki = np.array([0.1, 0.1, -0.1])
        self.euler_vel_K = np.array([0.7, 0.7, 0.5])
        self.euler_vel_MAX = np.array([1600.0, 1600.0, 1000.0])
        self.euler_vel_integral = 0.0
        self.euler_vel_integral_LIM = np.array([0.3, 0.3, 0.3])
        self.euler_vel_prev_error = 0.0

        self.euler_log = []
        self.euler_vel_log = []
        #### euler action space ####

        self._step_counter = 0

        self.reset(reload_urdf=True)

    def reset(self, reload_urdf=True):
        if reload_urdf:

            self.load_model_param()
            self.my_quadrotor = pybullet.loadURDF(self._urdf_path, self.INIT_XYZ, 
                                                  pybullet.getQuaternionFromEuler(self.INIT_RPY), 
                                                  physicsClientId=self.PYB_CLIENT)

            pybullet.changeDynamics(self.my_quadrotor, -1, linearDamping=0, angularDamping=0, 
                                    mass=self.MASS, 
                                    localInertiaDiagonal=self.J.diagonal())

            self._update_and_store_kinematic_information()
        else:
            pybullet.resetBasePositionAndOrientation(self.my_quadrotor, self.INIT_XYZ, 
                                                     pybullet.getQuaternionFromEuler(self.INIT_RPY), 
                                                     physicsClientId=self.PYB_CLIENT)
            pybullet.resetBaseVelocity(self.my_quadrotor, self.INIT_VEL, self.INIT_ANG_VEL, 
                                      physicsClientId=self.PYB_CLIENT)

            self._update_and_store_kinematic_information()

        self._step_counter = 0
        self.last_action = np.zeros(4)

    def step(self, action):
        for _ in range(self._pybullet_steps_per_ctrl):
            self._physics(action)
            if self._physics_type == Physics.PYB_DRAG:
                self._drag(action)
            pybullet.stepSimulation(physicsClientId=self.PYB_CLIENT)
            self.last_action = action
        self._update_and_store_kinematic_information()
        self._step_counter += 1

    def euler_step(self, action):
        R_wb = np.array(pybullet.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        R_bw = R_wb.T
        u1 = R_bw @ np.array([0, 0, self.MASS * (self.GRAVITY_ACC + action[0])])
        u1 = u1[2]
        euler_c = action[1:4]
        euler_vel_c = self._euler_pid(euler_c, self.rpy)
        euler_acc_c = self._euler_vel_pid(euler_vel_c, self.rpy_vel) * self.DEG2RAD
        # euler_acc_c = self._euler_vel_pid(euler_vel_c, self.rpy_vel)
        u2 = np.dot(self.J, euler_acc_c) + np.cross(self.rpy_vel, np.dot(self.J, self.rpy_vel))
        thrust = self._calculate_motor_thrusts(u1, u2[0], u2[1], u2[2])
        # thrust = self._calculate_motor_thrusts(u1, euler_acc_c[0], euler_acc_c[1], euler_acc_c[2])

        rpy_vel = np.array(self.rpy_vel)
        self.euler_log.append([self.rpy, euler_c])
        self.euler_vel_log.append([rpy_vel * self.RAD2DEG, euler_vel_c])
        return thrust

    def _euler_pid(self, euler_c, rpy):
        rpy = np.array(rpy)
        e = (euler_c - rpy) * self.RAD2DEG
        return np.clip(self.euler_Kp * e, -self.euler_vel_MAX, self.euler_vel_MAX)

    def _euler_vel_pid(self, euler_vel_c, rpy_vel):
        rpy_vel = np.array(rpy_vel)
        error = euler_vel_c - (rpy_vel * self.RAD2DEG)
        # error = euler_vel_c * self.DEG2RAD - rpy_vel
        self.euler_vel_integral += error
        # self.euler_vel_integral = np.clip(self.euler_vel_integral, -self.euler_vel_integral_LIM, self.euler_vel_integral_LIM)
        derivative = error - self.euler_vel_prev_error
        self.euler_vel_prev_error = error
        # return self.euler_vel_K * (self.euler_vel_Kp * error + self.euler_vel_Ki * self.euler_vel_integral + self.euler_vel_Kd * derivative)
        return np.clip(self.euler_vel_K * (self.euler_vel_Kp * error + self.euler_vel_Ki * self.euler_vel_integral + self.euler_vel_Kd * derivative), -self.euler_vel_MAX, self.euler_vel_MAX)

    def _calculate_motor_thrusts(self, total_thrust, roll_control, pitch_control, yaw_control):
        # u1 = KF * (rpm1**2 + rpm2**2 + rpm3**2 + rpm4**2)
        # u2[0] = L * KF * (rpm2**2 - rpm4**2)
        # u2[1] = L * KF * (rpm3**2 - rpm1**2)
        # u2[2] = KM * (rpm1**2 - rpm2**2 + rpm3**2 - rpm4**2)

        l = self.L
        mix_matrix = np.array([
            [1,  1,  1,  1],
            [0,  l,  0, -l],
            [-l, 0,  l,  0],
            [self.KM/self.KF, -self.KM/self.KF, self.KM/self.KF, -self.KM/self.KF]
        ])
        thrust_vector = np.array([total_thrust, roll_control, pitch_control, yaw_control])

        motor_thrusts = np.dot(np.linalg.inv(mix_matrix), thrust_vector)

        return motor_thrusts
    
    def plot_euler_and_vel(self):
        rpy_log = np.array([log[0] for log in self.euler_log])
        euler_c_log = np.array([log[1] for log in self.euler_log])
        
        # Extract rpy_vel and euler_vel_c from euler_vel_log
        rpy_vel_log = np.array([log[0] for log in self.euler_vel_log])
        euler_vel_c_log = np.array([log[1] for log in self.euler_vel_log])

        time_steps = range(len(rpy_log))

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Plot rpy
        axs[0, 0].plot(time_steps, rpy_log[:, 0], label='RPY X')
        axs[0, 0].plot(time_steps, rpy_log[:, 1], label='RPY Y')
        axs[0, 0].plot(time_steps, rpy_log[:, 2], label='RPY Z')
        axs[0, 0].set_title('RPY Over Time')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('RPY (degrees)')
        axs[0, 0].legend()

        # Plot euler_c
        axs[0, 1].plot(time_steps, euler_c_log[:, 0], label='Euler Angle Command X')
        axs[0, 1].plot(time_steps, euler_c_log[:, 1], label='Euler Angle Command Y')
        axs[0, 1].plot(time_steps, euler_c_log[:, 2], label='Euler Angle Command Z')
        axs[0, 1].set_title('Euler Angle Commands Over Time')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Euler Angle Commands (degrees)')
        axs[0, 1].legend()

        # Plot rpy_vel
        axs[1, 0].plot(time_steps, rpy_vel_log[:, 0], label='Angular Velocity X')
        axs[1, 0].plot(time_steps, rpy_vel_log[:, 1], label='Angular Velocity Y')
        axs[1, 0].plot(time_steps, rpy_vel_log[:, 2], label='Angular Velocity Z')
        axs[1, 0].set_title('Angular Velocities Over Time')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Angular Velocities (degrees/sec)')
        axs[1, 0].legend()

        # Plot euler_vel_c
        axs[1, 1].plot(time_steps, euler_vel_c_log[:, 0], label='Euler Angular Velocity Command X')
        axs[1, 1].plot(time_steps, euler_vel_c_log[:, 1], label='Euler Angular Velocity Command Y')
        axs[1, 1].plot(time_steps, euler_vel_c_log[:, 2], label='Euler Angular Velocity Command Z')
        axs[1, 1].set_title('Euler Angular Velocity Commands Over Time')
        axs[1, 1].set_xlabel('Time Step')
        axs[1, 1].set_ylabel('Euler Angular Velocity Commands (degrees/sec)')
        axs[1, 1].legend()

        # Plot error ( euler_vel_c - rpy_vel )
        error = euler_vel_c_log - rpy_vel_log
        fig, ax = plt.subplots()
        ax.plot(time_steps, error[:, 0], label='Error X')
        ax.plot(time_steps, error[:, 1], label='Error Y')
        ax.plot(time_steps, error[:, 2], label='Error Z')
        ax.set_title('Error Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Error (degrees/sec)')
        ax.legend()
        

        plt.tight_layout()
        plt.show()

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
        
    def _drag(self, rpm):
        '''PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Have to use last action to compute drag.

        Args:
            rpm (ndarray): (4)-shaped array of ints containing the RPMs values of the 4 motors.
            nth_drone (int): The ordinal number/position of the desired drone in list self.DRONE_IDS.

        '''
        # Rotation matrix of the base.
        base_rot = np.array(pybullet.getMatrixFromQuaternion(
            self.quat)).reshape(3, 3)
        # Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(
            np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot, drag_factors * np.array(self.vel))
        pybullet.applyExternalForce(self.my_quadrotor,
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=pybullet.LINK_FRAME,
                             physicsClientId=self.PYB_CLIENT)
        
    def _ground_effect(self, rpm):
        pass


    def _update_and_store_kinematic_information(self):
        self.pos, self.quat = pybullet.getBasePositionAndOrientation(self.my_quadrotor, physicsClientId=self.PYB_CLIENT)

        self.rpy = pybullet.getEulerFromQuaternion(self.quat)

        self.vel, self.ang_vel = pybullet.getBaseVelocity(self.my_quadrotor, physicsClientId=self.PYB_CLIENT)

        R_wb = np.array(pybullet.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        R_bw = R_wb.T
        self.rpy_vel = R_bw @ self.ang_vel
    
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

        # domain randomization
        if self._is_domain_randomization:
            self.MASS += np.random.uniform(-2e-3, 2e-3)
            self.J += np.random.uniform(-5e-6, 5e-6, size=self.J.shape)

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
    
    
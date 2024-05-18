import numpy as np
import math
from scipy.optimize import minimize
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from FlightEnv.trajCheck import TrajCheck
def bezier_curve(t, control_points):
    """
    Evaluate a point on the Bezier curve given parameter t and control points.
    """
    n = len(control_points) - 1
    point = np.zeros_like(control_points[0])
    for i in range(n + 1):
        point += control_points[i] * (math.factorial(n) / (math.factorial(i) * math.factorial(n - i))) * (t ** i) * ((1 - t) ** (n - i))
    return point

def trajectory(control_points, eval_points, total_time):
    """
    Compute trajectory using Bezier curve with constrained start and end points,
    and bounded velocity and acceleration.
    """
    num_control_points = len(control_points)
    num_eval_points = eval_points
    
    def objective(x):
        # Reshape the optimization variables into control points format
        control_points_opt = np.array(x).reshape(num_control_points, -1)
        # Compute the trajectory points
        trajectory_points = np.array([bezier_curve(t, control_points_opt) for t in np.linspace(0, 1, num_eval_points)])
        # Compute the squared differences between consecutive points for velocity
        velocity_diffs = np.diff(trajectory_points, axis=0)
        # Compute the squared differences between consecutive velocity differences for acceleration
        acceleration_diffs = np.diff(velocity_diffs, axis=0)
        # Objective function is the sum of squared acceleration differences
        # start and end velocities are zero
        penalty = 1000
        penalty_z = 0.1
        return np.sum(acceleration_diffs ** 2) + penalty * (velocity_diffs[0] ** 2).sum() + penalty * (velocity_diffs[-1] ** 2).sum() + penalty_z * (velocity_diffs[:, 2] ** 2).sum()
    
    # Define constraints to ensure start and end points are constrained
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[:3] - control_points[0]},  # Start point constraint
        {'type': 'eq', 'fun': lambda x: x[-3:] - control_points[-1]},  # End point constraint
        ]
    
    # Bounds for optimization (None means unbounded)
    bounds = [(None, None)] * (3 * num_control_points)  # Unbounded for all control points

    x0 = np.array(control_points).flatten()
    
    # Perform optimization
    result = minimize(objective, x0, method='SLSQP', constraints=constraints, bounds=bounds)
    
    if not result.success:
        raise ValueError("Optimization failed to converge")
    
    # Reshape optimized control points
    control_points_opt = np.array(result.x).reshape(num_control_points, -1)
    
    # Compute trajectory points
    trajectory_points = np.array([bezier_curve(t, control_points_opt) for t in np.linspace(0, 1, num_eval_points)])
    
    # Compute velocities
    velocities = np.diff(trajectory_points, axis=0) / (total_time / (num_eval_points - 1))
    velocities = np.concatenate((velocities, velocities[-1][np.newaxis]), axis=0)
    
    # Compute accelerations
    accelerations = np.diff(velocities, axis=0) / (total_time / (num_eval_points - 1))
    # the last acceleration is the same as the second last
    accelerations[-1] = accelerations[-2]
    accelerations = np.concatenate((accelerations, accelerations[-2][np.newaxis]), axis=0)
    
    return trajectory_points, velocities, accelerations

def generate_trajectory(episode_len_sec, sample_time, average_speed):
    max_position = episode_len_sec * average_speed
    min_position = 0.1
    # Example control points
    # TrajLib/traj_200Hz_len5_vel0.4: control_points_num = 4
    # TrajLib/traj_200Hz_len5_vel1.5: control_points_num = 6
    control_points_num = 6
    control_points = np.random.uniform(low=0, high=max_position, size=(control_points_num, 3))
    # generate random ending point
    # start and end points
    control_points[0] = [0, 0, 0.5]
    # control_points[-1] = [max_position, max_position, 0.8]
    control_points[-1] = [np.random.choice([np.random.uniform(-max_position, -min_position), np.random.uniform(min_position, max_position)]),
                            np.random.choice([np.random.uniform(-max_position, -min_position), np.random.uniform(min_position, max_position)]),
                            np.random.uniform(low=0.3, high=0.7)]
    # control_points[-1] = [np.random.uniform(low=-max_position, high=max_position), np.random.uniform(low=-max_position, high=max_position), np.random.uniform(low=0.3, high=0.7)]
    
    # wang lijie add
    points = control_points.copy()
    data = points.flatten().tolist()
    data.remove(data[0])
    data.remove(data[0])
    data.remove(data[0])
    eval_points = int(episode_len_sec / sample_time)
    trajectory_points, velocities, accelerations = trajectory(control_points, eval_points, episode_len_sec)


    return trajectory_points, velocities, accelerations, data

def plot_trajectory(trajectory_points, velocities, accelerations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], label='Trajectory')
    ax.plot(trajectory_points[::20, 0], trajectory_points[::20, 1], trajectory_points[::20, 2], 'ro')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Velocity plot
    fig, ax = plt.subplots()
    ax.plot(velocities[:, 0], label='X Velocity')
    ax.plot(velocities[:, 1], label='Y Velocity')
    ax.plot(velocities[:, 2], label='Z Velocity')
    ax.legend()

    # Acceleration plot
    fig, ax = plt.subplots()
    ax.plot(accelerations[:, 0], label='X Acceleration')
    ax.plot(accelerations[:, 1], label='Y Acceleration')
    ax.plot(accelerations[:, 2], label='Z Acceleration')
    ax.legend()

    plt.show()

def save_trajectory(cnt, trajectory_points, velocities, accelerations):
    # Save the trajectory points, velocities, and accelerations to a file, reading them back in dictionary format
    print(cnt)
    #np.savez(f'FlightEnv/TrajLib/traj_200Hz_len5_vel1.5/{cnt}.npz', trajectory_points=trajectory_points, velocities=velocities, accelerations=accelerations)
    np.savez(f'FlightEnv/TrajLib/traj_200Hz_len5_vel1.5/{cnt}.npz', trajectory_points=trajectory_points, velocities=velocities, accelerations=accelerations)

def load_trajectory(episode_len_sec, sample_time, average_speed, num_files=100):
    # from the episode length and sample time, get the folder name
    freq = int(1/sample_time)
    folder_name = f'FlightEnv/TrajLib/traj_{freq}Hz_len{episode_len_sec}_vel{average_speed}/'
    file_num = np.random.randint(num_files)
    # file_num = 0
    data = np.load(f'{folder_name}{file_num}.npz')
    trajectory_points = data['trajectory_points']
    velocities = data['velocities']
    accelerations = data['accelerations']
    return trajectory_points, velocities, accelerations

if __name__ == "__main__":
    episode_len_sec = 5
    sample_time = 0.005
    average_speed = 1.5
    checkData = []
    totalNum = 100
    cnt = 0
    checkingLine = 8
    num = 6
    # # start_time = time.time()
    # for i in range(100):
    #     trajectory_points, velocities, accelerations, data = generate_trajectory(episode_len_sec, sample_time, average_speed)
    #     save_trajectory(i, trajectory_points, velocities, accelerations)
    # # print("Time taken: ", time.time() - start_time)


    # trajectory_points, velocities, accelerations, data = generate_trajectory(episode_len_sec, sample_time, average_speed)
    # # Plotting
    # print(trajectory_points.shape, velocities.shape, accelerations.shape)
    # plot_trajectory(trajectory_points, velocities, accelerations)

    # Loading
    # load_trajectory(0)
    
    while True:
        if cnt == totalNum:
            break
        else:
            if cnt < checkingLine:
                print("trajs not enough, Generating!!!")
                trajectory_points, velocities, accelerations, data = generate_trajectory(episode_len_sec, sample_time, average_speed)
                save_trajectory(cnt+200, trajectory_points, velocities, accelerations)
                checkData.append(data)
                cnt += 1
                if cnt == checkingLine:
                    trajCheck = TrajCheck(checkData)
                    trajCheck.updateMeanVector()
                    trajCheck.generateCovMatrix()
                    trajCheck.computeEigenvalues()
            elif cnt >= checkingLine:
                print("长度", len(trajCheck.checkData))
                print("Checking!!!")
                trajectory_points, velocities, accelerations, data = generate_trajectory(episode_len_sec, sample_time, average_speed)
                trajCheck.updateMeanVectorAndCovMatrix(data)
                trajCheck.computeEigenvalues()
                if trajCheck.jugdeDivergence(num):
                    print("contributions", trajCheck.computeContributions())
                    print ("Divergence!!!")
                    save_trajectory(cnt+200, trajectory_points, velocities, accelerations)
                    cnt += 1
                    if num < 9:
                        num += 1
                else:
                    print("contributions", trajCheck.computeContributions())
                    trajCheck.checkData.remove(trajCheck.checkData[-1])
                    trajCheck.updateMeanVector()
                    trajCheck.generateCovMatrix()
                    print ("Convergence!!!")
                    continue
          
                





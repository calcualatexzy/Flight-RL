import numpy as np
import math
from scipy.optimize import minimize

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
        return np.sum(acceleration_diffs ** 2)
    
    # Define constraints to ensure start and end points are constrained
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[:3] - control_points[0]},  # Start point constraint
        {'type': 'eq', 'fun': lambda x: x[-3:] - control_points[-1]},  # End point constraint
        # start and end velocities are zero
        {'type': 'eq', 'fun': lambda x: bezier_curve(0, x.reshape(num_control_points, -1)) - control_points[0]},
        {'type': 'eq', 'fun': lambda x: bezier_curve(1, x.reshape(num_control_points, -1)) - control_points[-1]}
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
    
    # Compute accelerations
    accelerations = np.diff(velocities, axis=0) / (total_time / (num_eval_points - 1))
    
    return trajectory_points, velocities, accelerations

def generate_trajectory(episode_len_sec, sample_time, average_speed):
    max_position = episode_len_sec * average_speed
    # Example control points
    control_points_num = 8
    control_points = np.random.uniform(low=0, high=max_position, size=(control_points_num, 3))
    # start and end points
    control_points[0] = [0, 0, 0]
    control_points[-1] = [max_position, max_position, max_position]
    eval_points = int(episode_len_sec / sample_time)
    trajectory_points, velocities, accelerations = trajectory(control_points, eval_points, episode_len_sec)


    return trajectory_points, velocities, accelerations

# Example usage
if __name__ == "__main__":
    episode_len_sec = 10
    sample_time = 0.01
    average_speed = 3
    trajectory_points, velocities, accelerations = generate_trajectory(episode_len_sec, sample_time, average_speed)
    # Plotting
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], label='Trajectory')
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
    plt.show()

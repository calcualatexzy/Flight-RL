import numpy as np
from scipy.special import comb

def bezier_curve(control_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    n = len(control_points) - 1
    curve = np.zeros((num_points, 3))
    
    for i, point in enumerate(control_points):
        curve += np.outer(comb(n, i) * np.power(t, i) * np.power(1 - t, n - i), point)
    
    return curve

def bezier_curve_derivative(control_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    n = len(control_points) - 1
    curve_derivative = np.zeros((num_points, 3))
    
    for i in range(3):
        for j in range(n):
            curve_derivative[:, i] += n * (control_points[j + 1][i] - control_points[j][i]) * \
                                        comb(n - 1, j) * np.power(t, j) * np.power(1 - t, n - 1 - j)
    
    return curve_derivative


def evaluate_bezier_trajectory(control_points, num_eval_points):
    
    # Interpolate points and velocity at each timestep
    timesteps = np.linspace(0, 1, num_eval_points)
    trajectory_at_timesteps = bezier_curve(control_points, num_eval_points)
    velocity_at_timesteps = bezier_curve_derivative(control_points, num_eval_points)
    
    return trajectory_at_timesteps, velocity_at_timesteps

# Example usage
num_control_points = 3  # Number of control points (including start and end points)
num_eval_points = 1000  # Number of evaluation points
max_position = 100  # Maximum position in each dimension (meters)

# Generate random start and end points
start_point = np.random.uniform(low=-max_position, high=max_position, size=(1, 3))
end_point = np.random.uniform(low=-max_position, high=max_position, size=(1, 3))

# Generate random control points for the middle
middle_control_points = np.random.uniform(low=-max_position, high=max_position, size=(num_control_points - 2, 3))

# Concatenate start, middle, and end points
control_points = np.vstack([start_point, middle_control_points, end_point])

trajectory_at_timesteps, velocity_at_timesteps = evaluate_bezier_trajectory(control_points, num_eval_points)

# Plot the trajectory and velocity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(trajectory_at_timesteps[:, 0], trajectory_at_timesteps[:, 1], trajectory_at_timesteps[:, 2], marker='o', label='Trajectory')
ax1.scatter(start_point[:, 0], start_point[:, 1], start_point[:, 2], color='red', label='Start Point')
ax1.scatter(end_point[:, 0], end_point[:, 1], end_point[:, 2], color='green', label='End Point')
ax1.set_title('Trajectory')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.plot(np.linalg.norm(velocity_at_timesteps, axis=1))
ax2.set_title('Velocity Magnitude')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Velocity')

plt.tight_layout()
plt.show()

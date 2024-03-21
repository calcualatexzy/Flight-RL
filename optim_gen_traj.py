import numpy as np
from scipy.optimize import minimize
from scipy.special import comb


# generate_random_traj(start, end, 3, 1000, vel_bound, acc_bound) -> points, velocities(1000)

# Define Bézier curve function
def bezier_curve(control_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    n = len(control_points) - 1
    curve = np.zeros((num_points, 3))

    for i, point in enumerate(control_points):
        curve += np.outer(comb(n, i) * np.power(t, i) * np.power(1 - t, n - i), point)

    return curve

# Define derivative of Bézier curve function
def bezier_curve_derivative(control_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    n = len(control_points) - 1
    curve_derivative = np.zeros((num_points, 3))

    for i in range(3):
        for j in range(n):
            curve_derivative[:, i] += n * (control_points[j + 1][i] - control_points[j][i]) * \
                                        comb(n - 1, j) * np.power(t, j) * np.power(1 - t, n - 1 - j)

    return curve_derivative

# Define objective function to minimize
def objective_function(control_points):
    # Compute trajectory points
    bezier_points = bezier_curve(control_points.reshape(-1, 3), num_points)

    # Compute velocity and acceleration
    velocity = bezier_curve_derivative(control_points.reshape(-1, 3), num_points)
    acceleration = bezier_curve_derivative(velocity, num_points)

    # Calculate total violation of velocity and acceleration bounds
    velocity_violation = np.maximum(np.linalg.norm(velocity, axis=1) - max_velocity, 0).sum()
    acceleration_violation = np.maximum(np.linalg.norm(acceleration, axis=1) - max_acceleration, 0).sum()

    # Return sum of violations (to be minimized)
    return velocity_violation + acceleration_violation

# Define constraints for optimization
constraints = [{'type': 'eq', 'fun': lambda control_points: control_points[0] - start_point},
               {'type': 'eq', 'fun': lambda control_points: control_points[-1] - end_point}]

# Maximum position in each dimension (meters)
max_position = 100 
# Generate random start and end points
start_point = np.array([0, 0, 0])
end_point = np.array([100, 100, 100])
# Number of control points (including start and end points)
num_control_points = 5  

# Number of points along the trajectory
num_points = 100  

# Number of evaluation points
num_eval_points = 1000  
 

# Maximum allowable velocity and acceleration
max_velocity = 10  
max_acceleration = 5  

# Generate random control points for the middle
middle_control_points = np.random.uniform(low=-max_position, high=max_position, size=(num_control_points - 2, 3))

# Concatenate start, middle, and end points
initial_guess_control_points = np.vstack([start_point, middle_control_points, end_point])

# Run optimization
result = minimize(objective_function, initial_guess_control_points.flatten(), constraints=constraints)

# Extract optimized control points
optimized_control_points = result.x.reshape(-1, 3)

# Generate final trajectory using optimized control points
final_trajectory, final_velocity = bezier_curve(optimized_control_points, num_eval_points), bezier_curve_derivative(optimized_control_points, num_eval_points)

# Plot the trajectory and velocity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(final_trajectory[:, 0], final_trajectory[:, 1], final_trajectory[:, 2], marker='o', label='Trajectory')
ax1.scatter(start_point[0], start_point[1], start_point[2], color='red', label='Start Point')
ax1.scatter(end_point[0], end_point[1], end_point[2], color='green', label='End Point')
ax1.set_title('Trajectory')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.plot(np.linalg.norm(final_velocity, axis=1))
ax2.set_title('Velocity Magnitude')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Velocity')

plt.tight_layout()
plt.show()

#Jack curve-system 杰克曲线系
import numpy as np
from scipy.optimize import minimize
from scipy.misc import derivative
from numpy import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function with adjustable coefficients
def f(t, a, b, c, d, e, f):
    return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)

# Define the loss function
def loss(x):
    def f_adjusted(t):
        return f(t, *x)
    df0 = derivative(f_adjusted, 0, dx=1e-6)
    df10 = derivative(f_adjusted, 10, dx=1e-6)
    return df0**2 + df10**2

t_values = np.linspace(0, 10, 1000)

# x0_x = np.array([5, 5, 1, 4, 3, 2])
# x0_y = np.array([1, 5, 7, 3, 2, 4])
# x0_z = np.array([10, 10, 10, 10, 9, 3])

# Use scipy's minimize function to find the optimal coefficients for x, y, z
# res_x = minimize(loss, x0_x)
# res_y = minimize(loss, x0_y)
# res_z = minimize(loss, x0_z)

# Define the functions with the optimal coefficients for x, y, z
def f_x(t, res):
    a, b, c, d, e, f = res
    return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)

def f_y(t, res):
    a, b, c, d, e, f = res
    return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)

def f_z(t, res):
    a, b, c, d, e, f = res
    return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)

# # Calculate the function values at these time points for x, y, z
# f_values = np.array([(f_x(t_values) - f_x(0))/f_x(10), (f_y(t_values) - f_y(0))/f_y(10), (f_z(t_values) - f_z(0))/f_z(10)])

# # Calculate the velocity values
# v_values = np.gradient(f_values, t_values, axis=1)

# # Calculate the acceleration values
# a_values = np.gradient(v_values, t_values, axis=1)


def generateJackCurveSystem():
    low = 1
    high = 10
    size = 6
    x0_x = np.random.uniform(low, high, size)
    x0_y = np.random.uniform(low, high, size)
    x0_z = np.random.uniform(low, high, size)
    res_x = minimize(loss, x0_x)
    res_y = minimize(loss, x0_y)
    res_z = minimize(loss, x0_z)
    f_values = np.array([(f_x(t_values, res_x.x) - f_x(0, res_x.x))/f_x(10, res_x.x), (f_y(t_values, res_y.x) - f_y(0, res_y.x))/f_y(10, res_y.x), (f_z(t_values, res_z.x) - f_z(0, res_z.x))/f_z(10, res_z.x)])
    v_values = np.gradient(f_values, t_values, axis=1)
    a_values = np.gradient(v_values, t_values, axis=1)
    # Store all points' positions and velocities in two lists
    positions = f_values.T
    velocities = v_values.T
    accelerations = a_values.T
    return positions, velocities, accelerations



# plot
# Plot the 3D trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(f_values[0], f_values[1], f_values[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Trajectory')

# Plot the speed magnitude over time
# plt.figure()
# v_magnitude = np.sqrt(np.sum(v_values**2, axis=0))
# plt.plot(t_values, v_magnitude)
# plt.xlabel('Time')
# plt.ylabel('Speed Magnitude')
# plt.title('Speed Magnitude over Time')
# plt.grid(True)
# plt.show()

# # Plot the acceleration magnitude over time
# plt.figure()
# a_magnitude = np.sqrt(np.sum(a_values**2, axis=0))
# plt.plot(t_values, a_magnitude)
# plt.xlabel('Time')
# plt.ylabel('Acceleration Magnitude')
# plt.title('Acceleration Magnitude over Time')
# plt.grid(True)
# plt.show()


# Store all points' positions and velocities in two lists
# positions = f_values.T.tolist()
# velocities = v_values.T.tolist()
# accelerations = a_values.T.tolist()
# print(positions)
# print(velocities)
# print(len(positions))
# print(len(velocities))

if __name__ == '__main__':
    positions, velocities, accelerations = generateJackCurveSystem()
    print(positions)
    print(velocities)
    print(accelerations)
    print(len(positions))
    print(len(velocities))
    print(len(accelerations))
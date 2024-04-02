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

# # Initial guess for the coefficients
# x0 = np.array([1, 1, 1, 1, 1, 1])
#
# # Use scipy's minimize function to find the optimal coefficients
# res = minimize(loss, x0)
#
# # Define the function with the optimal coefficients
# def f(t):
#     a, b, c, d, e, f = res.x
#     return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)
#
# # # Generate a series of time values
t_values = np.linspace(0, 10, 1000)
# #
# # # Calculate the function values at these time points
# # f_values = (f(t_values) - f(0))/f(10)
# #
# # # Plot the function
# # plt.plot(t_values, f_values)
# # plt.xlabel('Time')
# # plt.ylabel('Function Value')
# # plt.title('Plot of the Function')
# # plt.grid(True)
# # # Calculate the derivative of the function values
# # df_values = np.gradient(f_values, t_values)
# #
# # # Plot the derivative
# # plt.plot(t_values, df_values)
# # plt.xlabel('Time')
# # plt.ylabel('Derivative of Function Value')
# # plt.title('Plot of the Derivative of the Function')
# # plt.show()
# # Calculate the function values at these time points for x, y, z
# t_values = np.linspace(0, 10, 1000)
# f_values = np.array([(f(t_values) - f(0))/f(10) for _ in range(3)])
#
# # Calculate the velocity values
# v_values = np.gradient(f_values, t_values, axis=1)
#
# # Plot the 3D trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(f_values[0], f_values[1], f_values[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Trajectory')
#
# # Plot the speed magnitude over time
# plt.figure()
# v_magnitude = np.sqrt(np.sum(v_values**2, axis=0))
# plt.plot(t_values, v_magnitude)
# plt.xlabel('Time')
# plt.ylabel('Speed Magnitude')
# plt.title('Speed Magnitude over Time')
# plt.grid(True)
# plt.show()
#
# # Store all points' positions and velocities in two lists
# positions = f_values.T.tolist()
# velocities = v_values.T.tolist()
# Define the initial guesses for the coefficients for x, y, z
x0_x = np.array([5, 5, 1, 4, 3, 2])
x0_y = np.array([1, 5, 7, 3, 2, 4])
x0_z = np.array([3, 3, 3, 3, 3, 3])

# Use scipy's minimize function to find the optimal coefficients for x, y, z
res_x = minimize(loss, x0_x)
res_y = minimize(loss, x0_y)
res_z = minimize(loss, x0_z)

# Define the functions with the optimal coefficients for x, y, z
def f_x(t):
    a, b, c, d, e, f = res_x.x
    return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)

def f_y(t):
    a, b, c, d, e, f = res_y.x
    return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)

def f_z(t):
    a, b, c, d, e, f = res_z.x
    return a * (1.1 ** t) + b * t + c * (t ** 2) + d + e * np.sin(t) + f * np.cos(t)

# Calculate the function values at these time points for x, y, z
f_values = np.array([(f_x(t_values) - f_x(0))/f_x(10), (f_y(t_values) - f_y(0))/f_y(10), (f_z(t_values) - f_z(0))/f_z(10)])

# Calculate the velocity values
v_values = np.gradient(f_values, t_values, axis=1)

# Plot the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(f_values[0], f_values[1], f_values[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory')

# Plot the speed magnitude over time
plt.figure()
v_magnitude = np.sqrt(np.sum(v_values**2, axis=0))
plt.plot(t_values, v_magnitude)
plt.xlabel('Time')
plt.ylabel('Speed Magnitude')
plt.title('Speed Magnitude over Time')
plt.grid(True)
plt.show()

# Store all points' positions and velocities in two lists
positions = f_values.T.tolist()
velocities = v_values.T.tolist()
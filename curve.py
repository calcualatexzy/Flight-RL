import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_random_points(start, end, n):
    # Generate n random points between start and end
    points = np.random.uniform(start, end, (n, 3))
    # Sort the points by x-coordinate
    points = points[points[:,0].argsort()]
    return points

def generate_smooth_curve(start, end, n):
    # Generate random points
    points = generate_random_points(start, end, n)
    # Add the start and end points
    points = np.vstack((start, points, end))
    # Create a cubic spline for each dimension
    cs_x = CubicSpline(points[:,0], points[:,0])
    cs_y = CubicSpline(points[:,0], points[:,1])
    cs_z = CubicSpline(points[:,0], points[:,2])
    return cs_x, cs_y, cs_z

# Define the start and end points
start = np.array([0, 0, 0])
end = np.array([100, 100, 100])

# Generate a smooth curve
cs_x, cs_y, cs_z = generate_smooth_curve(start, end, 5)

# Plot the curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = np.linspace(start[0], end[0], 1000)
ax.plot(xs, cs_y(xs), cs_z(xs))
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def generate_line(episode_len_sec=10, sample_time=0.01):
    start_pos = np.array([0, 0, 0.5])
    end_pos = np.array([2, 2, 0.5])

    num_waypoints = int(episode_len_sec / sample_time)

    max_vel = 2 * np.linalg.norm(end_pos - start_pos) / episode_len_sec
    vel = np.linspace(0, max_vel, num_waypoints // 2)
    vel = np.concatenate((vel, vel[::-1]))

    time = np.linspace(0, 1, num_waypoints)[:, np.newaxis]
    waypoints = start_pos + time * (end_pos - start_pos)

    velocities = np.column_stack((vel, vel, np.zeros_like(vel)))
    
    return waypoints, velocities
import numpy as np
import matplotlib.pyplot as plt

def generate_line(episode_len_sec=5, sample_time=0.005):
    start_pos = np.array([0, 0, 0.5])
    end_pos = np.array([2, 2, 0.5])

    num_waypoints = int(episode_len_sec / sample_time)

    max_vel = 2. * 2. / episode_len_sec
    vel = np.linspace(0, max_vel, num_waypoints // 2)
    vel = np.concatenate((vel, vel[::-1]))

    acc = max_vel / (episode_len_sec / 2)
    acc = np.concatenate((acc * np.ones(num_waypoints // 2), -acc * np.ones(num_waypoints // 2)))

    time = np.linspace(0, 1, num_waypoints)[:, np.newaxis]
    waypoints = start_pos + time * (end_pos - start_pos)

    velocities = np.column_stack((vel, vel, np.zeros_like(vel)))
    accelerations = np.column_stack((acc, acc, np.zeros_like(acc)))

    eulers = np.zeros((num_waypoints, 3))
    # print("waypoints: ", waypoints.shape)
    # print("velocities: ", velocities.shape)
    # print("accelerations: ", accelerations.shape)

    return waypoints, velocities, accelerations, eulers

def generate_uniform_line(episode_len_sec=10, sample_time=0.01):
    start_pos = np.array([0, 0, 0.5])
    end_pos = np.array([2, 2, 0.5])

    num_waypoints = int(episode_len_sec / sample_time)

    time = np.linspace(0, 1, num_waypoints)[:, np.newaxis]
    waypoints = start_pos + time * (end_pos - start_pos)

    v = 2. / episode_len_sec
    v = np.ones(num_waypoints) * v
    velocities = np.column_stack((v, v, np.zeros_like(v)))
    accelerations = np.zeros_like(velocities)
    # e = np.array([0, 0, -0.7853982])
    # eulers = np.tile(e, (num_waypoints, 1))
    eulers = np.zeros((num_waypoints, 3))
    
    return waypoints, velocities, accelerations, eulers

def generate_hover(episode_len_sec=10, sample_time=0.01):
    p_init = np.array([2, 2, 0.5])
    v_init = np.array([0, 0, 0])
    a_init = np.array([0, 0, 0])
    
    num_waypoints = int(episode_len_sec / sample_time)

    v_hover = np.tile(v_init, (num_waypoints, 1))
    a_hover = np.tile(a_init, (num_waypoints, 1))
    p_hover = np.tile(p_init, (num_waypoints, 1))

    # Calculate Euler angles (assuming hovering)
    # eulers = np.zeros((num_waypoints, 3))

    return p_hover, v_hover, a_hover


if __name__ == "__main__":
    waypoints, velocities, accelerations, eulers = generate_uniform_line()
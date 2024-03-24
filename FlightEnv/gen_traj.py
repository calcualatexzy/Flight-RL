import numpy as np


def generate_trajectory(episode_len_sec, sample_time):
        t = np.arange(0, episode_len_sec, sample_time)

        # Define parameters
        amplitude = 5  # Amplitude of the figure 8
        frequency = 0.1  # Frequency of the figure 8
        offset = 1  # Offset to avoid starting from the origin

        # Generate trajectory
        x = amplitude * np.sin(2 * np.pi * frequency * t) + offset
        y = amplitude * np.sin(4 * np.pi * frequency * t) + offset
        z = np.linspace(0, 10, len(t))  # Linear trajectory in z-direction

        # Assuming quadrotor starts at rest, velocity and acceleration are zero
        vx = np.gradient(x, sample_time)
        vy = np.gradient(y, sample_time)
        vz = np.gradient(z, sample_time)

        ax = np.gradient(vx, sample_time)
        ay = np.gradient(vy, sample_time)
        az = np.gradient(vz, sample_time)

        # Check and enforce velocity and acceleration bounds if needed

        pos_ref, vel_ref, acc_ref = np.column_stack((x, y, z)), np.column_stack((vx, vy, vz)), np.column_stack((ax, ay, az))
        return pos_ref, vel_ref

if __name__ == "__main__":
    episode_len_sec = 10
    sample_time = 0.01
    pos_ref, vel_ref = generate_trajectory(episode_len_sec, sample_time)

    #plot
    import matplotlib.pyplot as plt
    # 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], label='Trajectory')
    plt.show()



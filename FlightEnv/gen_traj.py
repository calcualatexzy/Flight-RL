import numpy as np
import math

def unit_vector(data, axis=None, out=None):
    '''Return ndarray normalized by length, i.e. Euclidean norm, along axis.
      >>> v0 = np.random.random(3)
      >>> v1 = unit_vector(v0)
      >>> np.allclose(v1, v0 / np.linalg.norm(v0))
      True
      >>> v0 = np.random.rand(5, 4, 3)
      >>> v1 = unit_vector(v0, axis=-1)
      >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
      >>> np.allclose(v1, v2)
      True
      >>> v1 = unit_vector(v0, axis=1)
      >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
      >>> np.allclose(v1, v2)
      True
      >>> v1 = np.empty((5, 4, 3))
      >>> unit_vector(v0, axis=1, out=v1)
      >>> np.allclose(v1, v2)
      True
      >>> list(unit_vector([]))
      []
      >>> list(unit_vector([1]))
      [1.0]
    '''
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def projection_matrix(point, normal, direction=None, perspective=None, pseudo=False):
    '''Return matrix to project onto plane defined by point and normal.
      Using either perspective point, projection direction, or none of both.
      If pseudo is True, perspective projections will preserve relative depth
      such that Perspective = dot(Orthogonal, PseudoPerspective).
      >>> P = projection_matrix([0, 0, 0], [1, 0, 0])
      >>> np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])
      True
      >>> point = np.random.random(3) - 0.5
      >>> normal = np.random.random(3) - 0.5
      >>> direct = np.random.random(3) - 0.5
      >>> persp = np.random.random(3) - 0.5
      >>> P0 = projection_matrix(point, normal)
      >>> P1 = projection_matrix(point, normal, direction=direct)
      >>> P2 = projection_matrix(point, normal, perspective=persp)
      >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
      >>> is_same_transform(P2, np.dot(P0, P3))
      True
      >>> P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
      >>> v0 = (np.random.rand(4, 5) - 0.5) * 20
      >>> v0[3] = 1
      >>> v1 = np.dot(P, v0)
      >>> np.allclose(v1[1], v0[1])
      True
      >>> np.allclose(v1[0], 3-v1[1])
      True
    '''
    M = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.array(perspective[:3], dtype=np.float64, copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective - point, normal)
        M[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= np.outer(normal, normal)
            M[:3, 3] = np.dot(point, normal) * (perspective + normal)
        else:
            M[:3, 3] = np.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = np.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = np.array(direction[:3], dtype=np.float64, copy=False)
        scale = np.dot(direction, normal)
        M[:3, :3] -= np.outer(direction, normal) / scale
        M[:3, 3] = direction * (np.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= np.outer(normal, normal)
        M[:3, 3] = np.dot(point, normal) * normal
    return M

def transform_trajectory(pos, vel, trans_info={}):
    '''Makes 2D reference trajectory into a 3D one.

    Args:
        pos: position in the reference trajectory, with shape (T,3).
        vel: velocity in the reference trajectory, with shape (T,3).
    '''
    # Shape (4,4) with augmented last dim (always 1).
    M = projection_matrix(trans_info['point'], trans_info['normal'])
    # Position.
    aug_pos = np.concatenate([pos, np.ones((pos.shape[0], 1))], -1)  # (T,4)
    trans_pos = np.matmul(aug_pos, M.transpose())[:, :3]  # (T,3)
    # Velocity (transfomration is linear, direclty multiply for derivatives).
    aug_vel = np.concatenate([vel, np.ones((vel.shape[0], 1))], -1)  # (T,4)
    trans_vel = np.matmul(aug_vel, M.transpose())[:, :3]  # (T,3)
    return trans_pos, trans_vel

def _figure8(t,
            traj_period,
            scaling
            ):
    '''Computes the coordinates of a figure8 trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_period (float): The period of the trajectory in seconds.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        coords_a (float): The position in the first coordinate.
        coords_b (float): The position in the second coordinate.
        coords_a_dot (float): The velocity in the first coordinate.
        coords_b_dot (float): The velocity in the second coordinate.
    '''

    traj_freq = 2.0 * np.pi / traj_period
    coords_a = scaling * np.sin(traj_freq * t)
    coords_b = scaling * np.sin(traj_freq * t) * np.cos(traj_freq * t)
    coords_a_dot = scaling * traj_freq * np.cos(traj_freq * t)
    coords_b_dot = scaling * traj_freq * (np.cos(traj_freq * t)**2 - np.sin(traj_freq * t)**2)
    return coords_a, coords_b, coords_a_dot, coords_b_dot

def _circle(t,
            traj_period,
            scaling
            ):
    '''Computes the coordinates of a circle trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_period (float): The period of the trajectory in seconds.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        coords_a (float): The position in the first coordinate.
        coords_b (float): The position in the second coordinate.
        coords_a_dot (float): The velocity in the first coordinate.
        coords_b_dot (float): The velocity in the second coordinate.
    '''

    traj_freq = 2.0 * np.pi / traj_period
    coords_a = scaling * np.cos(traj_freq * t)
    coords_b = scaling * np.sin(traj_freq * t)
    coords_a_dot = -scaling * traj_freq * np.sin(traj_freq * t)
    coords_b_dot = scaling * traj_freq * np.cos(traj_freq * t)
    return coords_a, coords_b, coords_a_dot, coords_b_dot

def _square(t,
            traj_period,
            scaling
            ):
    '''Computes the coordinates of a square trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_period (float): The period of the trajectory in seconds.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        coords_a (float): The position in the first coordinate.
        coords_b (float): The position in the second coordinate.
        coords_a_dot (float): The velocity in the first coordinate.
        coords_b_dot (float): The velocity in the second coordinate.
    '''

    # Compute time for each segment to complete.
    segment_period = traj_period / 4.0
    traverse_speed = scaling / segment_period
    # Compute time for the cycle.
    cycle_time = t % traj_period
    # Check time along the current segment and ratio of completion.
    segment_time = cycle_time % segment_period
    # Check current segment index.
    segment_index = int(np.floor(cycle_time / segment_period))
    # Position along segment
    segment_position = traverse_speed * segment_time
    if segment_index == 0:
        # Moving up along second axis from (0, 0).
        coords_a = 0.0
        coords_b = segment_position
        coords_a_dot = 0.0
        coords_b_dot = traverse_speed
    elif segment_index == 1:
        # Moving left along first axis from (0, 1).
        coords_a = -segment_position
        coords_b = scaling
        coords_a_dot = -traverse_speed
        coords_b_dot = 0.0
    elif segment_index == 2:
        # Moving down along second axis from (-1, 1).
        coords_a = -scaling
        coords_b = scaling - segment_position
        coords_a_dot = 0.0
        coords_b_dot = -traverse_speed
    elif segment_index == 3:
        # Moving right along second axis from (-1, 0).
        coords_a = -scaling + segment_position
        coords_b = 0.0
        coords_a_dot = traverse_speed
        coords_b_dot = 0.0
    return coords_a, coords_b, coords_a_dot, coords_b_dot

def _get_coordinates(t,
                    traj_type,
                    traj_period,
                    coord_index_a,
                    coord_index_b,
                    position_offset_a,
                    position_offset_b,
                    scaling):
    '''Computes the coordinates of a specified trajectory at time t.

    Args:
        t (float): The time at which we want to sample one trajectory point.
        traj_type (str, optional): The type of trajectory (circle, square, figure8).
        traj_period (float): The period of the trajectory in seconds.
        coord_index_a (int): The index of the first coordinate of the trajectory plane.
        coord_index_b (int): The index of the second coordinate of the trajectory plane.
        position_offset_a (float): The offset in the first coordinate of the trajectory plane.
        position_offset_b (float): The offset in the second coordinate of the trajectory plane.
        scaling (float, optional): Scaling factor for the trajectory.

    Returns:
        pos_ref (ndarray): The position in x, y, z, at time t.
        vel_ref (ndarray): The velocity in x, y, z, at time t.
    '''

    # Get coordinates for the trajectory chosen.
    if traj_type == 'figure8':
        coords_a, coords_b, coords_a_dot, coords_b_dot = _figure8(
            t, traj_period, scaling)
    elif traj_type == 'circle':
        coords_a, coords_b, coords_a_dot, coords_b_dot = _circle(
            t, traj_period, scaling)
    elif traj_type == 'square':
        coords_a, coords_b, coords_a_dot, coords_b_dot = _square(
            t, traj_period, scaling)
    # Initialize position and velocity references.
    pos_ref = np.zeros((3,))
    vel_ref = np.zeros((3,))
    # Set position and velocity references based on the plane of the trajectory chosen.
    pos_ref[coord_index_a] = coords_a + position_offset_a
    vel_ref[coord_index_a] = coords_a_dot
    pos_ref[coord_index_b] = coords_b + position_offset_b
    vel_ref[coord_index_b] = coords_b_dot

    return pos_ref, vel_ref

def _plot_trajectory(traj_type,
                        traj_plane,
                        traj_length,
                        num_cycles,
                        pos_ref_traj,
                        vel_ref_traj,
                        speed_traj
                        ):
    '''Plots a trajectory along x, y, z, and in a 3D projection.

    Args:
        traj_type (str, optional): The type of trajectory (circle, square, figure8).
        traj_plane (str, optional): The plane of the trajectory (e.g. 'xz').
        traj_length (float, optional): The length of the trajectory in seconds.
        num_cycles (int, optional): The number of cycles within the length.
        pos_ref_traj (ndarray): The positions in x, y, z of the trajectory sampled for its entire duration.
        vel_ref_traj (ndarray): The velocities in x, y, z of the trajectory sampled for its entire duration.
        speed_traj (ndarray): The scalar speed of the trajectory sampled for its entire duration.
    '''

    # Print basic properties.
    print(f'Trajectory type: {traj_type}')
    print(f'Trajectory plane: {traj_plane}')
    print(f'Trajectory length: {traj_length} sec')
    print(f'Number of cycles: {num_cycles}')
    print(f'Trajectory period: {traj_length / num_cycles:.2f} sec')
    print(f'Angular speed: {2.0 * np.pi / (traj_length / num_cycles):.2f} rad/sec')
    print(
        'Position bounds: x [%.2f, %.2f] m, y [%.2f, %.2f] m, z [%.2f, %.2f] m'
        % (min(pos_ref_traj[:, 0]), max(pos_ref_traj[:, 0]),
            min(pos_ref_traj[:, 1]), max(pos_ref_traj[:, 1]),
            min(pos_ref_traj[:, 2]), max(pos_ref_traj[:, 2])))
    print(
        'Velocity bounds: vx [%.2f, %.2f] m/s, vy [%.2f, %.2f] m/s, vz [%.2f, %.2f] m/s'
        % (min(vel_ref_traj[:, 0]), max(vel_ref_traj[:, 0]),
            min(vel_ref_traj[:, 1]), max(vel_ref_traj[:, 1]),
            min(vel_ref_traj[:, 2]), max(vel_ref_traj[:, 2])))
    print('Speed: min %.2f m/s max %.2f m/s mean %.2f' %
            (min(speed_traj), max(speed_traj), np.mean(speed_traj)))
    # Plot in x, y, z.
    fig, axs = plt.subplots(3, 2)
    t = np.arange(0, traj_length, traj_length / pos_ref_traj.shape[0])
    axs[0, 0].plot(t, pos_ref_traj[:, 0])
    axs[0, 0].set_ylabel('pos x (m)')
    axs[1, 0].plot(t, pos_ref_traj[:, 1])
    axs[1, 0].set_ylabel('pos y (m)')
    axs[2, 0].plot(t, pos_ref_traj[:, 2])
    axs[2, 0].set_ylabel('pos z (m)')
    axs[2, 0].set_xlabel('time (s)')
    axs[0, 1].plot(t, vel_ref_traj[:, 0])
    axs[0, 1].set_ylabel('vel x (m)')
    axs[1, 1].plot(t, vel_ref_traj[:, 1])
    axs[1, 1].set_ylabel('vel y (m)')
    axs[2, 1].plot(t, vel_ref_traj[:, 2])
    axs[2, 1].set_ylabel('vel z (m)')
    axs[2, 1].set_xlabel('time (s)')
    plt.show()
    # Plot in 3D.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(pos_ref_traj[:, 0], pos_ref_traj[:, 1], pos_ref_traj[:, 2])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()

def generate_trajectory(traj_type='figure8',
                        episode_len_sec=10.0,
                        num_cycles=2,
                        traj_plane='xy',
                        position_offset=np.array([0, 0]),
                        scaling=1.0,
                        sample_time=0.01):
        '''Generates a 2D trajectory.

        Args:
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            episode_len_sec (float, optional): The length of the trajectory in seconds.
            num_cycles (int, optional): The number of cycles within the length.
            traj_plane (str, optional): The plane of the trajectory (e.g. 'xz').
            position_offset (ndarray, optional): An initial position offset in the plane.
            scaling (float, optional): Scaling factor for the trajectory.
            sample_time (float, optional): The sampling timestep of the trajectory.

        Returns:
            ndarray: The positions in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The velocities in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The scalar speed of the trajectory sampled for its entire duration.
        '''

        # Get trajectory type.
        valid_traj_type = ['circle', 'square', 'figure8']
        if traj_type not in valid_traj_type:
            raise ValueError('Trajectory type should be one of [circle, square, figure8].')
        traj_period = episode_len_sec / num_cycles
        direction_list = ['x', 'y', 'z']
        # Get coordinates indexes.
        if traj_plane[0] in direction_list and traj_plane[
                1] in direction_list and traj_plane[0] != traj_plane[1]:
            coord_index_a = direction_list.index(traj_plane[0])
            coord_index_b = direction_list.index(traj_plane[1])
        else:
            raise ValueError('Trajectory plane should be in form of ab, where a and b can be {x, y, z}.')
        # Generate time stamps.
        times = np.arange(0, episode_len_sec + sample_time, sample_time)  # sample time added to make reference one step longer than episode_len_sec
        pos_ref_traj = np.zeros((len(times), 3))
        vel_ref_traj = np.zeros((len(times), 3))
        speed_traj = np.zeros((len(times), 1))
        # Compute trajectory points.
        for t in enumerate(times):
            pos_ref_traj[t[0]], vel_ref_traj[t[0]] = _get_coordinates(t[1],
                                                                           traj_type,
                                                                           traj_period,
                                                                           coord_index_a,
                                                                           coord_index_b,
                                                                           position_offset[0],
                                                                           position_offset[1],
                                                                           scaling)
            speed_traj[t[0]] = np.linalg.norm(vel_ref_traj[t[0]])

        POS_REF_TRANS, VEL_REF_TRANS = transform_trajectory(
                    pos_ref_traj, vel_ref_traj, trans_info={'point': [0, 0, 0.5], 'normal': [0, 1, 1]})
        return POS_REF_TRANS, VEL_REF_TRANS


if __name__ == "__main__":
    episode_len_sec = 10
    sample_time = 0.01
    pos_ref, vel_ref = generate_trajectory(episode_len_sec=episode_len_sec, sample_time=sample_time)

    #plot
    import matplotlib.pyplot as plt
    # 3d
    _plot_trajectory('figure8', 'xy', episode_len_sec, 1, pos_ref, vel_ref, np.linalg.norm(vel_ref, axis=1))
    
    plt.show()
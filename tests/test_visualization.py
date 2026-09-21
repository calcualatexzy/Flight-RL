import numpy as np
import pybullet as p

from FlightEnv.env import FlightEnv
from FlightEnv.visualization import TrajectoryViewer


def test_visual_markers_do_not_change_physics_and_are_removed_on_reset():
    with FlightEnv(mode='hover') as plain, FlightEnv(mode='hover') as displayed:
        plain.reset(seed=12)
        displayed.reset(seed=12)
        viewer = TrajectoryViewer(displayed.PYB_CLIENT)
        viewer.reset(displayed.state_goal[:, [0, 2, 4]], displayed.quadrotor.pos,
                     displayed._ground_id, displayed._episode_len_sec)
        displayed._viewer = viewer
        assert p.getNumBodies(physicsClientId=displayed.PYB_CLIENT) == 6
        for _ in range(40):
            expected = plain.step(np.zeros(4))
            actual = displayed.step(np.zeros(4))
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:] == actual[1:]
        # Reset clears viewer-owned objects before resetting/reloading the drone.
        displayed.reset(seed=13)
        assert p.getNumBodies(physicsClientId=displayed.PYB_CLIENT) == 2
        assert not viewer.bodies and not viewer.items and not viewer.hud
        displayed._viewer = None

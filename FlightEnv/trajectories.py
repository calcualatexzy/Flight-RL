"""Seeded trajectory sampling, including the bundled (unextracted) library."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

LIBRARY = Path(__file__).resolve().parent / "TrajLib/traj_200Hz_len5_vel1.5.zip"


@lru_cache(maxsize=4)
def library_members(path):
    with ZipFile(path) as archive:
        members = tuple(sorted(n for n in archive.namelist() if n.endswith(".npz")))
    if not members:
        raise ValueError(f"No trajectories in {path}")
    return members


@lru_cache(maxsize=64)
def read_trajectory(path, member):
    with ZipFile(path) as archive:
        with np.load(BytesIO(archive.read(member)), allow_pickle=False) as data:
            arrays = tuple(np.asarray(data[k], dtype=np.float64) for k in
                           ("trajectory_points", "velocities", "accelerations"))
    if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() for a in arrays):
        raise ValueError(f"Invalid trajectory: {member}")
    if len({a.shape for a in arrays}) != 1 or len(arrays[0]) < 2:
        raise ValueError(f"Inconsistent trajectory shapes: {member}")
    for a in arrays:
        a.setflags(write=False)
    return arrays


def sample_trajectory(duration, dt, rng, mode="trajectory", library_path=None):
    """Include t=0 and the final control step; resample the original 5 s curves."""
    n_steps = int(round(duration / dt))
    times = np.arange(n_steps + 1) * dt
    if mode == "hover":
        pos = np.tile([0., 0., 0.5], (n_steps + 1, 1))
        return pos, np.zeros_like(pos), np.zeros_like(pos)
    if mode != "trajectory":
        raise ValueError(f"Unknown task: {mode}")
    path = str(Path(library_path or LIBRARY).resolve())
    members = library_members(path)
    arrays = read_trajectory(path, members[int(rng.integers(len(members)))])
    # The offline generator used linspace(0, 5, 1000), not arange(1000)/200.
    source_times = np.linspace(0., 5., len(arrays[0]))
    scale = 5. / duration
    query = np.minimum(times * scale, 5.)
    return tuple(np.column_stack([np.interp(query, source_times, a[:, j])
                                  for j in range(3)]) * scale**order
                 for order, a in enumerate(arrays))

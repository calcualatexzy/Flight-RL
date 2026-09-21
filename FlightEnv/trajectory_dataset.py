"""Analytic, bounded flight paths with disjoint train/validation/test seeds."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

FAMILIES = ('hover', 'line', 'circle', 'figure8', 'helix', 'slalom', 'wave', 'vertical_loop', 'lissajous')
FLIGHT_FAMILIES = FAMILIES[1:]
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / 'data/trajectories_v3'
SPLITS = ('train', 'validation', 'test')


def generate_path(family, seed, split='train', duration=10., dt=.02):
    """Analytic position/velocity/acceleration with zero derivatives at endpoints."""
    if family not in FAMILIES or split not in SPLITS:
        raise ValueError('Unknown family or split')
    if duration <= 0 or dt <= 0 or not np.isclose(duration / dt, round(duration / dt)):
        raise ValueError('duration/dt must be a positive integer')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SPLITS.index(split), 314159]))
    u = np.linspace(0, 1, int(round(duration / dt)) + 1)
    s = 10*u**3 - 15*u**4 + 6*u**5
    sd = (30*u**2 - 60*u**3 + 30*u**4) / duration
    sdd = (60*u - 180*u**2 + 120*u**3) / duration**2
    a, b, c = rng.uniform(.8, 2.5), rng.uniform(.5, 1.8), rng.uniform(.2, .8)
    length = rng.uniform(1.5, 5.)
    cycles = int(rng.integers(1, 3)) if family == 'helix' else 1
    w = 2*np.pi*cycles
    zero = np.zeros_like(s)
    one = np.ones_like(s)

    def sine(k=1):
        return np.sin(k*w*s), k*w*np.cos(k*w*s), -(k*w)**2*np.sin(k*w*s)

    def cosine(k=1):
        return np.cos(k*w*s)-1, -k*w*np.sin(k*w*s), -(k*w)**2*np.cos(k*w*s)

    def mul(scale, values):
        return tuple(scale*x for x in values)

    flat = (zero, zero, zero)
    linear = (s, one, zero)
    if family == 'hover':
        xyz = (flat, flat, flat)
    elif family == 'line':
        xyz = (mul(length, linear), flat, mul(rng.uniform(-.3, .8), linear))
    elif family == 'circle':
        xyz = (mul(a, cosine()), mul(b, sine()), mul(-.25*c, cosine(2)))
    elif family == 'figure8':
        xyz = (mul(a, sine()), mul(.7*b, sine(2)), mul(-c, cosine(2)))
    elif family == 'helix':
        xyz = (mul(a, cosine()), mul(a, sine()), mul(rng.uniform(1., 3.), linear))
    elif family == 'slalom':
        xyz = (mul(length, linear), mul(b, sine(2)), mul(-.3*c, cosine()))
    elif family == 'wave':
        xyz = (mul(length, linear), mul(.5*b, sine()), mul(c, sine(2)))
    elif family == 'vertical_loop':
        xyz = (mul(a, sine()), mul(.25*b, sine(2)), mul(-a, cosine()))
    else:
        xyz = (mul(a, sine(2)), mul(b, sine(3)), mul(-c, cosine()))
    position = np.column_stack([v[0] for v in xyz])
    derivative = np.column_stack([v[1] for v in xyz])
    second = np.column_stack([v[2] for v in xyz])
    velocity = derivative * sd[:, None]
    acceleration = second * sd[:, None]**2 + derivative * sdd[:, None]
    yaw = rng.uniform(-np.pi, np.pi)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    if rng.random() < .5:
        rotation[:, 1] *= -1
    position, velocity, acceleration = (x @ rotation.T for x in (position, velocity, acceleration))
    peak_v = np.linalg.norm(velocity, axis=1).max()
    peak_a = np.linalg.norm(acceleration, axis=1).max()
    downward_a = max(0., -acceleration[:, 2].min())
    # Respect the idle-thrust floor as well as speed/acceleration limits.
    scale = min(1., 4.0 / max(peak_v, 1e-9), 6.0 / max(peak_a, 1e-9),
                3.0 / max(downward_a, 1e-9))
    position, velocity, acceleration = (x * scale for x in (position, velocity, acceleration))
    position[:, 2] += rng.uniform(1., 1.8)
    position[:, 2] += max(0., .8 - position[:, 2].min())
    meta = dict(family=family, seed=int(seed), split=split, yaw=float(yaw),
                amplitude_scale=float(scale), max_speed=float(np.linalg.norm(velocity, axis=1).max()),
                max_acceleration=float(np.linalg.norm(acceleration, axis=1).max()),
                min_altitude=float(position[:, 2].min()))
    return position, velocity, acceleration, meta


def dataset_root(directory=None):
    root = Path(directory or DEFAULT_DATASET)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    return root.resolve()


def evaluation_options(families, split='test', directory=None):
    """Enumerate every selected path once, interleaving trajectory families."""
    bank = load_bank(str(dataset_root(directory)), split)
    groups = [np.flatnonzero(bank['families'] == family) for family in families]
    if any(len(group) == 0 for group in groups):
        raise ValueError('An evaluation family is missing from the dataset')
    return [{'trajectory_id': int(group[i])} for i in range(max(map(len, groups)))
            for group in groups if i < len(group)]


@lru_cache(maxsize=6)
def load_bank(directory, split):
    path = Path(directory) / f'{split}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'{path} missing. Run: python -m FlightEnv.trajectory_dataset')
    with np.load(path, allow_pickle=False) as data:
        bank = {k: data[k] for k in data.files}
    return bank


def sample_bank(rng, duration, dt, family='mixed', difficulty=1., split='train',
                directory=None, trajectory_id=None):
    bank = load_bank(str(dataset_root(directory)), split)
    if trajectory_id is None:
        if family == 'mixed':
            if split == 'train' and difficulty < .3:
                family = str(rng.choice(('hover', 'line', 'circle'), p=(.15, .55, .30)))
            else:
                family = str(rng.choice(FLIGHT_FAMILIES))
        indices = np.flatnonzero(bank['families'] == family)
        if not len(indices):
            raise ValueError(f'No trajectories for family {family}')
        idx = int(rng.choice(indices))
    else:
        idx = int(trajectory_id)
        if idx < 0 or idx >= len(bank['position']):
            raise ValueError('trajectory_id is outside the selected split')
    source_time = bank['time']
    times = np.arange(int(round(duration / dt)) + 1) * dt
    query = times * float(source_time[-1]) / duration
    factor = .15 + .85 * difficulty
    time_scale = float(source_time[-1]) / duration
    arrays = []
    for order, key in enumerate(('position', 'velocity', 'acceleration')):
        source = bank[key][idx]
        values = np.column_stack([np.interp(query, source_time, source[:, j]) for j in range(3)])
        if order == 0:
            values = source[0] + factor * (values - source[0])
        else:
            values *= factor * time_scale**order
        arrays.append(values)
    metadata = dict(trajectory_id=idx, trajectory_family=str(bank['families'][idx]),
                    trajectory_seed=int(bank['seeds'][idx]), dataset_split=split,
                    difficulty=float(difficulty))
    return *arrays, metadata


def export_dataset(output=DEFAULT_DATASET, train_per_family=256, eval_per_family=32, seed=2026):
    output = Path(output)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'{output} already exists; use a new --output directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=3, seed=seed, duration=10., dt=.02, families=list(FAMILIES), splits={})
    for split in SPLITS:
        count = train_per_family if split == 'train' else eval_per_family
        positions, velocities, accelerations, labels, seeds, metadata = [], [], [], [], [], []
        for family_index, family in enumerate(FAMILIES):
            for i in range(count):
                path_seed = seed + family_index * 100000 + i
                pos, vel, acc, meta = generate_path(family, path_seed, split)
                positions.append(pos.astype(np.float32))
                velocities.append(vel.astype(np.float32))
                accelerations.append(acc.astype(np.float32))
                labels.append(family)
                seeds.append(path_seed)
                metadata.append(meta)
        np.savez_compressed(output / f'{split}.npz', time=np.linspace(0, 10, 501),
                            position=positions, velocity=velocities, acceleration=accelerations,
                            families=labels, seeds=seeds)
        manifest['splits'][split] = dict(count=len(labels), per_family=count, trajectories=metadata)
        print(f'{split}: {len(labels)} trajectories', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    plot_gallery(output)
    return manifest


def plot_gallery(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bank = load_bank(str(Path(directory).resolve()), 'test')
    fig = plt.figure(figsize=(13, 11))
    for i, family in enumerate(FAMILIES):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        indices = np.flatnonzero(bank['families'] == family)[:3]
        for index in indices:
            points = bank['position'][index]
            ax.plot(*points.T, linewidth=1.5)
            ax.scatter(*points[0], s=12)
        ax.set_title(family)
        ax.set(xlabel='X / m', ylabel='Y / m', zlabel='Z / m')
    fig.suptitle('Flight trajectory families / independent test seeds', fontsize=15)
    fig.tight_layout()
    fig.savefig(Path(directory) / 'trajectory_gallery.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(DEFAULT_DATASET))
    parser.add_argument('--train-per-family', type=int, default=256)
    parser.add_argument('--eval-per-family', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if min(args.train_per_family, args.eval_per_family) < 1:
        parser.error('Counts must be positive')
    export_dataset(args.output, args.train_per_family, args.eval_per_family, args.seed)

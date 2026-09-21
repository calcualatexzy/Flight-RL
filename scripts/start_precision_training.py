"""Launch detached precision fine-tuning in a fresh directory."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='runs/precision_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--resume', default='runs/v3_stable/best/best_model.zip')
    parser.add_argument('--timesteps', type=int, default=3_000_000)
    parser.add_argument('--num-envs', type=int, default=8)
    args = parser.parse_args()
    if min(args.timesteps, args.num_envs) <= 0:
        parser.error('timesteps and num-envs must be positive')
    root = Path(__file__).resolve().parents[1]
    run = (root / args.run_dir).resolve()
    checkpoint = (root / args.resume).resolve()
    if not checkpoint.is_file():
        parser.error(f'Missing checkpoint: {checkpoint}')
    if run.exists() and any(run.iterdir()):
        parser.error(f'{run} is not empty; use a new run directory')
    run.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(root/'main.py'), 'resume', '--resume', str(checkpoint),
        '--reward-profile', 'precision', '--time-profile', 'mixed', '--difficulty', '1',
        '--split', 'train', '--speed-min', '0.9', '--speed-max', '1.1',
        '--curriculum-max-speed', '1.3', '--learning-rate', '0.00003',
        '--ent-coef', '0.0002', '--target-kl', '0.015', '--num-envs', str(args.num_envs),
        '--timesteps', str(args.timesteps), '--torch-threads', '1', '--seed', '42',
        '--save-freq', '50000', '--eval-freq', '100000', '--eval-episodes', '16',
        '--eval-speed-scales', '1', '1.15', '1.3', '--eval-time-profiles', 'quintic', 'cruise',
        '--run-dir', str(run)]
    snapshot = run/'source_snapshot'
    for source in [root/'main.py', root/'TRAINING_PRECISION.md', *sorted((root/'FlightEnv').glob('*.py'))]:
        if source.is_file():
            target = snapshot/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = dict(command=command, cwd=str(root), starting_model=str(checkpoint),
                    starting_model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    started_at=datetime.now().astimezone().isoformat(), additional_steps=args.timesteps,
                    source_sha256={str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in snapshot.rglob('*.py')})
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (run/'training.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    manifest['pid'] = process.pid
    (run/'launch.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (run/'training.pid').write_text(str(process.pid)+'\n')
    print(json.dumps(dict(pid=process.pid, run_dir=str(run), log=str(run/'training.log')), indent=2))


if __name__ == '__main__':
    main()

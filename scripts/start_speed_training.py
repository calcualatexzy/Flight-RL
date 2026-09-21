"""Launch the explicit geometric-control + residual-RL speed experiment."""
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
    parser.add_argument('--run-dir', default='runs/residual_speed_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    parser.add_argument('--num-envs', type=int, default=8)
    args = parser.parse_args()
    if min(args.timesteps, args.num_envs) <= 0:
        parser.error('timesteps and num-envs must be positive')
    root = Path(__file__).resolve().parents[1]
    run = (root/args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        parser.error(f'{run} is not empty; choose a fresh directory')
    run.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(root/'main.py'), 'train', '--profile', 'residual',
        '--reward-profile', 'precision', '--time-profile', 'mixed', '--difficulty', '1',
        '--split', 'train', '--speed-min', '1', '--speed-max', '1.3', '--curriculum-max-speed', '1.5',
        '--learning-rate', '0.00005', '--ent-coef', '0', '--target-kl', '0.01',
        '--n-steps', '512', '--batch-size', '256', '--n-epochs', '5',
        '--num-envs', str(args.num_envs), '--timesteps', str(args.timesteps), '--torch-threads', '1', '--seed', '42',
        '--save-freq', '50000', '--eval-freq', '100000', '--eval-episodes', '16', '--eval-patience', '6',
        '--eval-speed-scales', '1', '1.3', '1.5', '--eval-time-profiles', 'quintic', 'cruise', '--run-dir', str(run)]
    snapshot = run/'source_snapshot'
    for source in [root/'main.py', Path(__file__).resolve(), root/'TRAINING_SPEED.md', *sorted((root/'FlightEnv').glob('*.py'))]:
        if source.is_file():
            target = snapshot/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = dict(command=command, cwd=str(root), started_at=datetime.now().astimezone().isoformat(),
        additional_steps=args.timesteps, control_mode='geometric_feedforward_plus_rl_residual',
        initialization='New 67-input policy, zero actor output, geometric baseline; not resumed pure-RL weights',
        source_sha256={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in snapshot.rglob('*.py')})
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (run/'training.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    manifest['pid'] = process.pid
    (run/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (run/'training.pid').write_text(str(process.pid)+'\n')
    print(json.dumps(dict(pid=process.pid,run_dir=str(run),log=str(run/'training.log')),indent=2))


if __name__ == '__main__':
    main()

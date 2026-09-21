"""Create the disclosed zero-residual control baseline without training/downloads."""
import argparse
import json
from pathlib import Path
import sys

import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import FORMAT_VERSION, create_env, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/residual_baseline/baseline.zip')
    parser.add_argument('--dataset-dir')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output).with_suffix('.zip')
    if output.exists():
        parser.error(f'{output} already exists; choose a new --output')
    torch.set_num_threads(1)
    config = dict(profile='residual', mode='trajectory', episode_len_sec=10.,
        is_domain_randomization=True, difficulty=1., trajectory_family='mixed',
        dataset_split='train', ctrl_freq=50, pybullet_freq=200, goal_stride=5,
        reward_heading_weight=.1, reward_yaw_rate_weight=.02, reward_profile='precision',
        time_profile='mixed', speed_min=1., speed_max=1.3)
    if args.dataset_dir:
        config['dataset_dir'] = args.dataset_dir
    with create_env(config) as env:
        model = PPO('MlpPolicy', env, n_steps=512, batch_size=256, n_epochs=5,
            learning_rate=5e-5, gamma=.995, gae_lambda=.98, ent_coef=0., target_kl=.01,
            policy_kwargs=dict(activation_fn=torch.nn.Tanh,
                net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-2.5),
            device='cpu', seed=args.seed, verbose=0)
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
        model.flight_format_version = FORMAT_VERSION
        model.flight_env_config = config
        save_model(model, output)
    metadata = dict(model_timesteps=0, seed=args.seed, environment=config,
        control_mode='geometric_feedforward_zero_residual',
        description='Deterministic policy output is zero; tracking comes from geometric control. No PPO updates.')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {output.resolve()} (zero residual; no training updates).')


if __name__ == '__main__':
    main()

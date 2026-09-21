"""Train, resume and evaluate PPO on the flight environment."""
import argparse
from datetime import datetime
from functools import partial
import json
import multiprocessing
from pathlib import Path
import signal
import sys
import time
from zipfile import ZipFile

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from FlightEnv.env import FlightEnv

FORMAT_VERSION = 4


def save_model(model, path):
    """Replace checkpoints atomically so an interrupted write leaves the old file."""
    path = Path(path).with_suffix('.zip')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.zip')
    model.save(temp)
    temp.replace(path)


def load_model(path, device, env=None):
    checkpoint = Path(path)
    if not checkpoint.exists() and checkpoint.suffix != '.zip':
        checkpoint = Path(str(checkpoint) + '.zip')
    # Inspect plain metadata first: old Python 3.9 pickles may not deserialize
    # under Python 3.11, and legacy actions must not be silently reinterpreted.
    with ZipFile(checkpoint) as archive:
        metadata = json.loads(archive.read('data'))
    if metadata.get('flight_format_version') not in (2, 3, FORMAT_VERSION):
        raise ValueError('This is a legacy checkpoint with physical-thrust actions. '
                         'The corrected environment uses normalized actions and different rewards. '
                         'Keep the old checkpoint and start with: python main.py train')
    return PPO.load(checkpoint, device=device, env=env)


class TrainingCheckpoint(BaseCallback):
    def __init__(self, save_freq, directory):
        super().__init__()
        self.save_freq = save_freq
        self.directory = directory
        self.stop_requested = False

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            save_model(self.model, self.directory / f'ppo_{self.num_timesteps}_steps.zip')
        return not self.stop_requested


def make_env(config):
    if multiprocessing.current_process().name != 'MainProcess':
        # Let the parent save the policy before closing workers on terminal Ctrl+C.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    return create_env(config)


def create_env(config):
    config = dict(config)
    profile = config.pop('profile', 'legacy')
    if profile == 'residual':
        from FlightEnv.residual_env import ResidualTrackingEnv
        return ResidualTrackingEnv(**config)
    if profile == 'tracking':
        from FlightEnv.tracking_env import TrackingEnv
        return TrackingEnv(**config)
    return FlightEnv(**config)


def tracking_options(args, config):
    options = {'difficulty': args.difficulty, 'trajectory_family': args.family,
               'dataset_split': args.split, 'dataset_dir': args.dataset_dir,
               'reward_profile': args.reward_profile, 'time_profile': args.time_profile,
               'speed_min': args.speed_min, 'speed_max': args.speed_max}
    for key, value in options.items():
        if value is not None:
            if config.get('profile') not in ('tracking', 'residual'):
                raise ValueError(f'{key} requires a V3 tracking model; train a new --profile tracking model')
            config[key] = value


def train(args):
    torch.set_num_threads(args.torch_threads)
    model = load_model(args.resume, args.device) if args.resume else None
    if model:
        config = dict(model.flight_env_config)
        if args.profile is not None and args.profile != config.get('profile', 'legacy'):
            raise ValueError('Cannot change observation/action profile when resuming; start a new model')
    elif args.profile == 'legacy':
        config = dict(mode='trajectory', episode_len_sec=5., is_domain_randomization=True, goal_stride=20)
    else:
        config = dict(profile=args.profile or 'tracking', mode='trajectory', episode_len_sec=10.,
                      is_domain_randomization=True, difficulty=.15, trajectory_family='mixed',
                      dataset_split='train', ctrl_freq=50, pybullet_freq=200, goal_stride=5,
                      reward_heading_weight=.1, reward_yaw_rate_weight=.02)
    tracking_options(args, config)
    if config.get('profile') in ('tracking', 'residual') and config.get('dataset_split') != 'train':
        raise ValueError('Training must use the train split; validation/test are held out')
    for attr, key in (('heading_weight', 'reward_heading_weight'), ('yaw_rate_weight', 'reward_yaw_rate_weight')):
        value = getattr(args, attr)
        if value is not None:
            if config.get('profile') not in ('tracking', 'residual'):
                raise ValueError(f'--{attr.replace("_", "-")} requires a V3 tracking model')
            config[key] = value
    if args.goal_stride is not None:
        config['goal_stride'] = args.goal_stride
    if args.task is not None:
        config['mode'] = args.task
    if args.episode_seconds is not None:
        config['episode_len_sec'] = args.episode_seconds
    if args.domain_randomization is not None:
        config['is_domain_randomization'] = args.domain_randomization
    if args.curriculum_max_speed is not None:
        if config.get('profile') not in ('tracking', 'residual'):
            raise ValueError('Speed curriculum requires tracking profile')
        if args.curriculum_max_speed < config.get('speed_max', 1.):
            raise ValueError('Curriculum ceiling must be >= current speed_max')
        if max(args.eval_speed_scales or [1.]) < args.curriculum_max_speed:
            raise ValueError('Validation must include a speed probe at/above the curriculum ceiling')
    run_dir = Path(args.run_dir or (Path(args.resume).resolve().parent if args.resume else
                    Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if not args.resume and (run_dir / 'final_model.zip').exists():
        raise ValueError(f'{run_dir} already contains a model; use --resume or a new --run-dir')
    run_dir.mkdir(parents=True, exist_ok=True)
    factories = [partial(make_env, config) for _ in range(args.num_envs)]
    vec_cls = SubprocVecEnv if args.num_envs > 1 and args.vec_env == 'subproc' else DummyVecEnv
    env = eval_env = None
    previous_sigterm = None
    try:
        env = vec_cls(factories, **({'start_method': 'spawn'} if vec_cls is SubprocVecEnv else {}))
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        monitor_name = f'monitor_resume_{session_id}.csv' if args.resume else 'monitor.csv'
        env = VecMonitor(env, str(run_dir / monitor_name))
        env.seed(args.seed)
        if model is None:
            model = PPO('MlpPolicy', env, n_steps=args.n_steps, batch_size=args.batch_size,
                        n_epochs=args.n_epochs, learning_rate=args.learning_rate or 3e-4,
                        gamma=0.995, gae_lambda=.98 if config.get('profile') == 'residual' else .95, ent_coef=.002 if config.get('profile') in ('tracking', 'residual') else 0., target_kl=0.03,
                        policy_kwargs=dict(activation_fn=torch.nn.Tanh,
                                           net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                           log_std_init=-2.5 if config.get('profile') == 'residual' else
                                                        (-.8 if config.get('profile') == 'tracking' else -1.5)),
                        tensorboard_log=str(run_dir / 'tensorboard'),
                        device=args.device, seed=args.seed, verbose=1)
        else:
            # Preserve policy and optimizer; explicit flags override selected PPO settings.
            model = load_model(args.resume, args.device, env=env)
            model.seed = args.seed
            model.set_random_seed(args.seed)
            if args.learning_rate is not None:
                model.learning_rate = args.learning_rate
                model._setup_lr_schedule()
            model.tensorboard_log = str(run_dir / 'tensorboard')
        if not args.resume and config.get('profile') == 'residual':
            # A new hybrid policy starts exactly at its disclosed controller
            # baseline; subsequent PPO updates learn the residual correction.
            torch.nn.init.zeros_(model.policy.action_net.weight)
            torch.nn.init.zeros_(model.policy.action_net.bias)
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
        if args.target_kl is not None:
            model.target_kl = args.target_kl
        model.flight_format_version = FORMAT_VERSION
        model.flight_env_config = config
        actual_config = dict(vars(args), environment=config, starting_timesteps=model.num_timesteps,
                             ppo=dict(n_steps=model.n_steps, batch_size=model.batch_size,
                                      n_epochs=model.n_epochs, learning_rate=model.learning_rate, gae_lambda=model.gae_lambda,
                                      ent_coef=model.ent_coef, target_kl=model.target_kl))
        config_name = 'config.json' if not args.resume else f'resume_{model.num_timesteps}_{session_id}.json'
        (run_dir / config_name).write_text(json.dumps(actual_config, indent=2) + '\n')
        checkpoint = TrainingCheckpoint(max(args.save_freq // args.num_envs, 1), run_dir / 'checkpoints')
        callbacks = [checkpoint]
        if args.eval_freq:
            if config.get('profile') in ('tracking', 'residual'):
                from FlightEnv.evaluation import TrackingEvaluation
                eval_config = dict(config, dataset_split='validation')
                eval_env = create_env(eval_config)
                callbacks.append(TrackingEvaluation(eval_env, config, run_dir / 'validation',
                                                    max(args.eval_freq // args.num_envs, 1),
                                                    args.eval_episodes, args.seed+10000, save_model,
                                                    speed_scales=args.eval_speed_scales,
                                                    time_profiles=args.eval_time_profiles,
                                                    curriculum_max_speed=args.curriculum_max_speed,
                                                    patience=args.eval_patience))
            else:
                eval_env = Monitor(create_env(config))
                eval_env.reset(seed=args.seed + 10000)
                callbacks.append(EvalCallback(eval_env, best_model_save_path=str(run_dir / 'best'),
                                              log_path=str(run_dir / 'evaluation' / session_id),
                                              eval_freq=max(args.eval_freq // args.num_envs, 1),
                                              n_eval_episodes=args.eval_episodes, deterministic=True))
        previous_sigterm = signal.signal(signal.SIGTERM,
                                         lambda *_: setattr(checkpoint, 'stop_requested', True))
        print(f'Run directory: {run_dir}', flush=True)
        try:
            model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not bool(args.resume),
                        tb_log_name='ppo_flight', callback=callbacks)
        except KeyboardInterrupt:
            print('Training interrupted; saving current policy and optimizer.', flush=True)
        save_model(model, run_dir / 'final_model.zip')
        print(f'Saved {run_dir / "final_model.zip"} at {model.num_timesteps} total steps.', flush=True)
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


def evaluate(args):
    torch.set_num_threads(args.torch_threads)
    model = load_model(args.model, args.device)
    config = dict(model.flight_env_config)
    if config.get('profile') in ('tracking', 'residual'):
        config['dataset_split'] = 'test'
    tracking_options(args, config)
    if args.task is not None:
        config['mode'] = args.task
    config['render_mode'] = 'human' if args.render else None
    output = Path(args.output_dir or Path(args.model).resolve().parent / 'eval')
    output.mkdir(parents=True, exist_ok=True)
    results, tracks, references = [], [], []
    schedule = [None] * args.episodes
    if config.get('profile') in ('tracking', 'residual'):
        from FlightEnv.evaluation import evaluation_families
        families = evaluation_families(config)
        schedule = [{'family': families[i % len(families)]} for i in range(args.episodes)]
        if args.all_trajectories:
            from FlightEnv.trajectory_dataset import evaluation_options
            schedule = evaluation_options(families, config['dataset_split'], config.get('dataset_dir'))
    elif args.all_trajectories:
        raise ValueError('--all-trajectories requires a V3 tracking model')
    with create_env(config) as env:
        for episode, options in enumerate(schedule):
            obs, _ = env.reset(seed=args.seed + episode, options=options)
            reference_path = np.column_stack((np.arange(len(env.state_goal)) * env._time_step,
                                               env.state_goal[:, [0, 2, 4]]))
            np.savetxt(output / f'reference_{episode:03d}.csv', reference_path, delimiter=',',
                       header='time,x,y,z', comments='')
            initial_position = np.array(env.quadrotor.pos)
            rows, velocities, reference_speeds, actual_speeds = [], [], [], []
            rewards = 0.
            next_step_time = time.monotonic()
            while True:
                if args.render and env._viewer.paused:
                    while env._viewer.paused:
                        env._viewer.refresh()
                        time.sleep(.02)
                    next_step_time = time.monotonic()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards += reward
                velocities.append(info['velocity_error'])
                reference_speeds.append(float(np.linalg.norm(env.state_goal[env._env_step_counter, [1, 3, 5]])))
                actual_speeds.append(float(np.linalg.norm(env.quadrotor.vel)))
                reference = env.state_goal[env._env_step_counter, [0, 2, 4]]
                rows.append([info['elapsed_seconds'], *env.quadrotor.pos, *reference,
                             info['position_error'], reward])
                if args.render:
                    next_step_time += env._time_step / args.playback_speed
                    time.sleep(max(0., next_step_time - time.monotonic()))
                if terminated or truncated:
                    break
            trajectory = np.asarray(rows)
            np.savetxt(output / f'episode_{episode:03d}.csv', trajectory, delimiter=',',
                       header='time,x,y,z,target_x,target_y,target_z,position_error,reward', comments='')
            results.append(dict(seed=args.seed + episode, reward=rewards, steps=len(rows),
                                position_rmse=float(np.sqrt(np.mean(trajectory[:, 7]**2))),
                                terminated=terminated, truncated=truncated,
                                collision=info['collision'], out_of_bounds=info['out_of_bounds']))
            results[-1]['max_position_error'] = float(trajectory[:, 7].max())
            from FlightEnv.evaluation import tracking_metrics
            results[-1].update(tracking_metrics(trajectory[:, 7], truncated, velocities,
                                                reference_speeds, actual_speeds))
            if config.get('profile') in ('tracking', 'residual'):
                results[-1].update({k: info[k] for k in ('trajectory_family', 'trajectory_id', 'dataset_split',
                    'time_profile', 'requested_speed_scale', 'effective_speed_scale', 'reference_duration', 'speed_limited', 'effective_time_profile', 'cruise_fallback')})
            initial = [0., *initial_position, *reference_path[0, 1:],
                       np.linalg.norm(initial_position - reference_path[0, 1:]), 0.]
            tracks.append(np.vstack((initial, trajectory)))
            references.append(reference_path)
            if args.render:
                deadline = time.monotonic() + args.hold_seconds
                while time.monotonic() < deadline:
                    env._viewer.refresh()
                    time.sleep(.03)
    summary = dict(model=str(Path(args.model).resolve()), model_timesteps=model.num_timesteps,
                   environment=config, episodes=results,
                   mean_reward=float(np.mean([r['reward'] for r in results])),
                   mean_position_rmse=float(np.mean([r['position_rmse'] for r in results])),
                   completion_rate=float(np.mean([r['truncated'] for r in results])),
                   tracking_success_rate=float(np.mean([r['tracking_success'] for r in results])))
    if config.get('profile') in ('tracking', 'residual'):
        summary['per_family'] = {}
        for family in sorted({r['trajectory_family'] for r in results}):
            group = [r for r in results if r['trajectory_family'] == family]
            summary['per_family'][family] = dict(episodes=len(group),
                mean_position_rmse=float(np.mean([r['position_rmse'] for r in group])),
                completion_rate=float(np.mean([r['truncated'] for r in group])),
                tracking_success_rate=float(np.mean([r['tracking_success'] for r in group])))
    summary['control_mode'] = ('geometric_feedforward_plus_rl_residual'
        if config.get('profile') == 'residual' else 'direct_rl')
    summary['acceptance'] = dict(rmse=.10, max_error=.30, must_complete=True)
    summary['legacy_tracking_success_rate'] = float(np.mean([r['legacy_tracking_success'] for r in results]))
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    from FlightEnv.visualization import save_tracking_plot, save_family_gallery
    save_tracking_plot(tracks, references, results, output / 'tracking.png')
    if config.get('profile') in ('tracking', 'residual'):
        save_family_gallery(tracks, references, results, output / 'family_comparison.png')
    print(json.dumps(summary, indent=2))
    print(f'Evaluation saved to {output}')


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the old train flag; retrain/load now require an explicit new checkpoint.
    if argv and argv[0] in ('--train', '--load', '--retrain'):
        argv[0] = {'--train': 'train', '--load': 'eval', '--retrain': 'resume'}[argv[0]]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'resume', 'eval'):
        sub = commands.add_parser(name)
        sub.add_argument('--device', default='cpu', choices=('cpu', 'cuda', 'auto'))
        sub.add_argument('--torch-threads', type=positive_int, default=1)
        sub.add_argument('--seed', type=int, default=42)
        sub.add_argument('--task', choices=('hover', 'trajectory'))
        sub.add_argument('--difficulty', type=float, help='V3 spatial difficulty in [0, 1]')
        sub.add_argument('--family', choices=('mixed','hover','line','circle','figure8','helix','slalom','wave','vertical_loop','lissajous'))
        sub.add_argument('--split', choices=('train','validation','test'))
        sub.add_argument('--dataset-dir')
        sub.add_argument('--reward-profile', choices=('legacy', 'precision'),
                         help='Omit to preserve checkpoint reward; precision emphasizes 5--10 cm errors')
        sub.add_argument('--time-profile', choices=('quintic', 'cruise', 'mixed'))
        sub.add_argument('--speed-min', type=float, help='Minimum reference speed multiplier (not playback speed)')
        sub.add_argument('--speed-max', type=float, help='Maximum reference speed multiplier; constrained by reference dynamics')
        if name == 'eval':
            sub.add_argument('--model', required=True)
            sub.add_argument('--episodes', type=positive_int, default=8)
            sub.add_argument('--all-trajectories', action='store_true',
                             help='Evaluate each selected V3 split trajectory once; overrides --episodes')
            sub.add_argument('--render', action='store_true')
            sub.add_argument('--playback-speed', type=float, default=1., help='GUI playback speed; 0.5 = half speed')
            sub.add_argument('--hold-seconds', type=float, default=8., help='Keep the final GUI view visible for this long')
            sub.add_argument('--output-dir')
        else:
            sub.add_argument('--resume', required=name == 'resume')
            sub.add_argument('--profile', choices=('tracking', 'residual', 'legacy'), help='tracking: pure RL; residual: geometric feedforward + learned correction')
            sub.add_argument('--timesteps', type=positive_int, default=1_000_000,
                             help='Additional environment transitions (rounded up to a PPO rollout)')
            sub.add_argument('--num-envs', type=positive_int, default=4)
            sub.add_argument('--vec-env', choices=('dummy', 'subproc'), default='subproc')
            sub.add_argument('--n-steps', type=positive_int, default=512)
            sub.add_argument('--batch-size', type=positive_int, default=256)
            sub.add_argument('--n-epochs', type=positive_int, default=10)
            sub.add_argument('--learning-rate', type=float,
                             help='New model: 3e-4; resume: keep saved rate unless supplied')
            sub.add_argument('--ent-coef', type=float, help='Override PPO entropy bonus when fine-tuning')
            sub.add_argument('--target-kl', type=float, help='Override PPO early-stop KL threshold')
            sub.add_argument('--curriculum-max-speed', type=float,
                             help='Promote speed_max by 0.1 after two passing validations, up to this ceiling')
            sub.add_argument('--eval-speed-scales', type=float, nargs='+')
            sub.add_argument('--eval-time-profiles', choices=('quintic', 'cruise'), nargs='+')
            sub.add_argument('--heading-weight', type=float, help='V3 yaw heading reward (new models: 0.1)')
            sub.add_argument('--yaw-rate-weight', type=float, help='V3 body yaw rate penalty (new models: 0.02)')
            sub.add_argument('--episode-seconds', type=float)
            sub.add_argument('--goal-stride', type=positive_int, help='Control steps between future goals (tracking: 5 at 50 Hz; legacy: 20 at 200 Hz)')
            sub.add_argument('--domain-randomization', action=argparse.BooleanOptionalAction, default=None)
            sub.add_argument('--save-freq', type=positive_int, default=50_000)
            sub.add_argument('--eval-freq', type=int, default=50_000, help='0 disables periodic evaluation')
            sub.add_argument('--eval-patience', type=positive_int, default=None,
                             help='Stop after this many validations without improvement; keeps best checkpoint')
            sub.add_argument('--eval-episodes', type=positive_int, default=5)
            sub.add_argument('--run-dir')
    args = parser.parse_args(argv)
    if args.difficulty is not None and not 0 <= args.difficulty <= 1:
        parser.error('--difficulty must be in [0, 1]')
    for value in (args.speed_min, args.speed_max):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error('Speed scales must be positive and finite')
    if args.speed_min is not None and args.speed_max is not None and args.speed_min > args.speed_max:
        parser.error('--speed-min must be <= --speed-max')
    if args.command == 'eval':
        if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
            parser.error('--playback-speed must be positive and finite')
        if not np.isfinite(args.hold_seconds) or args.hold_seconds < 0:
            parser.error('--hold-seconds must be nonnegative and finite')
    else:
        if args.split not in (None, 'train'):
            parser.error('Training must use --split train; validation/test are held out')
        for value in (args.heading_weight, args.yaw_rate_weight):
            if value is not None and (not np.isfinite(value) or value < 0):
                parser.error('Heading/yaw-rate reward weights must be finite and nonnegative')
        if args.ent_coef is not None and (not np.isfinite(args.ent_coef) or args.ent_coef < 0):
            parser.error('--ent-coef must be nonnegative and finite')
        for value in [args.target_kl, args.curriculum_max_speed, *(args.eval_speed_scales or [])]:
            if value is not None and (not np.isfinite(value) or value <= 0):
                parser.error('KL/curriculum/evaluation speed values must be positive and finite')
        if args.curriculum_max_speed is not None and args.eval_freq == 0:
            parser.error('Speed curriculum requires periodic validation')
        if args.eval_speed_scales and len(set(args.eval_speed_scales)) != len(args.eval_speed_scales):
            parser.error('--eval-speed-scales must not contain duplicates')
        if args.eval_time_profiles and len(set(args.eval_time_profiles)) != len(args.eval_time_profiles):
            parser.error('--eval-time-profiles must not contain duplicates')
        if args.eval_patience is not None and args.eval_freq == 0:
            parser.error('--eval-patience requires periodic validation')
        if args.eval_freq < 0:
            parser.error('--eval-freq must be nonnegative')
        if not args.resume and (args.batch_size < 2 or args.n_steps * args.num_envs < args.batch_size
                                or (args.n_steps * args.num_envs) % args.batch_size):
            parser.error('rollout size (n-steps * num-envs) must be divisible by batch-size >= 2')
        if args.learning_rate is not None and (not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
            parser.error('--learning-rate must be positive')
    return args


if __name__ == '__main__':
    multiprocessing.freeze_support()
    arguments = parse_args()
    if arguments.command == 'eval':
        evaluate(arguments)
    else:
        train(arguments)

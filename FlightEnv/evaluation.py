"""Fixed validation, precision metrics and competence-gated speed curriculum."""
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from FlightEnv.trajectory_dataset import FLIGHT_FAMILIES, evaluation_options

PRECISION_RMSE = .10
PRECISION_MAX_ERROR = .30


def evaluation_families(config):
    if config.get('mode') == 'hover':
        return ('hover',)
    family = config.get('trajectory_family', 'mixed')
    return FLIGHT_FAMILIES if family == 'mixed' else (family,)


def tracking_metrics(errors, completed, velocity_errors=None, reference_speeds=None, actual_speeds=None):
    errors = np.asarray(errors)
    rmse = float(np.sqrt(np.mean(errors**2)))
    maximum = float(errors.max())
    result = dict(rmse=rmse, max_error=maximum, p95_error=float(np.percentile(errors, 95)),
                  completed=bool(completed),
                  tracking_success=bool(completed and rmse <= PRECISION_RMSE and maximum <= PRECISION_MAX_ERROR),
                  legacy_tracking_success=bool(completed and rmse <= .35 and maximum <= 1.),
                  within_10cm_fraction=float(np.mean(errors <= .10)))
    if velocity_errors is not None:
        result['velocity_rmse'] = float(np.sqrt(np.mean(np.square(velocity_errors))))
    if reference_speeds is not None:
        speed = np.asarray(reference_speeds)
        fast = speed >= np.percentile(speed, 75)
        result['fast_quartile_rmse'] = float(np.sqrt(np.mean(errors[fast]**2)))
        result['mean_reference_speed'] = float(speed.mean())
        result['peak_reference_speed'] = float(speed.max())
    if actual_speeds is not None:
        result['mean_actual_speed'] = float(np.mean(actual_speeds))
    return result


def aggregate_metrics(results):
    return dict(episodes=len(results),
                mean_rmse=float(np.mean([r['rmse'] for r in results])),
                mean_p95_error=float(np.mean([r['p95_error'] for r in results])),
                completion_rate=float(np.mean([r['completed'] for r in results])),
                success_rate=float(np.mean([r['tracking_success'] for r in results])),
                legacy_success_rate=float(np.mean([r['legacy_tracking_success'] for r in results])))


def selection_key(results):
    # Never let an early crash's low, partially observed RMSE beat a completed
    # flight. Compare the same fixed suite, then precision success and error.
    return (sum(r['completed'] for r in results), sum(r['tracking_success'] for r in results),
            -float(np.mean([min(r['rmse'], 4.) for r in results])))


class TrackingEvaluation(BaseCallback):
    def __init__(self, env, config, directory, frequency, episodes, seed, save_fn,
                 speed_scales=None, time_profiles=None, curriculum_max_speed=None, patience=None):
        super().__init__()
        self.env = env
        self.config = config
        self.families = evaluation_families(config)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.episodes = max(episodes, len(self.families))
        options = evaluation_options(self.families, 'validation', config.get('dataset_dir'))
        rng = np.random.default_rng(seed)
        groups = [[option for i, option in enumerate(options) if i % len(self.families) == j]
                  for j in range(len(self.families))]
        for group in groups:
            rng.shuffle(group)
        if self.episodes > len(options):
            raise ValueError('Validation episodes exceed available unique trajectories')
        self.options = [groups[i % len(groups)][(i // len(groups)) % len(groups[i % len(groups)])]
                        for i in range(self.episodes)]
        self.speed_scales = speed_scales or [1.]
        default_profile = config.get('time_profile', 'quintic')
        self.time_profiles = time_profiles or (['quintic', 'cruise'] if default_profile == 'mixed' else [default_profile])
        self.seed = seed
        self.save_fn = save_fn
        self.best_key = (-1, -1, -np.inf)
        self.suite_signature = dict(options=self.options, seed=seed, speeds=self.speed_scales,
            time_profiles=self.time_profiles,
            environment={k: v for k, v in config.items() if k not in ('speed_min', 'speed_max')})
        previous = self.directory.parent / 'best/validation.json'
        if previous.exists():
            previous_summary = json.loads(previous.read_text())
            if previous_summary.get('suite_signature') != self.suite_signature:
                raise ValueError('Validation suite changed; choose a fresh run directory to preserve the old best model')
            self.best_key = tuple(previous_summary['selection_key'])
        self.curriculum_max_speed = curriculum_max_speed
        self.promotion_streak = 0
        self.patience = patience
        self.stale_evaluations = 0
        self.stop_requested = False

    def _on_training_start(self):
        # Record the starting policy under exactly the same acceptance rules.
        self.evaluate(initial=True)

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            self.evaluate()
        return not self.stop_requested

    def _evaluate_scenario(self, profile, speed):
        group = []
        for i, options in enumerate(self.options):
            obs, _ = self.env.reset(seed=self.seed+i,
                options=dict(options, speed_scale=speed, time_profile=profile))
            errors, rewards, velocities, reference_speeds, actual_speeds = [], [], [], [], []
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                errors.append(info['position_error'])
                velocities.append(info['velocity_error'])
                reference_speeds.append(info['reference_speed'])
                actual_speeds.append(info['actual_speed'])
                rewards.append(reward)
                if terminated or truncated:
                    break
            row = tracking_metrics(errors, truncated, velocities, reference_speeds, actual_speeds)
            row.update(family=info['trajectory_family'], trajectory_id=info['trajectory_id'],
                       seed=self.seed+i, scenario=f'{profile}_{speed:g}x', reward=float(sum(rewards)),
                       elapsed_seconds=info['elapsed_seconds'],
                       **{key: info[key] for key in ('time_profile', 'requested_speed_scale',
                           'effective_speed_scale', 'reference_duration', 'speed_limited', 'effective_time_profile', 'cruise_fallback')})
            group.append(row)
        return group

    def evaluate(self, initial=False):
        results, groups = [], {}
        for profile in self.time_profiles:
            for speed in self.speed_scales:
                scenario = f'{profile}_{speed:g}x'
                group = self._evaluate_scenario(profile, speed)
                groups[scenario] = aggregate_metrics(group)
                groups[scenario]['mean_effective_speed_scale'] = float(np.mean([r['effective_speed_scale'] for r in group]))
                groups[scenario]['speed_limited_fraction'] = float(np.mean([r['speed_limited'] for r in group]))
                groups[scenario]['cruise_fallback_fraction'] = float(np.mean([r['cruise_fallback'] for r in group]))
                results.extend(group)
                print(f'Validation {self.num_timesteps} {scenario}: RMSE={groups[scenario]["mean_rmse"]:.3f} m, '
                      f'completed={groups[scenario]["completion_rate"]:.0%}, '
                      f'precision success={groups[scenario]["success_rate"]:.0%}', flush=True)
                for key, value in groups[scenario].items():
                    self.logger.record(f'eval/{scenario}/{key}', value)
        summary = aggregate_metrics(results)
        summary.update(timesteps=self.num_timesteps, initial=initial, episodes=results, scenarios=groups,
                       acceptance=dict(rmse=PRECISION_RMSE, max_error=PRECISION_MAX_ERROR, must_complete=True),
                       selection_key=selection_key(results), seed=self.seed, suite_signature=self.suite_signature,
                       environment=dict(self.model.flight_env_config),
                       control_mode=('geometric_feedforward_plus_rl_residual'
                           if self.model.flight_env_config.get('profile') == 'residual' else 'direct_rl'))
        # Probe the speed currently present in training, not an unseen faster
        # bin. Keep this extra probe out of the fixed model-selection suite.
        current = self.model.flight_env_config.get('speed_max', 1.)
        curriculum_groups = dict(groups)
        if self.curriculum_max_speed is not None and current < self.curriculum_max_speed-1e-9:
            for profile in self.time_profiles:
                name = f'{profile}_{current:g}x'
                if name not in curriculum_groups:
                    curriculum_groups[name] = aggregate_metrics(self._evaluate_scenario(profile, current))
            summary['curriculum_probe'] = {f'{p}_{current:g}x': curriculum_groups[f'{p}_{current:g}x']
                                           for p in self.time_profiles}
        path = self.directory / f'{self.num_timesteps}.json'
        path.write_text(json.dumps(summary, indent=2) + '\n')
        for key in ('mean_rmse', 'completion_rate', 'success_rate'):
            self.logger.record(f'eval/{key}', summary[key])
        key = selection_key(results)
        if key > self.best_key:
            self.stale_evaluations = 0
            self.best_key = key
            self.save_fn(self.model, self.directory.parent / 'best/best_model.zip')
            (self.directory.parent / 'best/validation.json').write_text(json.dumps(summary, indent=2)+'\n')
        elif not initial:
            self.stale_evaluations += 1
        if initial:
            if not (self.directory.parent / 'initial_model.zip').exists():
                self.save_fn(self.model, self.directory.parent / 'initial_model.zip')
                (self.directory.parent / 'baseline.json').write_text(json.dumps(summary, indent=2)+'\n')
            else:
                (self.directory.parent / f'resume_baseline_{self.num_timesteps}.json').write_text(json.dumps(summary, indent=2)+'\n')
        else:
            self._advance_curriculum(curriculum_groups)
            if self.patience is not None and self.stale_evaluations >= self.patience:
                self.stop_requested = True
                reason = dict(timesteps=self.num_timesteps, reason='validation_plateau',
                              stale_evaluations=self.stale_evaluations, best_key=self.best_key)
                (self.directory.parent / 'early_stop.json').write_text(json.dumps(reason, indent=2)+'\n')
                print(f'Validation stopped training after {self.stale_evaluations} checks without improvement; best model retained.', flush=True)

    def _advance_curriculum(self, groups):
        if self.curriculum_max_speed is None:
            return
        config = self.model.flight_env_config
        current = config.get('speed_max', 1.)
        if current >= self.curriculum_max_speed - 1e-9:
            return
        # Graduation tests the current upper speed on both timing profiles.
        # The next speed is introduced only after two consecutive passes.
        probe = current
        names = [f'{p}_{probe:g}x' for p in self.time_profiles]
        if any(name not in groups for name in names):
            return
        ready = all(groups[name]['completion_rate'] >= .95 and
                    groups[name]['success_rate'] >= .75 and
                    groups[name]['mean_rmse'] <= .10 for name in names)
        self.promotion_streak = self.promotion_streak+1 if ready else 0
        if self.promotion_streak < 2:
            return
        maximum = round(min(current+.1, self.curriculum_max_speed), 6)
        minimum = config.get('speed_min', 1.)
        self.training_env.env_method('set_speed_range', minimum, maximum)
        config['speed_max'] = maximum
        self.promotion_streak = 0
        self.stale_evaluations = 0
        event = dict(timesteps=self.num_timesteps, speed_min=minimum, speed_max=maximum, probe=probe)
        with (self.directory.parent / 'curriculum.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(f'Speed curriculum promoted: {minimum:g}--{maximum:g}x', flush=True)

# User guide

[Back to the README](../README.md)

## Installation

Run all commands from the repository root. The tested environment is Linux x86_64, Python 3.11, PyTorch 2.7.1 CPU, Stable-Baselines3 2.7.0, Gymnasium 1.2.0, and PyBullet 3.2.7.

```bash
conda env create -f environment.yml
conda activate flight-rl
python -m pip check
```

Alternatively, create a Python 3.11 environment and install the CPU requirements:

```bash
conda create -n flight-rl python=3.11 pip -y
conda activate flight-rl
python -m pip install -r requirements-cpu.txt
# For the complete tested dependency set, use requirements-lock.txt instead.
```

Video export also requires FFmpeg and DejaVu Sans. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-dejavu-core
```

Training and headless evaluation work without a GPU or display. GUI playback needs a display server; EGL export needs compatible OpenGL drivers. CPU execution uses `--device cpu --torch-threads 1`; adjust `--num-envs` to available CPU and memory resources. CUDA training is supported by the CLI but has not been validated here. It requires a suitable CUDA-enabled PyTorch installation rather than the CPU requirements.

## Datasets

```bash
# Default: seed 2026; 256 training and 32 validation/test paths per family.
python -m FlightEnv.trajectory_dataset

# A separate, larger dataset; existing manifests are never overwritten.
python -m FlightEnv.trajectory_dataset --output data/trajectories_v3_large \
  --train-per-family 1024 --eval-per-family 128 --seed 2027
```

The nine families are `hover`, `line`, `circle`, `figure8`, `helix`, `slalom`, `wave`, `vertical_loop`, and `lissajous`. Scale, heading, reflection, and altitude vary by path. Train, validation, and test use separate random-number namespaces; the training CLI rejects held-out splits.

The default directory is `data/trajectories_v3/`, containing 2,304 training, 288 validation, and 288 test paths. Each original reference spans 10 seconds at 50 Hz, including 501 samples. Position, velocity, and acceleration are analytically consistent, with zero endpoint velocity and acceleration. Original reference limits are 4 m/s speed, 6 m/s² acceleration, 3 m/s² downward acceleration, and 0.8 m minimum altitude.

Each dataset includes:

- `train.npz`, `validation.npz`, and `test.npz`: time, position, velocity, acceleration, family labels, and seeds. State arrays have shape `(paths, 501, 3)`.
- `manifest.json`: generation parameters and per-path metadata.
- `trajectory_gallery.png`: examples from each family.

Pass the same `--dataset-dir` to initialization, training, and evaluation when using a custom dataset. Generated datasets stay outside Git. The six curated demos require the default seed and 32 test paths per family so that their trajectory IDs match.

### Difficulty and timing

`--difficulty` scales spatial displacement by `0.15 + 0.85 * difficulty`, with values from 0 to 1. It is separate from speed and GUI playback rate.

`--time-profile quintic` preserves the original smooth timing; `cruise` uses arc-length-based timing with additional time for tight bends; `mixed` samples both. `--speed-min` and `--speed-max` request reference speed multipliers without shrinking the path.

Retiming checks speed (5 m/s), acceleration (8 m/s²), jerk (40 m/s³), ideal thrust-to-weight ratio (1.5), and minimum vertical thrust ratio (0.45). It stretches time when needed and may fall back to a faster feasible quintic schedule. These reference checks do not prove full actuator or attitude feasibility. Report `effective_speed_scale`, `reference_duration`, `effective_time_profile`, and `cruise_fallback`, rather than treating the requested multiplier as achieved speed.

## Training

### Controller profiles

| Profile | Observation | Policy output |
| --- | ---: | --- |
| `tracking` | 59 values | Normalized collective thrust and three moments |
| `residual` | 67 values | Bounded corrections to geometric control |

Dimensions assume the default five future reference samples. Shared observations include attitude, body-frame velocity and angular velocity, altitude, relative position/velocity errors, reference acceleration, planned reference previews, and the preceding applied control. Residual observations also include the geometric command and previous residual.

The geometric controller uses position/velocity feedback, reference acceleration, and planned jerk to compute desired thrust, attitude, and angular velocity. It does not access future measured states. PPO corrections are scaled by `[0.10, 0.20, 0.20, 0.15]`, added to the geometric command, and passed through the same mixer and motor limits. A new residual actor has a zero output head, so deterministic evaluation starts at the disclosed baseline. The two profiles have different observation/action semantics and cannot be interchanged when resuming.

### Residual PPO and speed curriculum

```bash
python main.py train --profile residual --reward-profile precision \
  --difficulty 1 --split train --time-profile mixed \
  --speed-min 1 --speed-max 1.3 --curriculum-max-speed 1.5 \
  --learning-rate 0.00005 --ent-coef 0 --target-kl 0.01 \
  --n-steps 512 --batch-size 256 --n-epochs 5 \
  --timesteps 2000000 --num-envs 8 --torch-threads 1 --seed 42 \
  --save-freq 50000 --eval-freq 100000 --eval-episodes 16 --eval-patience 6 \
  --eval-speed-scales 1 1.3 1.5 --eval-time-profiles quintic cruise \
  --run-dir runs/residual_speed
```

The fixed validation suite uses two paths from each of eight flight families, two timing profiles, and three speed requests: 96 episodes. The curriculum tests the current training speed ceiling. Two consecutive checks must reach at least 95% completion, 75% strict success, and mean RMSE at most 0.10 m for each timing profile before adding 0.1 to the speed ceiling. Extra curriculum probes do not change the model-selection suite. Six validations without improvement stop training; promotion resets that counter.

The background launcher is an alternative to the foreground command. Use a new run directory:

```bash
python scripts/start_speed_training.py --run-dir runs/residual_background \
  --num-envs 4 --timesteps 2000000
tail -f runs/residual_background/training.log
```

The launcher records the PID, command, training log, and source snapshot. It never overwrites a nonempty run directory.

### Direct PPO

This example gradually increases spatial difficulty. It is a training recipe, not a guarantee of reproducing a historical checkpoint score.

```bash
python main.py train --profile tracking --reward-profile precision \
  --difficulty 0.1 --timesteps 1000000 --num-envs 4 \
  --eval-freq 100000 --eval-episodes 16 --run-dir runs/ppo_easy

python main.py resume --resume runs/ppo_easy/best/best_model.zip \
  --difficulty 0.5 --learning-rate 0.0001 --timesteps 1000000 \
  --num-envs 4 --eval-freq 100000 --eval-episodes 16 --run-dir runs/ppo_medium

python main.py resume --resume runs/ppo_medium/best/best_model.zip \
  --difficulty 1 --learning-rate 0.00005 --timesteps 2000000 \
  --num-envs 4 --eval-freq 100000 --eval-episodes 32 --run-dir runs/ppo_hard
```

For mixed-family training below difficulty 0.3, sampling uses hover, line, and circle; otherwise it uses the eight flight families. Use `--family figure8` for a single family or `--task hover` for hovering. Add `--dataset-dir data/trajectories_v3_large` to use a custom bank.

### Rewards and dynamics

Control runs at 50 Hz and physics at 200 Hz. Original episodes last 10 seconds; retimed episodes may differ. Initial pose, velocity, mass, and inertia are randomized by default; `--no-domain-randomization` disables this.

For `--reward-profile precision`, with position error `ep`, velocity error `ev`, and applied control `u`:

- Position: `0.25/(1+(ep/0.6)^2) + 0.35/(1+(ep/0.25)^2) + 0.8/(1+(ep/0.1)^2)`.
- Velocity: `0.5/(1+(ev/0.4)^2)`.
- Thrust alignment: `0.1 * dot(body_up, desired_thrust_direction)`; survival: `+0.1`.
- Default heading/yaw terms: `0.1 * cos(yaw) - 0.02 * body_yaw_rate^2`.
- Angular rate, control changes, and effort: `-0.002 * ||omega||^2 - 0.005 * ||delta_u||^2 - 0.003 * ||u||^2`.
- Residual control adds `-0.01 * ||residual||^2`.
- Collision or out-of-bounds termination adds `-30`.

Terms are exposed through `info['reward_*']`. The `legacy` reward uses wider tolerances and is distinct from the `legacy` environment profile. Roll/pitch beyond 85 degrees triggers termination: vertical loops describe the position path, not inverted flips. Control uses simulation mass/inertia parameters; real-world flight has not been validated.

### Checkpoints and resume

| Output | Purpose |
| --- | --- |
| `initial_model.zip`, `baseline.json` | Initial fixed-validation baseline |
| `best/best_model.zip`, `best/validation.json` | Best fixed-validation checkpoint and metrics |
| `final_model.zip` | Latest policy, optimizer, and total interaction count |
| `checkpoints/ppo_*_steps.zip` | Periodic checkpoints |
| `validation/`, `curriculum.jsonl`, `early_stop.json` | Validation and applicable curriculum/stop events |
| `config.json`, `resume_*.json` | Actual environment and PPO settings |
| `monitor*.csv`, `tensorboard/` | Episode logs and learning curves |

Selection ranks completed episodes, strict successes, then mean RMSE. The best checkpoint can remain at zero training steps. Test data never selects the checkpoint. `--eval-freq 0` disables validation and its initial/best checkpoint outputs.

```bash
python main.py resume --resume runs/residual_speed/best/best_model.zip \
  --run-dir runs/residual_resumed --timesteps 1000000 --num-envs 8 \
  --learning-rate 0.00005 --eval-freq 100000 --eval-episodes 16 --eval-patience 6 \
  --eval-speed-scales 1 1.3 1.5 --eval-time-profiles quintic cruise \
  --curriculum-max-speed 1.5

tensorboard --logdir runs
```

`--timesteps` adds total transitions across all workers, rounded up to a complete PPO rollout. Resume preserves policy and optimizer state; `--n-steps`, `--batch-size`, and `--n-epochs` apply only to new models. Learning rate, entropy coefficient, and target KL can be overridden. Same-directory resume checks validation-suite consistency; changed suites require a new directory. Ctrl+C or SIGTERM requests a save, while SIGKILL cannot. Resume starts new episodes rather than restoring in-flight physics state.

## Evaluation

Create a baseline without training or downloading weights:

```bash
python scripts/create_baseline.py
python main.py eval --model runs/residual_baseline/baseline.zip \
  --all-trajectories --seed 2026 --difficulty 1 \
  --time-profile cruise --speed-min 1.5 --speed-max 1.5 \
  --output-dir runs/baseline_test_all
```

Evaluation defaults to the test split and saved environment settings. `--all-trajectories` visits each selected path once and overrides `--episodes`; add `--family slalom` to evaluate just that family. Default complete flight evaluation covers 256 paths, excluding hover.

Strict success requires completion, position RMSE at most 0.10 m, and maximum error at most 0.30 m. The older 0.35 m / 1 m criterion is retained as `legacy_tracking_success`. Errors only cover the time survived, so report completion alongside RMSE.

Outputs include `metrics.json`, actual-flight `episode_*.csv`, reference `reference_*.csv`, `tracking.png`, and `family_comparison.png`. Summary plots use the first evaluated path per family rather than selecting the best-looking cases.

The published [256-episode benchmark](benchmarks/residual_baseline_cruise_150.json) records 254 completions and 227 strict successes for the zero-residual baseline, with mean per-episode RMSE 0.05674575 m and mean effective speed multiplier 1.429. Collisions at test IDs 257 and 184, plus all precision failures, remain in the report.

In a separate fixed 96-episode validation suite, the initial baseline achieved 3.429 cm RMSE; the final learned residual achieved 3.540 cm. Both passed 90/96 episodes. Training stopped at one million steps without beating the initial baseline. These validation results use a different suite from the complete test benchmark and should not be pooled.

## Visualization and export

GUI playback supports **F** to fit the whole path, **T** for a top view, and **Space** to pause. `--playback-speed` changes viewing pace only; reference speed is controlled by the speed arguments. GUI performance also depends on rendering throughput.

```bash
python main.py eval --model runs/residual_baseline/baseline.zip \
  --family figure8 --episodes 1 --seed 2026 \
  --time-profile cruise --speed-min 1.5 --speed-max 1.5 \
  --render --playback-speed 1 --hold-seconds 10
```

If Linux hardware OpenGL initialization hangs, the following Mesa fallback was tested on the development machine:

```bash
LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa \
python main.py eval --model runs/residual_baseline/baseline.zip \
  --family vertical_loop --episodes 1 --render --hold-seconds 10
```

Software rendering can be slow. Servers without a display can evaluate headlessly or export through EGL. EGL export does not require the Mesa variables above.

```bash
python -m FlightEnv.difficult_showcase \
  --model runs/residual_baseline/baseline.zip --output exports/difficult_flights_6
python scripts/export_readme_demos.py \
  --input-dir exports/difficult_flights_6 --output docs/media
```

The exporter runs actual physics, requires strict tracking success, then replays recorded positions and orientations in a separate PyBullet scene. It uses the original drone scale, linear position interpolation, and quaternion SLERP. It does not snap flights to the reference, alter physics states, or remove failed segments. A failing rollout raises an error.

- Default output: six 1280x720, 30 fps H.264 MP4s with faststart, a 0.5-second opening hold, and a 1.5-second final hold.
- EGL uses 2x supersampling; `--renderer tiny` is a slower CPU fallback for the detailed original model.
- `--preview` generates posters and metrics only. `--audit <metrics.json>` compares matching trajectory IDs/seeds against earlier recorded rollouts.
- The output includes a local `index.html`, posters, `*_rollout.npz`, metrics, a manifest, verification results, and a ZIP in the parent directory.
- README GIF files remain 640x360 at 15 fps, with infinite looping. The README displays them at 360 pixels wide in a wrapping layout. The [media manifest](media/manifest.json) records source-video and GIF hashes.

Training artifacts and full exports stay in ignored `runs/` and `exports/`; selected GIFs and published metrics are versioned. The historical pure-PPO exporter remains available as `python -m FlightEnv.showcase --model <tracking-checkpoint.zip>` and uses the older, looser success thresholds.

## Development and compatibility

```bash
python -m pip check
python -m pytest -q
python -m compileall -q main.py FlightEnv scripts tests
git diff --check
```

The 67 tests cover applied actions, Gymnasium semantics, seeding, client isolation, trajectory derivatives/splits, mixing, rewards, retiming, residual equivalence, curricula, stopping, and checkpoint round trips. Tests create temporary datasets. [Release verification](benchmarks/release_verification.json) also records clean-source generation, initialization, training/resume, and reproduction of the original five demos.

Current model format 4 supports tracking and residual tasks; formats 2/3 remain loadable. Original unversioned checkpoints in the root and `results/Jack/` use incompatible physical-thrust semantics. They are retained as historical artifacts; create a new model for the current tasks. The bundled ZIP trajectory library remains available to the legacy profile and compatibility tests.

Historical experiment reports are preserved in Chinese: [V3 PPO](../TRAINING_V3.md), [precision and retiming](../TRAINING_PRECISION.md), [residual control](../TRAINING_SPEED.md), and [legacy fixes](../VALIDATION.md). Their `runs/...` references refer to local historical experiments; those checkpoints are not part of the source release.

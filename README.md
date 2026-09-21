# Flight-RL

基于 **PyBullet · Gymnasium · Stable-Baselines3 PPO** 的四旋翼轨迹跟踪实验。提供纯 PPO 控制与几何前馈 + PPO 残差控制，支持多样化轨迹生成、并行训练、速度课程、断点续训、独立评估和飞行视频导出。

<table>
  <tr>
    <td align="center"><b>三维 8 字</b><br><img src="docs/media/figure8.gif" width="480" alt="PyBullet 三维八字跟踪：青色参考轨迹，橙色实际航迹"><br>RMSE 3.53 cm · 参考峰值 2.99 m/s</td>
    <td align="center"><b>双圈螺旋爬升</b><br><img src="docs/media/helix.gif" width="480" alt="PyBullet 双圈螺旋爬升跟踪"><br>RMSE 4.74 cm · 参考峰值 2.43 m/s</td>
  </tr>
  <tr>
    <td align="center"><b>垂直回环</b><br><img src="docs/media/vertical_loop.gif" width="480" alt="PyBullet 垂直空间回环跟踪"><br>RMSE 3.28 cm · 参考峰值 2.58 m/s</td>
    <td align="center"><b>三维交叉曲线</b><br><img src="docs/media/lissajous.gif" width="480" alt="PyBullet 三维 Lissajous 曲线跟踪"><br>RMSE 3.97 cm · 参考峰值 2.38 m/s</td>
  </tr>
  <tr>
    <td colspan="2" align="center"><b>连续蛇形</b><br><img src="docs/media/slalom.gif" width="640" alt="PyBullet 连续蛇形转弯跟踪"><br>RMSE 2.80 cm · 参考峰值 3.23 m/s</td>
  </tr>
</table>

**Demo 说明：** 青色虚线为参考轨迹，橙色实线为实际飞行轨迹。以上是独立测试集中的 5 条精选成功案例，使用**几何前馈 + 零 RL 残差基线**，不能视为 PPO 学习收益。该轮训练后的残差没有超过初始基线。GIF 保留完整飞行、真实偏差与原播放速度，循环播放；空间回环不等于机体翻滚。轨迹编号、随机种子、实际速度倍率及文件哈希见 [Demo 清单](docs/media/manifest.json)。

[环境配置](#环境配置) · [快速运行](#快速运行) · [数据生成](#数据生成) · [训练方法](#训练方法) · [评估与可视化](#评估与可视化) · [视频与-gif](#视频与-gif) · [实验结果](#实验结果) · [开发与测试](#开发与测试)

## 环境配置

已验证环境：Linux x86_64、Python 3.11、PyTorch 2.7.1 CPU、Stable-Baselines3 2.7.0、Gymnasium 1.2.0、PyBullet 3.2.7。训练和无界面评估不需要 GPU 或桌面；GUI 需要可用的显示服务，GPU 视频导出需要 EGL/OpenGL。

在仓库根目录执行后续命令：

```bash
git clone https://github.com/calcualatexzy/Flight-RL.git
cd Flight-RL
conda env create -f environment.yml
conda activate flight-rl
python -m pip check
```

也可以手动创建环境。常规安装固定主要依赖，完整锁定文件记录本项目实测的传递依赖：

```bash
conda create -n flight-rl python=3.11 pip -y
conda activate flight-rl
python -m pip install -r requirements-cpu.txt
# 如需安装完整实测版本集合，改用：
# python -m pip install -r requirements-lock.txt
```

渲染视频/GIF 额外需要 FFmpeg 和 DejaVu Sans 字体。Ubuntu/Debian 安装示例：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-dejavu-core
ffmpeg -version
```

默认 `--device cpu --torch-threads 1`，每个仿真进程使用 1 个 PyTorch 线程。`--num-envs` 控制并行环境数，按 CPU 核数和内存调整。CPU 环境中的 PyTorch 不支持 CUDA；需要 CUDA 时，应在单独环境中安装匹配机器的 PyTorch，再安装 `requirements.txt`，并传 `--device cuda`。本仓库没有验证 CUDA 训练的性能或结果。

## 快速运行

下面的流程不依赖预训练权重。先生成数据，再创建输出恒为零的残差策略；无人机由明确的几何控制基线驱动，**这一步没有执行 PPO 训练**。

```bash
# 默认生成 2,304 / 288 / 288 条 train / validation / test 轨迹
python -m FlightEnv.trajectory_dataset

# 创建可保存、评估和续训的零残差检查点
python scripts/create_baseline.py

# 无界面测试八类飞行轨迹，各一条
python main.py eval --model runs/residual_baseline/baseline.zip \
  --episodes 8 --seed 2026 --difficulty 1 \
  --time-profile cruise --speed-min 1.5 --speed-max 1.5 \
  --output-dir runs/baseline_eval
```

评估输出 `metrics.json`、逐条参考/实飞 CSV、轨迹对比图和误差图。需要在 PyBullet 窗口观看一条轨迹：

```bash
python main.py eval --model runs/residual_baseline/baseline.zip \
  --family figure8 --episodes 1 --seed 2026 \
  --time-profile cruise --speed-min 1.5 --speed-max 1.5 \
  --render --playback-speed 1 --hold-seconds 10 \
  --output-dir runs/baseline_figure8
```

已存在的数据或基线不会被生成脚本覆盖；需要重新生成时选择新的 `--output`。改变数据目录后，创建基线、训练和评估都应使用相同的 `--dataset-dir`。

## 数据生成

### 轨迹类型与集合划分

| `--family` | 内容 |
| --- | --- |
| `hover` | 不同高度悬停 |
| `line` | 不同朝向、距离和高度变化的直线 |
| `circle` | 圆/椭圆，带高度起伏 |
| `figure8` | 三维 8 字交叉 |
| `helix` | 一圈或两圈螺旋爬升 |
| `slalom` | 连续蛇形转弯 |
| `wave` | 前进中的上下波浪 |
| `vertical_loop` | 竖直空间回环 |
| `lissajous` | 三维多频交叉曲线 |

默认根目录为 `data/trajectories_v3/`，种子为 `2026`。每类分别生成 256 条训练、32 条验证、32 条测试轨迹，合计 **2,880 条**。三个集合使用不同 RNG 命名空间；训练入口拒绝使用验证/测试集合。默认完整飞行评估覆盖除悬停外的八类，共 256 条测试轨迹。

原始轨迹长 10 秒，以 0.02 秒采样，含首尾共 501 个点。位置、速度、加速度来自一致的解析表达式；五次时间曲线使起终点速度和加速度为零。尺度、朝向、镜像和高度均随机化。原始参考约束为：峰值速度 ≤ 4 m/s、加速度 ≤ 6 m/s²、向下加速度 ≤ 3 m/s²、最低高度 ≥ 0.8 m。

### 自定义数据规模

```bash
# 保留默认数据，另生成更大的库：每类 1,024 / 128 / 128 条
python -m FlightEnv.trajectory_dataset --output data/trajectories_v3_large \
  --train-per-family 1024 --eval-per-family 128 --seed 2027

# 使用新库训练
python main.py train --profile residual --reward-profile precision \
  --dataset-dir data/trajectories_v3_large --difficulty 1 \
  --timesteps 1000000 --num-envs 4 --run-dir runs/residual_large
```

每套数据包含 `train.npz`、`validation.npz`、`test.npz`、`manifest.json` 和 `trajectory_gallery.png`。NPZ 存储 `time`、`position`、`velocity`、`acceleration`、`families`、`seeds`；状态数组形状为 `(轨迹数, 501, 3)`。Manifest 保存生成参数和逐条轨迹元数据。生成数据不纳入 Git。

### 难度与速度

- `--difficulty` 在 `[0, 1]`，空间幅度系数为 `0.15 + 0.85 × difficulty`。这是空间课程，不是播放速度。
- `--time-profile quintic` 使用原始五次时间安排；`cruise` 按弧长安排巡航，并对急弯分配更多时间；`mixed` 在两者间采样。
- `--speed-min / --speed-max` 是请求的参考时间倍率。重定时不缩小路径，按速度、加速度、jerk 和推力余量检查后必要时延长时间。急弯可能回退到更快且可行的五次时间安排。
- 当前重定时检查上限：速度 5 m/s、加速度 8 m/s²、jerk 40 m/s³、理想推重比 1.5、竖直推力比下限 0.45。这是参考轨迹约束，不是完整电机/姿态可达性证明。
- 实际结果必须读取 `effective_speed_scale`、`reference_duration`、`effective_time_profile`、`cruise_fallback`，不能把请求的 1.5× 当作实际达到的速度。

## 训练方法

### 两种控制任务

| 配置 | 观测 | 策略输出 | 适用实验 |
| --- | ---: | --- | --- |
| `--profile tracking` | 59 维 | 归一化总推力与三个力矩 | 纯 PPO 直接控制，姿态稳定和纠偏均由策略学习 |
| `--profile residual` | 67 维 | 几何控制命令上的有界修正 | 检查学习残差是否能超过可复现控制基线 |

以上维度对应默认 5 个未来目标。基础观测包括姿态旋转矩阵、机体系速度/角速度、高度、目标位置/速度相对误差、参考加速度、未来 0.1～0.5 秒的参考目标和上一步执行命令。残差任务额外加入几何控制命令和上一残差。

残差控制使用位置/速度反馈、参考加速度及计划参考 jerk，求期望姿态和角速度；不访问未来实测状态。PPO 输出乘以 `[0.10, 0.20, 0.20, 0.15]` 后叠加到总推力/力矩命令，仍经过相同的静态混控和电机限制。新残差策略输出头初始化为零，确定性评估等于几何基线。两种任务的观测和动作语义不同，**不能通过修改 profile 相互续训**。

### 残差 PPO 与速度课程

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

固定验证使用八类各两条路径 × 两种时间安排 × 三档请求倍率，共 96 回合。验证**当前训练速度上限**；连续两次在各时间安排下达到完成率 ≥95%、严格成功率 ≥75%、平均 RMSE ≤10 cm，才提升 0.1 倍。额外的当前速度探测不混入固定选模集合。连续六次验证无改善则提前停止，升速后重新计数。

也可以用后台启动器执行同一套配置，并记录日志、PID、命令和源码快照。以下是上述前台命令的替代方式，应使用新的运行目录：

```bash
python scripts/start_speed_training.py --run-dir runs/residual_background
# 可传 --num-envs 4 --timesteps 1000000 调整资源和上限
tail -f runs/residual_background/training.log
```

### 纯 PPO 的空间课程

纯 PPO 可从低空间难度开始，再从上一阶段的验证最佳模型继续。以下是当前精度奖励下的训练配置示例，不保证复制某个历史检查点的成绩：

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

`mixed` 在训练难度低于 0.3 时采样悬停/直线/圆，否则均匀采样八类飞行路径。验证覆盖配置所选的飞行类别。可以用 `--family figure8` 专练一类，或 `--task hover` 训练悬停。

### 奖励与动力学

默认控制 50 Hz、物理仿真 200 Hz；原始回合 10 秒，重定时后时长可变。默认开启初始位置、姿态、速度、质量和惯量随机化，`--no-domain-randomization` 可关闭。

`--reward-profile precision` 的主要分项如下，`ep / ev` 为位置/速度误差范数：

| 分项 | 奖励 |
| --- | --- |
| 多尺度位置精度 | `0.25/(1+(ep/0.6)^2) + 0.35/(1+(ep/0.25)^2) + 0.8/(1+(ep/0.1)^2)` |
| 速度跟随 | `0.5/(1+(ev/0.4)^2)` |
| 推力方向 | `0.1 × dot(机体向上方向, 期望推力方向)` |
| 存活 | `+0.1` |
| 航向与偏航角速度 | 默认 `+0.1 × cos(yaw)`、`-0.02 × body_yaw_rate²` |
| 角速度、动作变化、动作幅度 | `-0.002 × ‖ω‖²`、`-0.005 × ‖Δcontrol‖²`、`-0.003 × ‖control‖²` |
| 残差幅度（仅 residual） | `-0.01 × ‖residual‖²` |
| 碰撞或越界 | `-30` 并终止 |

`legacy` 奖励保留较宽的位置/速度容差以兼容历史实验；它与 `--profile legacy` 是不同概念。奖励分项通过 `info['reward_*']` 返回。横滚/俯仰超过 85° 等越界条件会终止回合，因此当前不是倒飞翻滚任务。质量、惯量和执行器参数来自仿真模型，未验证实机部署。

### 保存、续训与日志

| 文件 | 内容 |
| --- | --- |
| `initial_model.zip`、`baseline.json` | 初始固定验证基线；启用 tracking/residual 验证时保存 |
| `best/best_model.zip`、`best/validation.json` | 固定验证选出的最佳策略和指标，可能仍为 0 步初始化 |
| `final_model.zip` | 结束或正常中断时保存的最新策略、优化器和总交互步数 |
| `checkpoints/ppo_<步数>_steps.zip` | 定期检查点 |
| `validation/`、`curriculum.jsonl`、`early_stop.json` | 验证、升速与提前停止记录；后两者按事件生成 |
| `config.json`、`resume_*.json` | 环境及实际 PPO 参数 |
| `monitor*.csv`、`tensorboard/` | 回合日志和训练曲线 |

选模按完整回合数、严格成功回合数、平均 RMSE 依次比较。测试集不参与选模。`--eval-freq 0` 关闭验证，此时没有最佳模型和初始验证基线文件。

```bash
python main.py resume --resume runs/residual_speed/best/best_model.zip \
  --run-dir runs/residual_resumed --timesteps 1000000 --num-envs 8 \
  --learning-rate 0.00005 --eval-freq 100000 --eval-episodes 16 --eval-patience 6 \
  --eval-speed-scales 1 1.3 1.5 --eval-time-profiles quintic cruise \
  --curriculum-max-speed 1.5

tensorboard --logdir runs
```

`--timesteps` 表示本次新增的总环境交互步数，包含所有并行环境，并向上取整到完整 PPO rollout。恢复保留网络和优化器状态；`--n-steps / --batch-size / --n-epochs` 只作用于新模型，学习率、熵系数和目标 KL 可显式覆盖。同目录恢复会检查固定验证配置是否一致；更改验证集合应选择新目录。Ctrl+C / SIGTERM 会请求保存，SIGKILL 无法保存。续训从新回合开始，不恢复中途物理状态。

## 评估与可视化

### 全部独立测试轨迹

```bash
python main.py eval --model runs/residual_baseline/baseline.zip \
  --all-trajectories --seed 2026 --difficulty 1 \
  --time-profile cruise --speed-min 1.5 --speed-max 1.5 \
  --output-dir runs/baseline_test_all
```

默认使用 test 集和检查点保存的环境配置。`--all-trajectories` 每条所选轨迹恰好测试一次，并覆盖 `--episodes`；加 `--family slalom` 可测试全部蛇形路径。

严格成功同时要求 **完整飞完、位置 RMSE ≤ 0.10 m、最大位置误差 ≤ 0.30 m**。历史宽松标准（0.35 m / 1 m）单独保存为 `legacy_tracking_success`。RMSE 仅统计实际存活时段，必须和完成率一起看，不能忽略提前坠毁的回合。

输出包括每回合/每类指标、实际飞行 `episode_*.csv`、参考 `reference_*.csv`、`tracking.png` 和 `family_comparison.png`。总览采用每类首次评估的路径，不按跟踪效果挑图；完整原始结果保留在 CSV/JSON。

### GUI 与渲染问题

青色表示参考，橙色表示实际，绿/紫标记起终点；窗口支持 **F** 适配全景、**T** 切换俯视、**空格** 暂停/继续。`--playback-speed` 只控制 GUI 的播放节奏，不改变目标轨迹速度；实际 GUI 帧率还取决于渲染性能。

若 Linux 硬件 OpenGL 初始化卡住，可尝试已在开发机器上验证的 Mesa 软件渲染：

```bash
LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa \
python main.py eval --model runs/residual_baseline/baseline.zip \
  --family vertical_loop --episodes 1 --render --hold-seconds 10
```

软件渲染可能较慢。无桌面的服务器可直接运行不带 `--render` 的评估，或使用下述 EGL 离屏视频导出。EGL 导出不需要上述 Mesa 环境变量。

## 视频与 GIF

README 中的 5 个 GIF 已随源码提交，无需本地运行就能显示。复现它们需要默认 `seed=2026`、每类 32 条 test 路径的数据库，才能对应清单中的轨迹编号。

```bash
# 前提：完成快速运行中的数据和基线生成
python -m FlightEnv.difficult_showcase \
  --model runs/residual_baseline/baseline.zip --output exports/difficult_flights_5

# 将 5 个完整 MP4 转为 README GIF；自动验证动画、循环和时长
python scripts/export_readme_demos.py \
  --input-dir exports/difficult_flights_5 --output docs/media
```

导出器先执行真实 PyBullet 仿真，通过严格跟踪检查后，再用独立 PyBullet 场景按测得的位置和姿态回放渲染。位置按时间插值，姿态用 SLERP；不吸附参考轨迹、不改写物理状态、不剪掉失败片段。模型不通过标准时导出会报错，不会把它标成成功示例。

- 默认生成 **5 个 MP4**，1280×720、30 fps、H.264/yuv420p、faststart，附 0.5 秒开头和 1.5 秒结尾停留。
- 默认使用 EGL 离屏渲染和 2 倍超采样；`--renderer tiny` 可作为 CPU 回退，但原始机模面数较多、渲染较慢。
- `--preview` 只生成封面和测试数据；`--audit <先前评估的metrics.json>` 可核对相同 ID/种子的逐步物理结果。
- 输出目录中有 `index.html` 预览页、封面、`*_rollout.npz`、逐条指标、`manifest.json` 和 `verification.json`，父目录中生成 ZIP 包。
- GIF 默认 640×360、15 fps、无限循环；全部帧可解码，时长与源视频一致到采样精度。公开媒体清单记录来源视频和 GIF 的 SHA-256。

完整导出、训练模型和日志保存在被忽略的 `exports/`、`runs/`；只将经过选择的 README GIF 与公开指标纳入版本管理。原四条纯 PPO 视频导出入口 `python -m FlightEnv.showcase --model <tracking模型.zip>` 继续保留，其成功阈值属于历史宽松标准。

## 实验结果

当前公开的完整基线结果来自默认测试集全部 256 条飞行路径，完整难度、域随机化开启、请求巡航 1.5×、初始种子 `2026 + episode_index`：

| 指标 | 结果 |
| --- | ---: |
| 完整飞行 | 254 / 256（99.22%） |
| 严格精度成功 | 227 / 256（88.67%） |
| 每回合位置 RMSE 的均值 | 5.675 cm |
| 平均实际时间倍率 | 1.429× |

[完整 256 回合指标](docs/benchmarks/residual_baseline_cruise_150.json) 保留未达标与提前结束的回合，包括 ID 257（Lissajous）、ID 184（Slalom）的碰撞。README 中的 5 条 Demo 是精选成功案例，不是整体成功率。

残差训练曾在 200,000 / 400,000 步分别升至 1.4 / 1.5×，在 1,000,000 步因验证持续无改善提前停止。同一组 96 回合固定验证中，初始零残差基线 RMSE 为 3.429 cm，末次 PPO 残差为 3.540 cm，均为 90/96 严格达标；验证最佳仍是 0 步基线。**本次实验没有证明 RL 残差超过几何控制。** 验证与上述全部 test 的设置不同，不能混用两者的均值。

历史记录见 [V3 纯 PPO](TRAINING_V3.md)、[精度奖励与重定时](TRAINING_PRECISION.md)、[残差控制与提速](TRAINING_SPEED.md)、[原始问题修复](VALIDATION.md)。这些报告中的 `runs/...` 是历史本地实验位置，检查点未随本次源码发布；新用户应使用本 README 的生成/训练入口。

## 开发与测试

```bash
python -m pip check
python -m pytest -q
python -m compileall -q main.py FlightEnv scripts tests
git diff --check
```

目前 67 项测试覆盖动作实际执行、Gymnasium 接口、随机种子和重置、多个 PyBullet 客户端隔离、解析导数与数据划分、混控响应、奖励、参考重定时、残差基线一致性、验证与速度课程、提前停止，以及检查点保存/恢复。测试会生成临时小型数据，不依赖开发机器上的训练数据或模型。另验证了临时干净源码目录中的数据生成、基线导出、短训练、续训和评估流程，详见 [版本整理记录](CHANGELOG.md)。

```text
FlightEnv/
  env.py, quadrotor.py       物理环境、飞行器与 legacy 接口
  trajectory_dataset.py     解析轨迹生成与独立集合
  tracking_env.py            纯 PPO 跟踪任务
  residual_env.py            几何前馈 + 有界 PPO 残差
  retiming.py                保持几何路径的参考时间安排
  evaluation.py             固定验证、选模、课程与早停
  visualization.py          GUI 航迹对比及离线图
  showcase.py               PyBullet 相机、布局、视频/GIF 编码
  difficult_showcase.py      五条困难轨迹实测与 MP4 导出
main.py                     train / resume / eval 命令入口
scripts/                    基线初始化、后台训练、README GIF 生成
tests/                      自动检查
docs/media/                 可直接展示的五个 GIF 与来源清单
docs/benchmarks/            保留失败样本的公开指标
environment.yml             Conda 环境
requirements*.txt           主要依赖、CPU 安装入口、完整实测锁定版本
```

模型格式 4 支持 `tracking` 和 `residual`，加载器继续兼容格式 2/3。仓库历史根目录和 `results/Jack/` 下的无格式旧检查点使用不同的物理推力接口，不能直接用于当前任务；旧文件保留，新模型请从头创建。旧 ZIP 轨迹库仍供 `--profile legacy` 和兼容测试使用。

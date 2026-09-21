# V3 多样化轨迹与重新训练记录（2026-09-18）

已生成 2,880 条轨迹并实际完成课程训练与航向约束微调。本次各训练会话合计执行 **5,005,312 步**；按独立验证集选中的交付模型累计计数为 **3,100,000 步**。由于各阶段从历史最佳 checkpoint 分支继续训练，两者含义不同。

完整难度、域随机化开启、256 条独立飞行测试轨迹上，**252/256 完成且跟踪达标（98.4375%）**，每回合位置 RMSE 的平均值为 **0.1419 m**。仍有 4 条多频 Lissajous 路径提前结束；不宣称全部复杂轨迹都已可靠解决。

## 可用模型与结果

- 模型：runs/v3_stable/best/best_model.zip（本地文件 `runs/v3_stable/best/best_model.zip`）。
- 完整指标：test_all/metrics.json（本地文件 `runs/v3_stable/test_all/metrics.json`）。
- 八类对比图：family_comparison.png（本地文件 `runs/v3_stable/test_all/family_comparison.png`）。每类取第一次评估，包含失败例，没有挑选最漂亮的回合。
- 八字单独对比图：demo_figure8/tracking.png（本地文件 `runs/v3_stable/demo_figure8/tracking.png`）。
- 数据清单：[manifest.json](data/trajectories_v3/manifest.json)，轨迹形状：[trajectory_gallery.png](data/trajectories_v3/trajectory_gallery.png)。
- 模型 SHA-256：`e19c23783954145b6e6b22c6fc69c7372e29fa8621b5060db9db99a2d7cdb319`。

## 数据与设计变化

轨迹库包含悬停、直线、圆/椭圆、三维八字、螺旋、蛇形、波浪、竖直回环和三维 Lissajous。每类训练 256 条、验证 32 条、测试 32 条，总计 2,304 / 288 / 288。混合飞行测试不含悬停，因此评估 8 × 32 = 256 条。三个集合使用独立 RNG 命名空间；测试集未参与梯度更新或 checkpoint 选择。

原始轨迹时长 10 秒，解析生成一致的位置、速度、加速度，起终点速度/加速度为零。随机尺度、朝向、镜像和高度；全库验证最低高度 0.8 m、峰值速度不超过 4 m/s、加速度不超过 6 m/s²（float32 存储误差约 5e-7）、向下加速度不超过 3 m/s²。难度按 `0.15 + 0.85*difficulty` 同时缩放相对位移和导数。

59 维相对观测包括当前/未来位置速度误差、参考加速度、姿态和上一动作。策略输出总推力与三个机体系力矩，静态混控分配到四个电机；没有隐藏的 PID 跟踪器。混控依赖仿真质量/惯量，结果尚不代表实机控制性能。

奖励分解为位置、速度、期望推力方向、存活、角速度、动作变化、动作幅度和失败惩罚。完整难度训练发现螺旋转弯时偏航自旋导致失稳，随后增加 `0.1*cos(yaw)` 和 `-0.02*body_yaw_rate²`。机体允许倾斜完成转弯；空间回环不包含倒飞/翻滚，仍有 85° 横滚/俯仰终止条件。精确公式和命令见 [README.md](README.md)。

## 实际训练

CPU、4 个 PyBullet 环境、每个 PyTorch 进程 1 线程；控制 50 Hz、物理 200 Hz。PPO 两层 128 单元 MLP，gamma 0.995，每环境 rollout 512，batch 256，每次更新最多 10 epochs，target KL 0.03。

| 阶段 | 难度 | 本次起始→结束步数 | 验证最佳步数 | 验证 RMSE (m) | 验证跟踪成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `runs/v3_easy` | 0.1 | 0 → 1,001,472 | 700,000 | 0.0326 | 100.00% |
| `runs/v3_medium` | 0.5 | 700,000 → 1,701,472 | 900,000 | 0.0589 | 100.00% |
| `runs/v3_hard` | 1 | 900,000 → 2,900,896 | 2,700,000 | 0.1421 | 96.88% |
| `runs/v3_stable` | 1 | 2,500,000 → 3,501,472 | 3,100,000 | 0.1415 | 100.00% |

学习率依次为 3e-4、1e-4、5e-5、1e-4；最后一阶段启用航向奖励和偏航角速度惩罚，前面权重为 0。最后阶段从 `v3_hard` 当时 2,500,000 步的最佳模型拷贝为 `v3_stable/initial_model.zip` 后开始；此前阶段后来得到 2,700,000 步最佳模型，没有据此替换已经开始的微调。各运行目录保留训练日志、配置、周期 checkpoint、最佳和最终模型。

早期 easy/medium 每轮验证 8 个回合；hard 每轮 32 个回合，按类型固定种子抽样，存在重复路径；stable 改为每类 4 条不重复路径，共 32 条固定验证轨迹。因此表中不同阶段的难度、权重和验证路径不同，不能当作受控消融对照。后期 stable 模型仍有退化，保留 3,100,000 步验证最佳模型供使用，未按测试结果重新挑选 checkpoint。

## 独立测试

确定性策略、完整难度 1.0、域随机化开启、每回合 10 秒。逐条覆盖 test 集全部飞行轨迹，每条恰好一次；初始状态随机种子为 `2026 + episode_index`。成功条件为飞满回合、RMSE ≤ 0.35 m 且最大误差 ≤ 1 m。

| 类型 | 轨迹数 | 完成/达标 | 平均位置 RMSE (m) |
| --- | ---: | ---: | ---: |
| circle | 32 | 32/32 | 0.1235 |
| figure8 | 32 | 32/32 | 0.1611 |
| helix | 32 | 32/32 | 0.1765 |
| line | 32 | 32/32 | 0.0720 |
| lissajous | 32 | 28/32 | 0.1724 |
| slalom | 32 | 32/32 | 0.1509 |
| vertical_loop | 32 | 32/32 | 0.1753 |
| wave | 32 | 32/32 | 0.1038 |

失败回合保留在原始结果中：

| test 轨迹 ID | 持续时间 (s) | RMSE (m) | 最大误差 (m) |
| --- | ---: | ---: | ---: |
| 256 | 7.42 | 0.3424 | 1.2835 |
| 265 | 7.56 | 0.4777 | 2.5781 |
| 272 | 7.00 | 0.2252 | 0.7392 |
| 275 | 6.92 | 0.3430 | 2.0361 |

RMSE 只统计实际飞行时段，因此必须结合完成率看；每种类型的均值也包含提前结束的回合。测试后没有再针对这些测试轨迹调整策略。训练和测试来自同一生成器的不同轨迹，未验证外部轨迹库、强风或实机泛化。

## 检查与使用

- `python -m pytest -q`：**39 passed**，第三方 Matplotlib/Pyparsing 的 14 条弃用 warning。测试自行生成临时数据。
- 检查涵盖解析导数、速度/加速度/高度边界、集合隔离、难度缩放、完整枚举无重复、验证固定/分层、混控轴响应、相对观测平移不变性、奖励分项/失败惩罚、偏航惩罚、V3 保存恢复及原有环境回归。
- V3 `check_env`、`compileall`、`git diff --check` 通过。格式 2 旧模型的实际无界面评估仍可运行，输出保存在 `runs/v3_compatibility/legacy_eval/`。
- 全部 256 条测试输出了 CSV、指标和 PNG。默认硬件 OpenGL 的 GUI 尝试在初始化时未返回；停止该次尝试后，使用 Mesa 软件渲染实际完成完整八字回放。GUI 与无界面 CSV 都为 500 × 9，逐项最大差异为 **0**。结果保存在 `runs/v3_stable/demo_figure8_gui/`。未关闭用户原有的旧模型 GUI 进程；本次启动的训练/评估进程均已退出。

```bash
conda activate flight-rl

# 先关闭原有旧模型窗口，再启动新的八字跟踪回放
LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa \
python main.py eval --model runs/v3_stable/best/best_model.zip \
  --family figure8 --difficulty 1 --episodes 1 --seed 2026 \
  --render --playback-speed 0.5 --hold-seconds 15

# 复现全部测试
python main.py eval --model runs/v3_stable/best/best_model.zip \
  --all-trajectories --seed 2026 --output-dir runs/v3_stable/test_recheck

# 继续训练（保留本次模型）
python main.py resume --resume runs/v3_stable/best/best_model.zip \
  --timesteps 1000000 --learning-rate 0.00003 --num-envs 4 --run-dir runs/my_finetune
```

模型、数据、日志保留在本地；`runs/` 和生成数据目录被 `.gitignore` 排除。未提交或覆盖原始模型。

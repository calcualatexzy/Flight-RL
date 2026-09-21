# 跟踪诊断与提速升级（2026-09-21）

## 实际发现

上一轮 V4 精度微调新增 3,002,368 步，最佳点出现在新增约 700,000 步；后期退化。固定 96 回合上，旧最佳纯 RL 模型 RMSE 为 0.1570 m、完成率 97.92%、严格精度成功率 16.67%。训练速度始终 0.9～1.1 倍，没有实际进入更高速度。

可确认的两处训练设计问题：

1. 升级用“高于当前训练上限的下一个验证档”考核。例如训练最高 1.1 倍，却要求在 1.15 倍上两种时间安排同时达到精度门槛。这个规则容易卡住。现改为验证**当前训练上限**，通过后再引入更快数据；若固定验证档里没有当前上限，单独增加探测，不改变选模所用的固定集合。
2. 后期验证连续变差时没有停止条件。现加入连续六次无改善提前停止、保留最佳模型，并检查同目录恢复时验证集合一致性，防止用退化的恢复模型覆盖历史最佳。

为了区分策略能力与物理极限，保持 V4 的同一组 96 条验证设置、随机初始状态、50 Hz 控制/200 Hz 物理、原推力/力矩和电机限制，运行控制基线：

| 控制方式 | 平均 RMSE | 完成率 | 严格精度成功率 |
| --- | ---: | ---: | ---: |
| V4 最佳纯 RL | 15.70 cm | 97.92% | 16.67% |
| 几何姿态/位置反馈，无姿态角速度前馈 | 10.71 cm | 100% | 64.58% |
| 加速度及姿态角速度前馈 + 几何反馈 | 3.21 cm | 100% | 95.83% |

原始对照记录：`runs/v5_speed_audit_20260921/controller_probe.json`。两种几何控制都使用参考加速度，第三行额外用参考 jerk 和状态反馈计算期望姿态角速度。对照表不是新 RL 训练收益，说明在同样物理约束下更准确跟踪是可行的；未声称单独某个 PPO 超参数导致了全部退化。

## 新结构：前馈控制 + RL 残差

`--profile residual` 是明确的混合控制结构，和原 `--profile tracking` 的纯 RL 直接控制区别开来。

- 基础控制器使用当前位置/速度误差、参考加速度、下一控制时刻的**计划参考加速度**，计算期望推力方向、姿态和角速度。不访问未来实测状态，不修改无人机实际位置。
- 位置增益 8、速度增益 5，姿态/角速度增益 100/20，包含惯量陀螺项补偿。质量/惯量依赖与原静态混控一致；仍是仿真模型参数，不代表实机验证。
- PPO 输出四维有界修正，乘以 `[0.10, 0.20, 0.20, 0.15]` 后叠加到基础总推力/三个力矩命令，再按原动作和电机界限执行。零残差严格对应控制基线。
- 观测从 59 维扩为 67 维，增加基础控制命令和上一残差；旧 59 维中的上一动作仍保存实际执行命令。使用新的策略，输出头初始化为零，不把旧纯 RL 动作误解为残差。
- 保持精度奖励，另加 `-0.01*sum(residual^2)`，限制无用修正。GAE lambda 从旧纯 RL 的 0.95 提高到 0.98，使奖励估计覆盖更长的纠偏时间。
- 新模型格式 4；原格式 2/3 仍能加载。模型环境配置明确记录 `profile=residual`。测试、报告必须区分控制基线效果与学习残差的额外收益。

## 正式训练

从零残差初始化，计划最多 2,000,000 步，8 个 CPU 环境。初始训练速度 1.0～1.3 倍；两种时间安排在当前上限连续两次达到完成率 ≥95%、严格精度成功率 ≥75%、平均 RMSE ≤0.10 m，才增加 0.1，最高尝试 1.5 倍。

固定验证：八类轨迹各两条，原时间安排/巡航安排 × 1.0/1.3/1.5 倍，共 96 回合。新增的当前上限探测与固定验证分开，训练和验证使用不同数据集合，测试集不用于选模。

“1.5 倍”是请求速度。参考生成仍检查速度、加速度、jerk 和理想推力余量；必要时减速，逐条记录 `requested_speed_scale`、`effective_speed_scale` 和 `cruise_fallback`。没有通过缩小轨迹或视频加速伪装提速。

```bash
conda activate flight-rl
python scripts/start_speed_training.py --run-dir runs/v5_residual_speed_20260921
tail -f runs/v5_residual_speed_20260921/training.log
```

- `initial_model.zip` / `baseline.json`：零残差基础控制的起点，始终保留。
- `best/best_model.zip` / `best/validation.json`：固定验证所选最佳，可能仍是初始基线，必须检查步数和指标。
- `checkpoints/`：每新增 50,000 步保存，可续训。
- `validation/`：每 100,000 步验证，含实际速度和当前上限探测。
- `curriculum.jsonl`：实际升速记录；`early_stop.json`：若有，记录停止原因。
- `launch.json` / `source_snapshot/`：进程、命令和源码快照。

平稳停止：`kill -TERM "$(cat runs/v5_residual_speed_20260921/training.pid)"`。

```bash
# 恢复新结构；不要传 --profile tracking 改变动作语义。
python main.py resume --resume runs/v5_residual_speed_20260921/checkpoints/ppo_200000_steps.zip \
  --run-dir runs/residual_speed_resumed --timesteps 1000000 --num-envs 8 \
  --learning-rate 0.00005 --eval-freq 100000 --eval-episodes 16 --eval-patience 6 \
  --eval-speed-scales 1 1.3 1.5 --eval-time-profiles quintic cruise --curriculum-max-speed 1.5

# 固定速度，在 PyBullet 中看一条八字；不同环境配置不直接比较随机速度回合。
LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa \
python main.py eval --model runs/v5_residual_speed_20260921/best/best_model.zip \
  --family figure8 --time-profile cruise --speed-min 1.3 --speed-max 1.3 \
  --episodes 1 --seed 2026 --render --playback-speed 1 --hold-seconds 10
```

## 本次实际运行结果

- 67 项自动检查通过。残差命令实际影响电机与飞行状态，零残差与显式基础控制逐步一致；checkpoint 恢复、当前速度考核、早停和最佳模型保护已验证。
- 正式运行：`runs/v5_residual_speed_20260921/`。200,000 步升至 1.4 倍，400,000 步升至 1.5 倍，1,000,000 步因连续六次验证无改善提前停止，当前已无该训练进程。
- 新固定验证包含 1.0/1.3/1.5 倍，比上面的 V4 对照套件更快，不能直接混用两张表的整体平均。零残差基线平均 RMSE 3.429 cm、96/96 完成、90/96 严格达标；末次 RL 策略 RMSE 3.540 cm、达标数仍为 90/96。验证最佳仍是 0 步初始化基线。RL 参数确实更新，但本轮没有额外验证收益，不能把控制结构带来的改善归功于 RL 学习。
- 独立测试仅用于报告，没有用于选模型或课程升级：冻结的零残差基线在全部 256 条 test 飞行轨迹上请求巡航 1.5 倍，实际平均 1.429 倍，254/256 完成，227/256 严格达标（88.67%），平均 RMSE 5.675 cm。提前终止回合的误差只统计存活时段，完成率必须同时报告。
- 失败保留：test ID 257（Lissajous）与 ID 184（Slalom）碰撞终止。其余未严格达标回合也保留在 `runs/v5_speed_audit_20260921/baseline_test_cruise_150/metrics.json`，不声称全部困难轨迹解决。
- 单条 1.5 倍八字示例：`runs/v5_speed_audit_20260921/figure8_150/tracking.png`。实际 1.497 倍，6.68 秒飞完，RMSE 2.602 cm、最大误差 6.881 cm；该示例同样来自零残差基线。
- 总核查记录：`runs/v5_speed_audit_20260921/verification.json`。当前建议使用 `best/best_model.zip`，完整训练末态另存为 `final_model.zip`。

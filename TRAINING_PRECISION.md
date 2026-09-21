# 精确跟踪与速度课程（2026-09-21）

本轮从 `runs/v3_stable/best/best_model.zip` 的 3,100,000 步策略微调。目标是提高贴轨精度和真实飞行速度；效果需由后续固定验证集判断，不能把启动训练当作已达到目标。

## 已实现

- `--reward-profile precision`：位置奖励为 `0.25/(1+(ep/0.6)^2) + 0.35/(1+(ep/0.25)^2) + 0.8/(1+(ep/0.1)^2)`，兼顾大误差恢复与 5～10 cm 精度。速度奖励改为 `0.5/(1+(ev/0.4)^2)`；动作变化惩罚从 0.02 降到 0.005，减少对快速纠偏的抑制。其余姿态、航向、偏航角速度、失败惩罚保留。
- 保持 59 维观测、推力/力矩动作和物理仿真接口，直接恢复原策略及优化器。学习率 3e-5、熵系数 0.0002、target KL 0.015，采用较小更新微调。旧 checkpoint 未显式切换时保留原奖励及原速参考轨迹。
- `--time-profile cruise`：保持路径形状，采用弧长和曲率分配时间，中段较均匀、急弯减速，首尾各约 15% 时间平滑启停。`mixed` 每回合按 40% 原时间安排、60% 巡航安排采样。现有 2,304 条训练轨迹通过时间安排与连续速度采样增广，没有把验证/测试轨迹加入训练。
- 参考速度乘数会同步改变速度、加速度和回合长度；未来观测仍预览相同的真实时间。检查速度 ≤5 m/s、加速度 ≤8 m/s²、jerk ≤40 m/s³、理想合推力 ≤1.5 倍重力及竖直推力余量。不满足时延长时间，不缩小空间轨迹。检查是平移动力学参考筛查，未证明含阻力、转矩和纠偏需求的完整电机可行性。
- 极紧的曲线若巡航方案迫使全程过慢，会与原时间安排的可行提速方案比较，保留耗时较短的方案；元数据记录 `effective_time_profile=quintic_fallback` 和 `cruise_fallback`。`requested_speed_scale` 与 `effective_speed_scale` 分开记录，不能把被限速的请求当成实际 1.3 倍飞行。

## 验证与课程

每次固定 16 条验证路径（八类各两条、无重复），初始状态也固定。分别运行原时间安排/巡航安排 × 1.0/1.15/1.3 倍，共 96 回合。训练开始前保存同样规则下的基线，以后每新增 100,000 步复测。测试集不参与选模型。

**精度成功标准：完整飞完、位置 RMSE ≤0.10 m、最大误差 ≤0.30 m。** 旧的 0.35 m / 1 m 指标单独标为 `legacy_tracking_success`。同时记录 P95、速度误差、高速四分位位置 RMSE、实际/参考平均速度、限速与时间安排回退情况。提前失败的 RMSE 只覆盖存活时间，必须结合完成率。

保存最佳模型按以下顺序比较同一固定验证套件：完成回合数、精度成功回合数、平均 RMSE。这避免失败回合因为飞行时间短、平均误差暂时小而被优选；初始模型也参与候选。

训练初始速度范围 0.9～1.1 倍。连续两次验证在不低于当前训练速度上限的探测速度上，两种时间安排都满足完成率 ≥95%、精度成功率 ≥50%、平均 RMSE ≤0.13 m，才把训练上限提高 0.1，最高 1.3。阶段升级门槛与最终 0.10 m 精度标准不同；不达标就保留当前速度训练，不承诺自动升满 1.3。

## 启动、观察与恢复

```bash
conda activate flight-rl
python scripts/start_precision_training.py --run-dir runs/v4_precision_20260921
# 默认新增 3,000,000 步，8 个 CPU 环境；后台运行并写 training.pid。
tail -f runs/v4_precision_20260921/training.log
```

每新增 50,000 步保存 checkpoint；验证最佳为 `best/best_model.zip`，正常完成/收到 SIGTERM 后保存 `final_model.zip`。`launch.json` 保存命令、PID、起始模型哈希；`source_snapshot/` 保留启动时源码。运行目录拒绝覆盖已有文件。

```bash
# 平稳停止并保存（只给训练主进程发信号）
kill -TERM "$(cat runs/v4_precision_20260921/training.pid)"

# 从某次 checkpoint 恢复到新的运行目录；保留保存的奖励和速度设置
python main.py resume --resume runs/v4_precision_20260921/checkpoints/ppo_3200000_steps.zip \
  --timesteps 1000000 --num-envs 8 --learning-rate 0.00003 \
  --eval-freq 100000 --eval-episodes 16 --eval-speed-scales 1 1.15 1.3 \
  --eval-time-profiles quintic cruise --curriculum-max-speed 1.3 \
  --run-dir runs/precision_resumed

# 评估时显式固定速度与时间安排，避免使用训练的随机速度范围
python main.py eval --model runs/v4_precision_20260921/best/best_model.zip \
  --time-profile cruise --speed-min 1.15 --speed-max 1.15 --episodes 16 \
  --output-dir runs/v4_precision_20260921/eval_cruise_115
```

初始权重和历史结果保留在 `runs/v3_stable/`。V3 报告的 98.4% 使用旧阈值；新日志的成功率使用严格阈值，不能直接比较。

## 本次启动核查

- 正式后台运行目录：`runs/v4_precision_20260921/`，启动 PID 记录于 `training.pid`，本次为 2654386。
- 自动检查：60 passed；短训练已验证恢复优化器、执行更新、周期保存及初始/周期验证。评估 CLI 也已实际运行。
- 启动核查时已新增 500,000 步；读取检查点成功，参数均有限且相对起始策略确实发生变化。训练仍继续，最新结果看运行目录。
- 逐条检查了 2,048 条训练飞行路径与 256 条验证飞行路径的两种时间安排（不含悬停），请求 1.3 倍时均保持几何路径、满足配置的参考动力学筛查。巡航安排的平均实际速度倍率约 1.273，最慢约 1.126；被限速/回退的路径已逐条记录。
- 复核报告：`runs/v4_precision_20260921_preflight/verification.json`；逐条时间安排检查：同目录 `retiming_audit.json`。这些是程序与参考轨迹检查，不等于新策略已经达到跟踪目标。

# 旧版任务验证记录（2026-09-18）

> 本文保留 V2 / legacy 实验，不能用于判断新 V3 模型。多样化轨迹、奖励改进和重新训练结果见 [TRAINING_V3.md](TRAINING_V3.md)。

## 结论

环境配置、PPO 更新、模型保存、恢复训练、无界面评估及中断保存均已实际跑通。悬停模型在 10 个固定种子回合中全部飞满 5 秒，平均每回合位置 RMSE 为 **0.0507 m**。

**复杂轨迹尚未收敛。** 原始 5 秒任务直接训练约 100 万步仍全部提前结束；悬停预训练、10 秒轨迹迁移和低学习率微调后，最终模型在同一组种子上完成 **4/10** 个完整回合，平均每回合位置 RMSE 为 **2.0329 m**。10 秒任务比原始 5 秒任务更慢，不能将两者直接视为同难度性能比较。当前交付是可复现、可继续训练的实现，以及稳定悬停模型，不是已验证的高精度轨迹跟踪控制器。

## 环境与检查

- conda 环境：`flight-rl`。
- Python 3.11.16，PyTorch 2.7.1+cpu，SB3 2.7.0，Gymnasium 1.2.0，PyBullet 3.2.7，NumPy 2.2.6。
- 使用 CPU、4 个 PyBullet 子进程、每进程 1 个 PyTorch 线程。本机虽有 RTX 5090，本次未使用 CUDA。
- `python -m pip check`：无依赖冲突；`requirements-cpu.txt` 安装解析验证通过，完整版本在 `requirements-lock.txt`。
- `python -m pytest -q`：**18 passed**。14 条 warning 来自第三方 Matplotlib/Pyparsing 弃用接口，不是测试失败。
- Gymnasium 和 SB3 环境检查通过；Gymnasium 对无界 Box 空间有提示，观测空间允许飞行边界之外的终止状态是有意设计。
- `compileall` 和 `git diff --check` 通过。

回归覆盖动作确实改变升降/姿态、种子复现、软/硬重置、历史清空、无历史模式、时间截断/碰撞终止、多 PyBullet 客户端隔离、惯量合法性、阻力耗散、ZIP 加载及任意工作目录、未来目标预览、实际图像录制、模型更新/保存/恢复和旧模型拒绝。

## 实际运行

1. `runs/trajectory_smoke`：4 进程训练 16,384 步，生成 8,192/16,384 步 checkpoint 和周期评估。
2. `runs/resume_smoke`：将上述模型改为 2 进程恢复，累计步数从 16,384 增至 20,480，并完成 3 回合评估、CSV 和 PNG 导出。
3. `runs/trajectory_baseline`：原始 5 秒轨迹，实际 1,001,472 步，约 144 秒、6,900 步/秒。
4. `runs/hover_baseline`：悬停，实际 1,001,472 步，约 146 秒。
5. `runs/trajectory_curriculum`：从悬停最终模型恢复到 10 秒轨迹，再训练 1,001,472 步；累计 2,002,944 步。原学习率 3e-4 下观察到后期退化，因此保留周期最佳与最终模型。
6. `runs/trajectory_refined`：从上一阶段 1,401,472 步的周期最佳恢复，以 3e-5 再训练 1,001,472 步；累计 2,402,944 步，约 147 秒。
7. `runs/signal_smoke`：向训练主进程发送 SIGTERM，正常退出并恢复第 366 步保存的模型。
8. `runs/interrupt_smoke`：向包含两个工作进程的独立进程组发送 SIGINT，模拟终端 Ctrl+C，正常退出并恢复第 480 步保存的模型。
9. `runs/cli_regression_resume`：最终 CLI 验证从 1 个环境的 128 步恢复到 2 个环境的 256 步，显式学习率 1e-5 及其 schedule 均保存正确。

中断测试的具体步数受信号到达时机影响。保存的是当时最新策略/优化器，不包含未完成的 rollout 或 PyBullet 物理状态。根目录和 `results/Jack/` 中的原模型文件未修改。

## 固定种子评估

以下每项均使用种子 2026～2035、确定性策略、启用域随机化、10 个回合。`mean_position_rmse` 是各回合欧氏位置误差 RMSE 的平均值，**仅统计实际存活时段**；完整回合率只表示飞满时长，不表示精度达标。不能仅凭提前坠落模型的较低 RMSE 判定它更好。

| 模型/任务 | 累计训练步数 | 平均回报 | 位置 RMSE 均值 (m) | 完整回合率 | 原始结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 悬停，最终模型 | 1,001,472 | 996.59 | 0.0507 | 100% | 指标（本地文件 `runs/hover_baseline/eval/metrics.json`） |
| 5 秒轨迹直接训练，最终模型 | 1,001,472 | 163.06 | 4.1529 | 0% | 指标（本地文件 `runs/trajectory_baseline/eval/metrics.json`） |
| 5 秒轨迹直接训练，周期最佳 | 1,000,000 | 163.23 | 4.0931 | 0% | 指标（本地文件 `runs/trajectory_baseline/eval_best/metrics.json`） |
| 10 秒轨迹迁移，周期最佳 | 1,401,472 | 343.77 | 1.1894 | 20% | 指标（本地文件 `runs/trajectory_curriculum/eval_best/metrics.json`） |
| 10 秒轨迹迁移，最终模型 | 2,002,944 | 168.66 | 0.3441 | 0% | 指标（本地文件 `runs/trajectory_curriculum/eval/metrics.json`） |
| 10 秒轨迹低学习率微调，周期最佳 | 1,801,472 | 276.96 | 0.5981 | 10% | 指标（本地文件 `runs/trajectory_refined/eval_best/metrics.json`） |
| 10 秒轨迹低学习率微调，最终模型 | 2,402,944 | 321.51 | 2.0329 | 40% | 指标（本地文件 `runs/trajectory_refined/eval/metrics.json`） |

这些评估与训练使用同一轨迹库的不同随机种子，尚未建立独立轨迹测试集。周期最佳模型由当次少量回合平均奖励选择，未必在另一组 10 回合种子上优于最终模型；因此两者均予保留并单独评估。

轨迹库共 400 条，虽然文件名包含 `vel1.5`，实际各轨迹峰值速度范围约 1.45～6.67 m/s，峰值加速度约 2.32～14.00 m/s²。时间拉长到 10 秒后，速度减半、加速度变为四分之一。

训练回报及完整回合率曲线（本地文件 `runs/validation/learning_curves.png`） · 悬停轨迹图（本地文件 `runs/hover_baseline/eval/tracking.png`） · 轨迹微调评估图（本地文件 `runs/trajectory_refined/eval/tracking.png`）

## 继续使用

```bash
conda activate flight-rl

# 查看已学会的悬停
python main.py eval --model runs/hover_baseline/final_model.zip --episodes 10 --seed 2026

# 从轨迹微调模型继续训练；自动恢复 10 秒轨迹环境配置
python main.py resume --resume runs/trajectory_refined/final_model.zip \
  --timesteps 1000000 --learning-rate 0.00003 --num-envs 4

# 评估轨迹模型
python main.py eval --model runs/trajectory_refined/final_model.zip --episodes 10 --seed 2026
```

模型、日志和评估产物保留在本机 `runs/`，该目录被 `.gitignore` 排除，避免把大体积训练结果混入源代码。未保留后台训练进程。更完整的使用说明见 [README.md](README.md)。


## 可视化补充验证

- PyBullet 窗口已实际运行：完整参考轨迹、实时飞行尾迹、起终点、目标点、误差连线与 HUD；支持 F 全景、T 俯视和空格暂停。
- 原有 18 项回归测试通过；新增 1 项测试确认无碰撞的视觉标记不会改变动力学，并在 reset 时正确清理。
- 模型 `runs/trajectory_refined/final_model.zip`、种子 2026 的 GUI 与无界面飞行 CSV 共 1,729 步，逐项比较在绝对误差 1e-10 内一致。
- 提前结束的实际飞行仍导出完整 10 秒、2,001 点参考轨迹；多回合评估的图表和 CSV 导出已验证。
- 新的轨迹对比图（本地文件 `runs/visualization_review/tracking.png`）展示三维轨迹、俯视轨迹和位置误差。新增绘制功能无需重新训练模型。

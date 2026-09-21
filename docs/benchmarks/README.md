# 公开测试指标

`residual_baseline_cruise_150.json` 保存冻结的零残差几何控制基线在默认独立 test 集中的全部 256 条飞行结果。未删除碰撞、提前终止或不满足精度门槛的回合。

- 数据：生成种子 2026，每类 32 条 test 路径，八类飞行路径各评估一次。
- 初始状态：`2026 + episode_index`，开启域随机化。
- 控制/物理频率：50 / 200 Hz；完整空间难度 1。
- 时间安排：请求 cruise 1.5×，按物理约束记录实际倍率和回退。
- 严格成功：完整飞完、位置 RMSE ≤ 0.10 m、最大误差 ≤ 0.30 m。
- 结果：254/256 完成，227/256 严格达标，各回合 RMSE 均值 0.05674575 m。
- ID 257 和 184 碰撞终止，其误差只覆盖实际存活时段。

复现（仓库根目录，已安装环境并生成默认数据）：

```bash
python scripts/create_baseline.py
python main.py eval --model runs/residual_baseline/baseline.zip \
  --all-trajectories --seed 2026 --difficulty 1 \
  --time-profile cruise --speed-min 1.5 --speed-max 1.5 \
  --output-dir runs/baseline_test_all
```

基线创建脚本不执行训练。JSON 中的 `model_timesteps=0` 及控制方式明确表示没有学习残差贡献。评估命令默认使用 test 集；训练与选模使用独立集合。原始逐步 CSV 和轨迹图由上述命令本地生成，不包含在此目录。

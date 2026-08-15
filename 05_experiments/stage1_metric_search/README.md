# 第 1 阶段输出

- 每轮结果：`round_<NNN>/`（与 `04_memory/experiment_logs/round_<NNN>.json` 互相引用）
- 阶段性快照：每 5 轮一次 `best_snapshot.json`
- 最终最优组合：`final_combo.json`（冻结，进入第 2 阶段不可修改）
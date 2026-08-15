# SKILL · log_round

> 把一轮实验的关键信息完整写入 `04_memory/experiment_logs/round_<NNN>.json`。

## 必填字段

```python
{
    "round": int,
    "stage": "stage1" | "stage2",
    "timestamp": str,
    "combo": MetricCombo,
    "consistency": {"kappa": ..., "accuracy": ..., "f1_macro": ...},
    "diff_vs_prev": ...,
    "failures": [{"poem_id": ..., "predicted": ..., "gold": ..., "note": ...}, ...],
    "check_agent": {"pre": "PASS|FAIL", "post": "PASS|FAIL", "invalid_markers": [...]},
    "params": {...},          # 阶段相关
    "artifacts": ["..."]      # 指向 05_experiments/ 下的文件
}
```

## 不可省略字段

- `check_agent` 两项：缺失则该轮 INVALID
- `failures`：显著不一致样本的逐条记录
- `artifacts`：所有写出的文件相对路径

## 命名

- 文件名固定 `round_<NNN>.json`，NNN 自增，不允许跳跃
- 失败的轮次**也**按 NNN 命名，状态字段标 `INVALID`
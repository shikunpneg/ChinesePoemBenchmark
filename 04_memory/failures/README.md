# 失败样本与失败模式

## 失败样本

- `round_<NNN>.jsonl`：该轮中指标判断与人类判断显著不一致的样本
- 字段：`{poem_id, predicted, gold, note}`
- 来自 `SKILL_log_round.md` 的 `failures` 字段

## 失败模式

- `failure_mode_<tag>.md`：当连续多轮出现负结果 / 异常 / 反例聚集时触发
- 内容：
  - 触发条件
  - 受影响的轮次范围
  - 假设来源（特征空间 / 组合方式 / 数据分布 / 问题定义）
  - 修正方向

## 反例（counter examples）

- 与失败样本同等对待，单独记录「代表性反例」便于人工复核
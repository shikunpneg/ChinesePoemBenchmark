# 实验日志

> 每轮实验一条 `round_<NNN>.json`，结构见 `03_agent_harness/skills/SKILL_log_round.md`。
> INVALID 轮次**不删除**，仅标记状态。

## 索引

- `index.json`：轮次索引，按 NNN 升序
- `round_<NNN>.json`：单轮实验记录
- `audit_round_<NNN>.json`：每 3 轮一次的完整审计

## 命名规则

- NNN 自增，不允许跳跃
- 即便是失败的轮次也按 NNN 写入
# Check-Agent · 审计周期

## pre_round（每轮实验前必做）

1. 读取上一轮 `experiment_logs/round_<NNN-1>.json` 的 `check_agent.post`
2. 校验本轮拟用的：
   - 数据划分（必须为 train / val）
   - 特征 ID（必须在 `feature_catalog.md` 内）
   - 模型（必须为 `SKILL_ai_poem_generate.md` 允许的清单内）
3. 写入 `experiment_logs/round_<NNN>.json.check_agent.pre`

## post_round（每轮实验后必做）

1. 校验本轮写出的所有文件路径**必须**在以下范围内：
   - `04_memory/`
   - `05_experiments/`
   - `06_artifacts/`
   - `07_reproducibility/`（仅在人工授权时）
2. 校验特征 ID 没有超出 `feature_catalog.md`
3. 校验 LLM 调用日志（仅允许 `SKILL_ai_poem_generate.md` 中的模型）
4. 写入 `experiment_logs/round_<NNN>.json.check_agent.post`

## full_audit（每 3 轮一次）

1. 重新校验最近 3 轮的 pre + post
2. 校验 `04_memory/rules_memory/` 中是否有新增规则被 Agent 触发
3. 校验总 Token 消耗是否逼近预算上限
4. 写入 `04_memory/experiment_logs/audit_round_<NNN>.json`

## 失败处理

任意检查不通过 → 立刻：
1. 把当前轮标记为 INVALID（详见 `invalid_marker.md`）
2. 把「违规行为—触发条件—修正规则」写入 `04_memory/rules_memory/`
3. 通知项目负责人（人工）；不进入下一轮优化
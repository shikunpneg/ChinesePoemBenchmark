# 第 2 阶段输出

每轮目录 `round_<NNN>/` 含：

- `ai_poems.jsonl`：本轮生成的 AI 诗歌
- `similarity_report.json`：与目标人类诗歌的相似度
- `consistency.json`：固定指标下的一致性变化
- `failures.jsonl`：显著不一致样本
- `params.json`：生成参数 + 选用的目标人类诗歌 ID
# 永久性记忆 · 规则库（rules memory）

> **违规行为—触发条件—修正规则**的永久记录。仅由 Check-Agent 与项目负责人写入；Agent 不得修改或删除本目录下的条目。

## 格式

每条规则一条 markdown，文件命名 `YYYY-MM-DD_<short-tag>.md`：

```markdown
## YYYY-MM-DD · <违规类型>

- 触发条件：<越界发生时的具体动作>
- 修正规则：<之后不允许的动作>
- 关联轮次：round_<NNN>
- 处理人：Check-Agent / 项目负责人
```

## 当前规则（草案占位）

> 实际规则将在首个 INVALID 触发时由 Check-Agent 自动写入。
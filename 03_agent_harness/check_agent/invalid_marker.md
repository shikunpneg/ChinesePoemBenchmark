# Check-Agent · INVALID 标记规范

## 字段

每轮实验记录 `experiment_logs/round_<NNN>.json` 必须含：

```json
{
    "status": "VALID" | "INVALID",
    "invalid_reasons": [
        {"type": "DATA_OOB" | "RULE_BYPASS" | "PROTOCOL_CHANGE" | "WRITE_OOB" | "UNAUTH_API",
         "detail": "...",
         "rule_ref": "rules.md#..."
        }
    ],
    "audit": {
        "pre": "PASS|FAIL",
        "post": "PASS|FAIL"
    }
}
```

## 后果

- 状态为 INVALID 的轮次**不得**进入后续优化和评价
- 该轮的 `consistency` 与 `failures` 仅作研究证据保留
- `04_memory/experiment_logs/round_<NNN>.json` 文件**保留不删**

## 永久性记忆追加

触发 INVALID 时，必须在 `04_memory/rules_memory/` 追加一条：

```markdown
## YYYY-MM-DD · <违规类型>
- 触发条件：...
- 修正规则：...
- 关联轮次：round_<NNN>
```
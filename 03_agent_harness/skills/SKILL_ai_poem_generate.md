# SKILL · ai_poem_generate（第 2 阶段）

> 使用声明范围内的 LLM API 生成 AI 诗歌；不得引入未声明模型。

## 接口

```python
def generate_ai_poems(
    target_human_poem: str,
    style: Literal["tang", "song", "modern", ...],
    n: int,
    model: Literal["deepseek-v4-flash"],   # 仅此一个
    temperature: float,                    # [0.0, 1.5]
    top_p: float,                          # [0.0, 1.0]
    seed: int | None,
) -> list[str]:
    ...
```

## 资源约束

- 总 Token 消耗必须 < `02_environment/record_budget.md` 中声明的总预算上限
- 单轮实验规模需记录到 `experiment_logs/round_<NNN>.json`

## 禁止

- 使用除 `model` 字段允许之外的任何模型
- 引入未声明的 API / 数据
- 在生成内容中混入外部参考文本（避免训练数据泄漏）
# SKILL · similarity_compute（第 2 阶段）

> 计算 AI 生成诗歌与人类诗歌之间的相似度。

## 接口

```python
def similarity(
    ai_poems: list[str],
    human_poems: list[str],
    method: Literal["bleu", "rouge", "embedding", "llm_judge"],
) -> SimilarityReport:
    ...

@dataclass
class SimilarityReport:
    method: str
    per_pair: list[float]
    aggregate: float
```

## 与第 1 阶段指标的区分

- 本 SKILL 输出的「相似度」仅用于**驱动生成参数调整**
- **不得**把相似度当作人类一致性的代理
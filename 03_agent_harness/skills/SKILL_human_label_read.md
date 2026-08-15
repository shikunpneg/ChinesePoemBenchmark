# SKILL · human_label_read

> 读取 `human_labels` 的训练 / 验证划分；不得修改、不得使用测试划分。

## 接口

```python
def read_human_labels(split: Literal["train", "val"]) -> list[HumanLabel]:
    ...

@dataclass
class HumanLabel:
    poem_id: str
    label: int          # 0 / 1
    rater_id: str
    confidence: float | None
    note: str | None
```

## 调用前置

- 必须在 `SKILL_dataset_read.md` 之后调用
- 不得通过任何途径访问 `split="test"` 划分
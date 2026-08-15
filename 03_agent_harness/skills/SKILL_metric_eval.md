# SKILL · metric_eval

> 输入：一组指标组合；训练/验证集；人工标签。
> 输出：与人类判断的一致性（Kappa / Accuracy / 各项子指标）。

## 接口（草案，下一轮实现）

```python
def eval_metric_combo(
    combo: MetricCombo,         # 来自 feature_catalog
    split: Literal["train", "val"],
    human_labels: list[int],    # 0=非诗歌 1=诗歌
    return_per_sample: bool = False,
) -> EvalResult:
    ...
```

## 返回值

```python
@dataclass
class EvalResult:
    kappa: float
    accuracy: float
    f1_macro: float
    per_sample: list[SampleResult] | None
```

## 调用前置

- 必须先调用 `SKILL_dataset_read.md` 拿到数据划分
- 必须先调用 `SKILL_human_label_read.md` 拿到标签
- 检查 `check_agent/audit_cycle.md#pre_round` 通过

## 禁止

- 在 SKILL 内修改数据划分 / 标签 / 指标库
- 跳过 Check-Agent
- 隐式使用测试集
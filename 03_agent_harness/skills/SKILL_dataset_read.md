# SKILL · dataset_read

> 只读地从 `data_registry` 加载数据集；不得复制到本地写入路径以外的位置。

## 接口

```python
def read_split(name: Literal["human_poems", "human_labels", "annotations", "ai_poems", "hard_judge"],
               split: Literal["train", "val", "test"]) -> Dataset:
    ...
```

## 读权限矩阵

| name          | train | val | test                |
| ------------- | ----- | --- | ------------------- |
| human_poems   | ✅    | ✅  | ❌（仅第 2 阶段可选参考） |
| human_labels  | ✅    | ✅  | ❌                     |
| annotations   | ✅    | ✅  | ❌                     |
| ai_poems      | ✅    | ✅  | ❌                     |
| hard_judge    | ✅    | ✅  | ❌                     |

## 禁止

- 在测试集上做指标组合搜索
- 修改 `data_registry/` 中任何文件
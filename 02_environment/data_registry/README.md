# 数据集只读引用清单（data registry）

> 本目录是**只读引用清单**，不是数据本身。所有路径下的数据**不允许修改**。

## 引用方式

```python
from pathlib import Path
DATA = {
    "human_poems":   Path(r"E:\生成诗歌\诗歌集"),
    "human_labels":  Path(r"E:\生成诗歌\poetry-judge-train"),
    "annotations":   Path(r"E:\生成诗歌\eval-annotation"),
    "ai_poems":      Path(r"E:\生成诗歌\dataset-build"),
    "hard_judge":    Path(r"E:\生成诗歌\ChineseHardJudgePoem"),
}
```

## 引用清单（草案；正式实验前需逐一核对实际文件结构）

| 名称          | 路径                                            | 用途                                | 写权限 |
| ------------- | ----------------------------------------------- | ----------------------------------- | ------ |
| human_poems   | `E:\生成诗歌\诗歌集\`                           | 中文诗歌原文（古典 / 现当代）        | 只读   |
| human_labels  | `E:\生成诗歌\poetry-judge-train\`               | 人类「诗歌/非诗歌」二分类标注         | 只读   |
| annotations   | `E:\生成诗歌\eval-annotation\`                  | 人工评价协议的细粒度标注             | 只读   |
| ai_poems      | `E:\生成诗歌\dataset-build\`                    | 已生成的 AI 诗歌（第二阶段起始池）   | 只读   |
| hard_judge    | `E:\生成诗歌\ChineseHardJudgePoem\`             | 困难样本候选集                       | 只读   |

## 隔离规则

- 在本项目中**禁止**在以上路径下创建 / 修改 / 删除任何文件。
- 需要写入的中间产物统一落到 `05_experiments/` 或 `06_artifacts/`。
- 越界行为由 `03_agent_harness/check_agent/` 标记 INVALID。
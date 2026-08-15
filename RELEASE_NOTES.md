# ChinesePoemBenchmark v1.0 Release Notes

> 中文诗歌「诗歌性」自动评测指标 — 首个完整发布版本

## 概览

这是 `ChinesePoemBenchmark` 的第一个稳定版本（v1.0），包含：

- **26 维结构与语言学特征库**（含可解释性）
- **指标迭代闭环协议**（v1 → v2 → v3 → v4b → v5 → v6）
- **4 个训练好的指标版本**（v1 / v2 / v4b / v6a/b），推荐使用 **v4b**
- **完整实验数据**：13+ 轮、5 张图、3 张表
- **论文初稿 v1.1** + 指标迭代方案
- **生产标注分析**：首次基于真实生产 Neon PostgreSQL 数据

## 推荐基线

| 指标 | val Kappa | expert Kappa | AI 诗 Kappa | AI 假阳性 |
|---|---|---|---|---|
| **v4b**（推荐）| **0.974** | 0.940 | **0.431** | **0** |
| v2 | 0.940 | 0.940 | -0.012 | 31 |
| v6b | 0.948 | 0.920 | 0.459 | 0 |

**v4b 是最佳版本**——三个数据集都表现良好。

## 主要发现

1. **指标是结构性诗性指标**——不区分人类诗 vs AI 诗
2. **真实人类 IAA 仅 0.504**（远低于文献值 0.72）——每个标注者都有 ~30% 系统噪声
3. **AI 仿诗的失效边界**：31 条「夹带英文/乱码」的 AI 诗，v2 全部误判为诗
4. **迭代闭环验证**：加入 32 条 AI 垃圾反例后，fp=0，val Kappa 反而升到 0.974
5. **L2 ablation 局限**：单数据集的改善可能是其他数据集失效的信号——v6 在 Racter 上完全崩塌

## 使用方法

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("02_environment/baseline_metrics").resolve()))

from code.frozen_metric import build_and_freeze, FrozenMetric

fm = build_and_freeze(seed=42, val_ratio=0.2)  # trains v4b
pred = fm.apply("床前明月光，疑是地上霜。举头望明月，低头思故乡。")
print(f"prob={pred.prob_poem:.3f}  pred={pred.pred}")
```

## 关键文件

- `paper_draft.md` — 论文初稿 v1.1（18 KB）
- `metric_iteration_plan.md` — 指标组合迭代方案（9.6 KB）
- `docs/figures/` — 5 张 publication 图
- `docs/tables/` — 3 张表
- `02_environment/baseline_metrics/code/` — 完整实现

## 路线图

- ✅ v1.0: v4b 基线 + 完整实验 + 论文 + 开源

未来工作（**暂停迭代**）：
- ⚠️ v6 试错：drop struct+style 在 Racter 上崩塌，已停止
- ⏭️ LLM-as-judge 接入（需要 DeepSeek API key）
- ⏭️ 数据集开源（Hugging Face 镜像）

## 引用

```bibtex
@misc{chinesepoembenchmark2026,
  title={ChinesePoemBenchmark: A Structural Indicator for Chinese Poetry "Poeticity"},
  author={Project Team},
  year={2026},
  howpublished={\url{https://github.com/shikunpneg/ChinesePoemBenchmark}},
  note={v1.0 release}
}
```

## 致谢

本项目是「AI for Research」开放探索赛的参赛项目。

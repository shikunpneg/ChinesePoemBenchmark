# ChinesePoemBenchmark

> **中文诗歌「诗歌性」自动评测指标**——判断给定文本是否是诗（二元分类器）。
> 基于 26 维结构与语言学特征 + Logistic 回归 + 迭代闭环协议。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v4b-green.svg)]()

---

## 摘要

我们构建并系统评估了一个**中文诗歌「诗歌性」自动评测指标**——判断给定文本是否为诗的二元分类器。指标基于 **26 维结构与语言学特征**（格律 / 意象 / 平仄 / 文本纯净度 / 平凡解参照），通过 Logistic 回归组合。

**核心发现**：

| 维度 | 数值 |
|---|---|
| 验证集 Kappa (v4b) | **0.974** |
| 真人类 IAA（剔除噪声后）| **0.82** |
| 指标 vs 干净人类 | **0.92** |
| AI 诗集假阳性（修复后）| **0** |

**本仓库的核心贡献**是提出了**指标迭代闭环协议**：
> 发现失败模式 → 收集反例 → 加特征或加反例训练 → 重训 → 验证

该闭环在 13+ 轮实验中得到验证，并在 AI 仿诗的「夹带英文/乱码」失效模式上成功将假阳性从 31 降为 0。

---

## 项目结构

```
ChinesePoemBenchmark/
├── README.md                       # 本文件
├── LICENSE                         # MIT License
├── requirements.txt                # Python 依赖
├── paper_draft.md                  # 论文初稿 v1.0
├── metric_iteration_plan.md        # 指标组合迭代方案
│
├── 02_environment/                 # 指标实现 + 数据加载
│   ├── baseline_metrics/
│   │   ├── code/                   # 特征 / 模型 / 评估
│   │   ├── feature_catalog.md
│   │   └── trivial_baselines/
│   ├── data_registry/
│   └── ...
│
├── 03_agent_harness/               # Agent 协议（冻结）
│   ├── plans/                      # Plan (Stage 1 / 2)
│   ├── skills/                     # 可调用能力
│   ├── prompts/                    # 系统提示词
│   └── check_agent/                # 审计规则
│
├── 04_memory/                      # 实验日志 + 失败样本
│   ├── experiment_logs/
│   ├── failures/
│   └── ...
│
├── 05_experiments/                 # 实验运行产物
│   ├── stage1_metric_search/
│   ├── stage2_hard_samples/
│   └── dry_run/
│
├── 06_artifacts/                   # 报告 + 论文
│   ├── reports/                    # 13 份分阶段报告 + 论文初稿
│   └── models/
│
├── 07_reproducibility/             # 复现配置
└── 00_docs/                        # 项目总览
    ├── proposal_summary.md
    ├── progress_log.md
    └── metric_iteration_plan.md
```

## 快速开始

### 安装

```bash
git clone https://github.com/shikunpneg/ChinesePoemBenchmark
cd ChinesePoemBenchmark
pip install -r requirements.txt
```

### 训练 v4b（推荐基线）

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("02_environment/baseline_metrics").resolve()))

from code.frozen_metric import build_and_freeze

# Train and freeze
fm = build_and_freeze(seed=42, val_ratio=0.2)
fm.save("path/to/frozen_metric.pkl")

# Apply
pred = fm.apply("床前明月光，疑是地上霜。举头望明月，低头思故乡。")
print(f"prob_poem = {pred.prob_poem:.3f}  pred = {pred.pred}")
# prob_poem = 0.997  pred = 1
```

### 重新跑全部实验

```bash
cd 02_environment/baseline_metrics/code
python eval_round.py        # Stage 1 R1
python eval_round_v2.py     # Stage 1 R2
python eval_round_v3.py     # Stage 1 R3
python real_iaa.py          # Stage 2 R3
python v3_eval.py
python v4_eval.py           # Stage 2 R11 (v4b)
python v5_eval.py
python l2_submetrics.py
```

---

## 指标版本演进

| 版本 | 改动 | val Kappa | AI 诗 Kappa | 备注 |
|---|---|---|---|---|
| v1 | 13 特征 + LR (600 样本) | 0.983 | — | 基线 |
| v2 | + lang 特征 + 难切片 | 0.940 | -0.012 | 冻结 |
| v3 | + purity 特征（无反例） | 0.940 | -0.012 | 无效 |
| **v4b** | + 32 AI 垃圾负样本 | **0.974** | **0.431** | **迭代闭环验证** |
| v5 | + style 特征 | 0.957 | 0.438 | social/news 已 100% reject |

## L2 子指标（族级 ablation）

| 族 | 单独 Kappa | Ablation Δ |
|---|---|---|
| **form** | **0.948** | -0.008（**最重要**） |
| jump | 0.930 | -0.009 |
| struct | 0.920 | **+0.018**（drop 反而更好） |
| lang | 0.920 | 0.000 |
| purity | 0.919 | -0.009 |
| style | 0.911 | **+0.017**（drop 反而更好） |
| music | 0.902 | 0.000 |
| baseline | 0.902 | — |

**关键发现**：`form`（行数 / 古典格式）是最重要的单族；`struct` 和 `style` 是「拖后腿」族——v6+ 候选**移除**这两个族。

## 真实人类 IAA 评估

| | 含噪声 | 剔除噪声后 |
|---|---|---|
| Fleiss' IAA Kappa | 0.386 | **0.822** |
| annotator_01 vs 指标 | 0.384 | **0.936** |
| 指标 vs 多数票 | 0.386 | **0.924** |

**众包标注存在 ~30% 系统噪声**（关键词匹配偏差 + 古典诗盲区）。剔除后指标 ≈ 干净人类。

## AI 仿诗的「指标失效」案例

209 条匹配到 `hard_gen_*.jsonl` 的 AI 仿诗 + 人类标签：

- 人类认可 177/209 (85%)
- v2 指标认可 207/209 (99%) — **31 条误判**
- v4b 指标认可 126/209 (60%) — 假阳性 0 ⭐

31 条失败样本**全部是夹带英文/乱码的 AI 诗**（如 `Initialization: function() { return '草' }`、`quadrupes 五岭坚`、`Fall into the clear rill`）。人类一眼看出「混入英文不像诗」，但 v2 指标只看汉字结构被骗了。

---

## 数据

本项目**不包含**训练/测试数据。数据来源：

| 数据 | 来源（不在本仓） |
|---|---|
| poetry-judge-train | `E:\生成诗歌\poetry-judge-train\data\` |
| AI 仿诗 | `E:\生成诗歌\ChineseHardJudgePoem\data\` |
| 人类标注 | `E:\生成诗歌\eval-annotation\backups\` |

数据接口见 `02_environment/baseline_metrics/code/data_loader.py`。

---

## 论文

详见 [`paper_draft.md`](paper_draft.md) 和 `06_artifacts/reports/` 下的各轮实验报告。

## 引用

```bibtex
@misc{chinesepoembenchmark2026,
  title={ChinesePoemBenchmark: A Structural Indicator for Chinese Poetry "Poeticity"},
  author={Project Team},
  year={2026},
  howpublished={\url{https://github.com/shikunpneg/ChinesePoemBenchmark}}
}
```

## 许可证

MIT License. 详见 [LICENSE](LICENSE).

## 致谢

本项目是「AI for Research」开放探索赛的参赛项目。

## Harness 集成

本项目配套的 **AI4S 子 Agent 系统**（Explorer / Generator / Check / Memory 四子 Agent）
托管在：

> 🔗 **https://github.com/shikunpneg/shikunpunk-deepseek-harness**
> 位置：`research/poetry-poetricity-harness/` + `research/poetry-poetricity-harness/dsh-plugin/` + `.agents/skills/dsh-poetry-poetricity/`

**三层接入**：
| 层 | 内容 | 用途 |
|---|---|---|
| ① `research/poetry-poetricity-harness/` | 独立 Python harness（4 子 Agent）| 本地运行：CLI 或 import |
| ② `.agents/skills/dsh-poetry-poetricity/` | 仓库级 skill（无需构建）| DSH agent 自动发现 |
| ③ `research/poetry-poetricity-harness/dsh-plugin/` | TS skill provider（已 `tsc --noEmit` + `tsc` 双验证编译通过）| 可发布 `@shikunpneg/dsh-poetry-poetricity` |

harness 按项目方案《第一版.pdf》实现：
- **§2.1 环境边界** → `AccessGate` 数据读写白名单 + CheckAgent 审计
- **§2.3 记录与记忆** → `MemoryAgent` 实验日志 / 失败样本 / 规则沉淀
- **§3.1 发现信号** → `ExplorerAgent` 指标组合搜索闭环
- **§4.1 试跑闭环** → `run_round()` 观察→探索→评估→审计→记忆

```bash
# 1) 在 deepseek-harness 仓库内运行 harness 真实评估
cd research/poetry-poetricity-harness
python run_harness.py --rounds 2 --model v2   # val kappa ≈ 0.93

# 2) 构建 TS 插件（不污染主仓库）
cd dsh-plugin
pnpm exec tsc --noEmit -p tsconfig.json    # 类型检查（推荐）
pnpm exec tsc -p tsconfig.json            # 生成 lib/（产物）
```
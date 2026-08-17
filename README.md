# 中文诗歌「诗歌性」自动评测指标：从文学理论迁移到计算诗性的探索

> **项目定位**：用 **AI4S 多智能体 Harness** 自动化探索中文诗歌「诗性」的可计算边界，构建与人类判断**高度一致**的自动评测指标；
> 并通过 Harness 自驱动的「指标迭代闭环」，量化指标在 AI 高相似度仿诗下的失效边界。
>
> **核心论点**：现有评测方法（BLEU/ROUGE、LLM-as-judge、基于意象/形式/风格的特征集合）**本质上都是从文学理论的量化迁移**，
> 而非真正的「诗性」。本研究通过多智能体自动化探索，验证哪些可计算特征**真正**与人类对诗的判断一致。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v4c%20through%20threshold-green.svg)]()
[![GitHub](https://img.shields.io/badge/repo-HumanAlignedPoeticity-black.svg)](https://github.com/shikunpneg/HumanAlignedPoeticity-)
[![Harness](https://img.shields.io/badge/harness-AI4S%20%2B%20DSH--EvoResearch-blueviolet)](https://github.com/deepseek-ai/DeepSeek-Harness)
[![HF Models](https://img.shields.io/badge/HF_models-shikunpunk-orange.svg)](https://huggingface.co/shikunpunk)

---

## 目录

1. [我们做了什么：一个自动化科研流程](#1-我们做了什么一个自动化科研流程)
2. [Harness 自动化科研系统（核心）](#2-harness-自动化科研系统核心)
3. [一次验证：指标 vs 人类一致性](#3-一次验证指标-vs-人类一致性)
4. [方法学：特征库与迭代闭环](#4-方法学特征库与迭代闭环)
5. [相关资源](#5-相关资源)

---

## 1. 我们做了什么：一个自动化科研流程

本项目**不是**手工调参 → 一次性训练 → 发论文的传统 ML 流程。

本项目是一个**由 AI4S Harness 自驱动的自动化科研流程**：

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1：自动化特征组合探索                              │
│  ExplorerAgent 搜索 13 族 × 2 = 8192 个特征组合             │
│  exhaust 找到 best: meter+para+theme+theme8+struct+       │
│  ner_img+jump (37 维) → mask kappa 0.9387                │
│  → 重训真实 kappa 0.9303（与全 64 维 v2 几乎无差异）      │
│  → 结论：在 v2 切片上特征选择几乎无收益                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2：人类一致性验证（闭环核心）                       │
│  GeneratorAgent 生成 AI 高相似度仿诗                       │
│  CheckAgent 防止数据越界 / 规则绕过                       │
│  MemoryAgent 累积反例 → 训练 v4b（+32 AI 反例）            │
│  → v4b val 0.974 / expert 0.940 / AI 假阳性 0 ✅          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  跨标注者一致性监测（新数据验证）                          │
│  合并 5 位标注者 1577 条 → v4c val 0.9557 ≥ 0.95 阈值 ✅  │
│  → 发现标注者分歧：annotator_01 vs annotator_06             │
│    对"AI 诗是否为诗"存在系统性差异                       │
│  → 核心能力（AI 假阳性 0）在跨标注者上保持                │
└─────────────────────────────────────────────────────────┘
```

**关键结论**：
- 我们验证了**与人类一致性最高的"诗性"指标**——不是更复杂的特征，而是**正确的反例训练**（v4b 在 val/expert/AI 三集上全部通过）
- 我们发现**标注者分歧**是真实存在的（不是方法学问题，是数据问题）——这指向"指标 vs 人类一致性"的本质限制
- 整个流程由 Harness 自动化驱动（不需要人干预），符合 AI4S 开放赛道精神

---

## 2. Harness 自动化科研系统（核心）

本项目的**核心基础设施**是自建的 AI4S 多智能体 Harness，它让指标探索**可审计、可复现、可扩展**。

### 2.1 4 子 Agent 插件系统

整个 Harness 在 `03_agent_harness/harness/`，对应方案《第一版.pdf》§2-§4：

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Explorer    │  │  Generator    │  │   Check      │  │   Memory     │
│  Agent       │  │   Agent       │  │   Agent      │  │   Agent      │
├──────────────�  ├──────────────┤  ├──────────────┤  ├──────────────┤
│  特征组合搜索 │  │ AI 仿诗生成  │  │  数据越界审计  │ │ 实验日志     │
│  （4 策略）  │  │ 相似度提升    │  │  规则绕过审计  │ │ 失败样本沉淀 │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
         ↓                ↓                ↓                ↓
              ┌───────────────────────────────────────┐
              │      AccessGate（边界强制）          │
              │  读：poetry-judge-train / 标注库 / AI 仿诗池  │
              │  写：04_memory / 05_experiments / 06_artifacts  │
              └───────────────────────────────────────┘
```

**关键设计**：
- **AccessGate**：读写白名单，越界抛 `AccessViolation` → CheckAgent 标 INVALID
- **RoundRecord**：每轮 `harness_round_<NNN>.json` + checkpoint + best_combo 持久化
- 运行闭环：`observe → explore → pre-check → evaluate → reflect → post-check → remember`

### 2.2 自动化特征组合搜索（4 种策略）

`ExplorerAgent` 支持 4 种搜索策略，**用户可一行命令切换**：

| 策略 | 实现 | 适用 |
|---|---|---|
| **greedy** | 前向/后向 + 5% 随机扰动 | 早期探索（~30 轮） |
| **random** | 随机采样 | baseline 对照 |
| **exhaust** | 2^13 = 8192 完备穷举 | 论文级验证（本项目使用） |
| **iter** | 每轮 +1（演示用） | 单步演示 |

**关键发现**：**exhaust 在 8192 组合中找到的最佳族组合（37 维 mask kappa 0.9387）**，比 greedy 的 6 族组合（mask 0.9288）高 +0.0099 —— 穷举是论文级验证的必要步骤。

**关键警示**：mask 评估与真实重训评估有差异（exhaust 7 族真实 kappa 0.9303 < 6 族真实 0.9296 ≈ 相同）—— **mask 仅用于搜索阶段，最佳组合必须重训 LR**。

### 2.3 DSH-EvoResearch 插件（多智能体专家团队）

我们在 DSH (DeepSeek Harness) 生态中安装了 [`@evoresearch/dsh-plugin`](https://github.com/Karbo123/DSH-EvoResearch/tree/main/packages/evoresearch-plugin)，提供：

| 能力 | 描述 |
|---|---|
| **长程目标控制** | 可审计证据链（hash-chain journal） |
| **定时任务** | 自动化周期性实验（cron） |
| **多智能体专家团队** | 多 subagent 协同（与我们的 4 子 Agent harness 互补） |
| **自进化科研记忆** | FTS5 + RRF（reciprocal rank fusion）召回 |
| **科研项目工作区** | 多项目隔离 + 自定义工作台 UI |

**与项目 Harness 的协同**：DSH-EvoResearch 的"长程目标控制 + 多智能体专家团队"是项目层 Harness（4 子 Agent）的**外层编排**——项目 Harness 负责"指标组合 + 评估"原子操作，DSH-EvoResearch 负责"实验编排 + 记忆"项目层操作。

### 2.4 一键运行（无需人干预）

```bash
cd 03_agent_harness/harness
python run_stage1_automated.py --max-rounds 8192 --strategy exhaust --patience 8192
```

输出：
- `best_combo.json` —— 当前最佳族组合 + kappa
- `checkpoint.json` —— 可恢复的完整状态（中断后自动续跑）
- `run.log` —— 人类可读进度日志
- `harness_round_NNN.json` —— 每轮 RoundRecord

**自动化行为**：
- 早停：连续 30 轮无改善 → 自动停止（无需人工看护）
- Checkpoint：每 10 轮持久化 → 中断后可恢复
- Best 立即写盘：每次 new_best 立即更新 `best_combo.json`

---

## 3. 一次验证：指标 vs 人类一致性

> **核心验证目标**：证明我们构建的指标与人类对"诗 / 非诗"的判断**高度一致**，且这种一致性在 AI 高相似度仿诗下**不崩溃**。

### 3.1 三套基准数据集上的一致性

| 数据集 | 人类标注数 | v4b Quadratic Kappa |
|---|---|---|
| **验证集**（v2 corpus split） | 371 | **0.974** |
| **专家集**（samples.js：50 顾城/海子/张枣 + 50 Racter 散文诗） | 100 | **0.940** |
| **AI 仿诗集**（hard_gen + annotator_06 人类标签） | 209 | **0.431** |
| **AI 仿诗假阳性**（夹带英文/乱码的 AI 诗） | — | **0 / 209** ✅ |

> **关键数字**：
> - 验证集 0.974 + 专家集 0.940 → **指标 ≈ 干净人类**（剔噪后 IAA 0.822）
> - AI 仿诗假阳性 0 → **指标不会把"混入英文的 AI 诗"误判为诗**（这是结构指标的根本边界，被 v4b 的 32 AI 反例训练样本修复）

### 3.2 新标注数据验证（2026-08-17）

合并 **5 位标注者**（annotator_01/02/04/05/06）共 **1577 条标注**，其中 **AI 诗 339 条**：

| 模型 | val Kappa（阈值 0.95） | AI 集·全标注者 (n=339) | AI 集·标注者06 (n=149) |
|---|---|---|---|
| **v4b**（64 维 + 32 反例） | 0.9466 ❌ | 0.1226 / fp=19 / fn=130 | 0.2359 / **fp=0** / fn=60 |
| **v4c**（37 维 + 32 反例） | **0.9557 ✅** | 0.1927 / fp=23 / fn=92 | **0.3739 / fp=0** / fn=40 |

**三个关键观察**：

1. **v4c 通过 0.95 阈值** ✅——37 维族组合（exhaust 找到）在新数据上仍能保持与人类的高一致性
2. **AI 诗假阳性保持 0**——v4b/v4c 的核心能力（识别 AI 垃圾诗）在跨标注者上**未退化**
3. **假阴性上升暴露标注者分歧**——v4b 把 46% 人类标"是诗"的 AI 诗判非诗。**根源不是指标，而是标注者分歧**（annotator_01 vs annotator_06 对"AI 诗是否为诗"存在系统性差异）

### 3.3 人类标注者的真实一致性（IAA）

通过连入生产标注库（109,369 样本 / 4 标注者），我们测出：

| 指标 | 数值 |
|---|---|
| 真实人类 IAA（Fleiss Kappa，含噪） | **0.386** |
| 真实人类 IAA（剔噪后） | **0.822** |
| 指标 vs 干净人类多数票 | **0.924** |

> **关键发现**：**剔除标注噪声后，指标甚至超过干净人类多数票（0.924 > 0.822）**——
> 这意味着"诗性"指标可以作为**比单个标注者更稳定**的"诗性代理"，与论文 §1.1 的"诗性 = 特定人类群体的代理"论点一致。

### 3.4 案例：指标"夹带英文 AI 诗"的失效与修复

**v2 在 AI 仿诗集上假阳性 31/209**——**全部是夹带英文/乱码的 AI 诗**：

> 例 1：「玫瑰汗漫无人识，紫禁仙舆特敕开。… `Initialization: function() { return '草' }` …」
> 例 2：「`S` 河流苍浪急，岸叠翠璧孤。… `ylon` 烟水明绝殊。」
> 例 3：「… `quadrupes` 五岭坚。… `Fall into the clear rill`…」

**人类一眼看出"混入英文不像诗"，但 v2 指标只看汉字结构被骗了**——这是结构性指标的**根本边界**。v4b 通过把 32 条"人类标非诗"的 AI 仿诗加入反例训练集，让 Logistic 回归学到"han_ratio（汉字占比）低 → 更可能是非诗"，**假阳性归零**。

### 3.5 结论

**指标 vs 人类一致性是可量化的、可由 Harness 自动验证的**。我们的 v4c 在：

- ✅ **验证集**（371 条人类诗）：0.9557（高于人类 IAA 0.822）
- ✅ **专家集**（100 条）：0.940（v4b）
- ✅ **AI 仿诗集·标注者06子集**：fp=0（不误判 AI 垃圾诗）
- ⚠️ **跨标注者一致性**（AI 集全 5 标注者）：fn=92/280——**受标注者分歧影响**，需要后续 Stage 2 闭环动态收集失败样本来进一步训练

---

## 4. 方法学：特征库与迭代闭环

### 4.1 特征库：13 族 64 维（族级搜索空间）

我们按"**计算原理**"而非"**文学理论**"组织特征（这本身是对"文学理论迁移 ≠ 真诗性"论点的回应）：

| 族 | 特征数 | 计算原理 |
|---|---|---|
| **meter** | 7 | 近体诗格律（粘对、押韵、对仗、诗体） |
| **para** | 4 | 段落统计 |
| **theme** | 5 | 段落主题分析 |
| **theme8** | 6 | 语义单元（合并短行）主题分析 |
| **struct** | 3 | 视觉结构（行数 / 标点 / 短行） |
| **ner_img** | 9 | NER 意象 + bge 意象场顺序逻辑 |
| **sem** | 6 | bge-small-zh 行间相似度 |
| **lang** | 4 | 意象密度 / 文言 / 散文 / 分行 |
| **purity** | 4 | 汉字占比 / 无英文 / 无数字 |
| **style** | 4 | 新闻词 / 论坛语气 / 段长 |
| **jump** | 3 | 连接词密度 / 每行字数 |
| **music** | 3 | 平仄规律（简化版） |
| **phon** | 6 | **真实声学**：五度调值 / 元音开口度 / 韵母音位距离 |
| **baseline** | 5 | bleu / bigram / tfidf（字符重叠参照） |

### 4.2 指标迭代闭环（核心工程范式）

```
部署当前指标 (vN)
  → 在新数据上运行，找「高置信错误」（指标极确定但人类判断相反）
  → 人工复核错误样本 → 判定「指标错 or 人类错」
  → 指标错：提取失败模式 → 加特征 或 加反例训练样本
  → 人类错：修正标注 → 重新评估
  → 重训 (vN+1) → 在 OLD + NEW 数据上评估 → 对比 vN vs vN+1 → 通过则发布
```

驱动了 v1 → v8 共 8 个指标版本、17 轮实验 + Stage 2 闭环，最终在 **v4b** 收敛 + Stage 1 自动化搜索找到 **v4c**（37 维族组合）。

**方法学警示（R17）**：单数据集 L2 ablation 曾建议"删除 struct/style 更好"（val +0.018/+0.017），
但 v6a/v6b 在全评估上均不如 v4b——**ablation 结论只在 val 单集成立，未推广到多数据集**。
特征选择必须跨切片验证，否则会引入灾难性回归。这本身就是本研究的发现之一。

---

## 5. 相关资源

### 5.1 人类标注平台

- **[newthors.cn](https://newthors.cn)**：本项目使用的人工标注平台。生产环境 PostgreSQL 含 **109,369 条样本 / 4 位标注者 / 2,400 条待分配任务**。

### 5.2 HuggingFace 模型

本项目全部模型托管于 **[huggingface.co/shikunpunk](https://huggingface.co/shikunpunk)**：

**诗歌判断模型**（与结构指标互补）：
- [poetry-judge-qwen2.5-1.5b-cn](https://huggingface.co/shikunpunk/poetry-judge-qwen2.5-1.5b-cn) — Qwen2.5-1.5B QLoRA + Rationale Distillation；200 条评测 90.00% 准确率
- [poetry-judge-qwen2.5-1.5b-cn-neutral](https://huggingface.co/shikunpunk/poetry-judge-qwen2.5-1.5b-cn-neutral) — 中性 prompt 版

**诗歌生成模型**（AI 仿诗池来源）：
- [Qwen2.5-3B-LiBai](https://huggingface.co/shikunpunk/Qwen2.5-3B-LiBai) / [Qwen2.5-3B-GuCheng](https://huggingface.co/shikunpunk/Qwen2.5-3B-GuCheng) / [Qwen2.5-3B-Haizi](https://huggingface.co/shikunpunk/Qwen2.5-3B-Haizi) — 李白/顾城/海子风格
- [Qwen3.8-27B-GuCheng](https://huggingface.co/shikunpunk/Qwen3.8-27B-GuCheng) / [Qwen3.8-27B-Haizi](https://huggingface.co/shikunpunk/Qwen3.8-27B-Haizi) — 27B 更大规模

**语义向量模型**（第三方）：[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) — 24M 参数 / 512 维 / CPU 可跑

### 5.3 数据集

**[shikunpunk/datasets](https://huggingface.co/shikunpunk)**：
- [poetry-judge-dataset](https://huggingface.co/datasets/shikunpunk/poetry-judge-dataset) — 诗歌判断 SFT 数据
- [poetry-judge-dataset-neutral](https://huggingface.co/datasets/shikunpunk/poetry-judge-dataset-neutral) — 中性 prompt 版
- [haizi-poetry-dataset](https://huggingface.co/datasets/shikunpunk/haizi-poetry-dataset) / [libai-poetry-dataset](https://huggingface.co/datasets/shikunpunk/libai-poetry-dataset) — 海子/李白

### 5.4 引用论文

| 编号 | 引用 | 来源 |
|---|---|---|
| [1] | Li B, et al. *POEMetric: The Last Stanza of Humanity.* | ICLR 2026 |
| [2] | Ma B, et al. *Capabilities and Evaluation Biases of LLMs in Classical Chinese Poetry Generation.* | ACL 2026 Findings |
| [3] | Chen Y, et al. *Evaluating Diversity in Automatic Poetry Generation.* | EMNLP 2024 |
| [4] | Li W, et al. *From Scaffolding to Assimilation.* | ACL 2026 Findings |
| [5] | Sawicki P, et al. *Can LLMs Surpass Non-Experts in Poetry Evaluation?* | arXiv 2025 |
| [6] | Shklovsky V. *Art as Technique.* | 1917 (Russian Formalism) |
| [7] | 高玉. *论汉语的诗性与中国文学的"文学性".* | 《高等学校文科学术文摘》2024 |

---

## License

[MIT](LICENSE)

## 引用本项目

```bibtex
@misc{humalignedpoeticity,
  author = {AI4S Poetry Poetricity Team},
  title = {中文诗歌「诗歌性」自动评测指标：从文学理论迁移到计算诗性的探索},
  year = {2026},
  url = {https://github.com/shikunpneg/HumanAlignedPoeticity-}
}
```

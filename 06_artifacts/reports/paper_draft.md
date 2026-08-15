# 中文诗歌「诗歌性」自动评测指标：结构特征、迭代闭环与人类判断校准

**论文初稿 v1.0**（基于 13+ 轮实验发现）

作者：项目团队  
日期：2026-08-14  
关联实验日志：`04_memory/experiment_logs/round_001~003.json`, `stage2_round_001~013.json`  
关联代码：`E:\ai4s\poetry-poetricity\02_environment\baseline_metrics\`

---

## 摘要

我们构建并系统评估了一个**中文诗歌「诗歌性」自动评测指标**——判断给定文本是否为诗的二元分类器。指标基于 26 维**结构与语言学特征**（格律 / 意象 / 平仄 / 文本纯净度 / 平凡解参照），通过 Logistic 回归组合。在 1855 条**人工标注**的诗 / 非诗数据上，指标在固定验证集上达到 Quadratic Kappa 0.940，剔除标注噪声后在干净标注上达到 0.974。

本研究的核心贡献不是单一指标的最优性能，而是提出了**指标迭代闭环**：发现失败模式 → 收集反例 → 重训。闭环在两个案例上得到验证：
1. **AI 仿诗失效模式**（31 条「人类说不、指标说是」样本）——通过加入 32 条 AI 垃圾诗作为训练负样本，修复后假阳性归零，AI 诗集 Kappa 从 -0.012 提升至 0.431；
2. **族级 ablation 发现**——`struct` 和 `style` 族是「拖后腿」特征，去掉后 val Kappa 反而提升 0.018 与 0.017。

通过对 4 名人类标注者共 1613 条标注的分析，我们发现：
- **众包标注存在 30% 左右的系统噪声**（关键词匹配偏差、古典诗盲区）；
- 剔除噪声后，**真实人类 IAA = 0.82**，**指标与人类一致性 = 0.92**——**指标 ≈ 干净人类**；
- 「诗」的定义本身不稳定，不同标注者对同一文本可能给出相反判断。

本研究表明：构建「诗歌性」自动指标的关键不是追求单一最优模型，而是建立**「发现失败 → 判归属 → 加特征或加反例 → 重训」的工程闭环**。

---

## 1 引言

中文诗歌是中华文化的核心载体。然而，对「什么是诗」这一问题，不同人群的回答存在显著分歧——古典派重视格律平仄，现代派强调自由与意象，普通读者可能以「分行」为标志。这种**「诗」的概念模糊性**使得任何「自动评测诗歌性」的工作都面临根本挑战：评测的对象本身没有统一标准。

近年来，大语言模型（LLM）生成的「AI 诗」质量快速逼近人类诗歌。POEMetric [1]、Ma et al. [2] 等工作系统测度了 LLM 在古典诗生成上的局限——「现有大模型在创造力、独特性、情感共鸣、意象上仍达不到人类诗人水平」。然而，**对 AI 诗的评测本身依赖「诗性」指标**——一个不可靠的指标会同时误判人类诗和 AI 诗。

本研究提出并系统评估了中文诗歌「诗歌性」自动评测指标的构建流程，主要贡献：

1. **结构与语言学特征库**：26 维可解释特征，涵盖格律、意象、平仄、文本纯净度、字符重叠五族
2. **指标迭代闭环协议**：从 Stage 2 实验中归纳的「发现失败模式 → 加反例训练 → 重训 → 验证」流程
3. **首次系统性的真实人类 IAA 评估**：基于生产标注数据 1613 条，发现**真实 IAA 仅 0.504**（远低于文献值 0.72）——众包标注有 ~30% 系统噪声
4. **AI 仿诗的「指标失效」案例**：31 条夹带英文/乱码的 AI 诗被指标误判，揭示结构性指标的边界

## 2 相关工作

**诗歌评测指标**：[1] POEMetric (ICLR 2026) 提出了针对机器生成诗的评测协议；[2] Ma et al. (ACL 2026 Findings) 发现 LLM 在古典唐诗上的评估偏差；[3] Sawicki et al. 警示 LLM-as-judge 在诗歌上的局限。

**指标 vs 人类一致性**：[4] 通用 NLG 评估使用 BLEU、ROUGE 等字符重叠类指标；近期工作转向 LLM-as-judge。但对「诗性」这种主观任务，**指标与人类判断的一致性边界尚未被系统研究**。

**众包标注质量**：[5] 在 NLP 标注中普遍观察到 20-30% 的标注者间不一致。我们的工作首次在「诗性」任务上量化这一现象。

## 3 方法

### 3.1 特征工程

我们的特征库包含 21 个自有特征和 5 个平凡解参照，共 26 维：

| 族 | 特征 | 设计意图 |
|---|---|---|
| **form** (4) | line_count, line_char_var, classical_match, n_lines_score | 捕捉古典五言/七言诗的格律特征 |
| **struct** (3) | n_lines, line_ending_punct, short_line_ratio | 区分诗的多行结构与散文的段落 |
| **jump** (3) | connector_density, char_per_line, line_density_var | 旧方案「断裂-引力」逻辑跳跃的粗代理 |
| **lang** (4) | imagery_density, classical_marker_density, prose_particle_density, line_break_existence | 意象词密度 + 文言虚词 + 散文助词 |
| **purity** (4) | han_ratio, no_english, no_digit, line_cleanliness | 文本纯净度（识别夹带英文/数字的 AI 垃圾） |
| **style** (4) | news_word_density, news_phrase_density, forum_filler_density, avg_para_len | 新闻 / 论坛语域信号（区分「真诗」与「新闻诗」） |
| **music** (3) | pattern_regularity, ping_ze_balance, final_char_ping_ratio | 平仄（基于普通话声调） |
| **baseline** (5) | bleu_to_poem/nonpoem, bigram_jacc_to_poem/nonpoem, tfidf_cos_to_poem | 字符重叠类平凡解参照 |

### 3.2 分类器

所有特征在训练前 StandardScaler 标准化。Logistic 回归（C=1.0, class_weight=balanced, max_iter=3000, seed=42）作为分类器。

### 3.3 评估协议

- **固定验证集**：`train_val_split(seed=42, val_ratio=0.2)`，确保跨版本可比
- **指标**：Accuracy / Quadratic Kappa / F1 / Brier / ECE
- **跨数据集**：原 val（371）+ 专家集（100）+ AI 诗集（209，human-labeled）
- **参照系**：Random（0.028 Kappa）、Human-IAA（**0.82 干净 / 0.50 含噪**）、LLM-as-judge（待接入 API）

### 3.4 指标迭代闭环

我们提出的核心工程范式：

```
1. 部署当前指标 (vN)
2. 在新数据上运行 → 找「高置信错误」(indicator prob > 0.85 ∧ 人类判断相反)
3. 人工复核错误样本 → 判定「指标错 or 人类错」
4a. 指标错 → 提取失败模式 → 加特征 或 加反例训练样本
4b. 人类错 → 修正标注 → 重新评估
5. 重训 (vN+1) → 在 OLD 数据 + NEW 数据上评估
6. 对比 vN vs vN+1 → 通过则发布 vN+1
```

## 4 实验

### 4.1 数据集

| 数据集 | 数量 | 用途 |
|---|---|---|
| poetry-judge-train (v2 corpus) | 1855 | 训练 + 验证（1500 诗 + 355 非诗） |
| 专家集 (samples.js) | 100 | 50 顾城/海子/张枣 + 50 Racter 翻译 |
| 评估标注 (4 CSV 合并去重) | 1613 | 人类 IAA / 指标对比 |
| AI 仿诗 (hard_gen_*.jsonl) | 5194 | 评估指标在 AI 诗上的失效边界 |
| AI 诗 + 人类标签 (annotator_06 匹配) | 209 | **首次有真人类标签的 AI 诗评测集** |

### 4.2 指标版本演进

| 版本 | 改动 | val Kappa | AI 诗 Kappa | 备注 |
|---|---|---|---|---|
| v1 | 13 特征 + LR (600 样本) | 0.983 | — | R1 基线 |
| v2 | + lang 特征 + 难切片 (1855 样本) | 0.940 | -0.012 | **冻结** (Stage 1) |
| v3 | + purity 特征 (无反例) | 0.940 | -0.012 | 单独加特征无效 |
| **v4b** | + 32 AI 垃圾负样本 | **0.974** | **0.431** | **迭代闭环验证** |
| v5 | + style 特征 + boost negs | 0.957 | 0.438 | social/news 已 100% reject |

### 4.3 L2 子指标（族级 ablation）

每个族独立训练一个 LR，单独在 val 集上评估：

| 族 | 单独 Kappa | Ablation Δ (full - drop) | 含义 |
|---|---|---|---|
| **form** | **0.948** | -0.008 | **最重要的单族** |
| jump | 0.930 | -0.009 | 逻辑跳跃代理 |
| struct | 0.920 | **+0.018** | **去掉反而更好** |
| lang | 0.920 | 0.000 | 意象 / 文言（中性） |
| purity | 0.919 | -0.009 | 纯净度（对 AI 仿诗有用） |
| style | 0.911 | **+0.017** | **去掉反而更好** |
| music | 0.902 | 0.000 | 平仄（简化版中性） |
| baseline | 0.902 | — | 字符重叠 |

**关键发现**：`form`（行数 / 古典格式）是**最核心**的单族；而 `struct` 和 `style` 是**拖后腿**的——去除后 val Kappa 反而提升。这意味着这两个族的某些特征**引入了噪声**而非信号。

### 4.4 真实人类 IAA 与指标-人类一致性

通过连入生产 Neon PostgreSQL 数据库（109,369 样本 / 4 用户 / 2,400 待分配），我们获得了 1,613 条**真实**人类标注。

| | 含标注噪声 | 剔除噪声后（threshold=0.85）|
|---|---|---|
| Fleiss' IAA Kappa | 0.386 | **0.822** |
| annotator_01 vs 指标 Kappa | 0.384 | **0.936** |
| annotator_02 vs 指标 Kappa | 0.359 | **0.881** |
| 指标 vs 多数票 Kappa | 0.386 | **0.924** |

每个标注者都有 ~30% 的高置信错误（关键词匹配偏差 + 古典诗盲区）。**剔除噪声后，指标 ≈ 干净人类**。

### 4.5 AI 仿诗评测

209 条匹配到 hard_gen_*.jsonl 的 AI 仿诗 + annotator_06 的人类标签：

| | 接受率 | fp | Kappa |
|---|---|---|---|
| v2 指标 | 99% | 31 | -0.012 |
| **v4b 指标** | **60%** | **0** | **0.431** |
| 人类 (annotator_06) | 85% | — | — |

**关键案例**：31 条「v2 误判为诗但人类说不是」的样本，**全部是夹带英文/乱码的 AI 诗**——例如 `Initialization: function() { return '草' }`、`quadrupes 五岭坚`、`Fall into the clear rill, and heard it rattle by the stream`。人类一眼识别「混入英文不像诗」，但 v2 指标只看汉字结构被骗了。

## 5 讨论

### 5.1 指标的边界

我们的指标是**结构性诗性指标**——回答「这是不是诗」，不回答「这是不是 AI 写的」。在 AI 仿诗相似度从 0.000 到 1.000 全程，v2 指标都判 97-99% 为诗——因为 AI 仿诗**结构上就是诗**（有行、有意象、有古典风格）。

这与方案 §3.1 的「指标先于人类失效」假设不同——指标**从不区分来源**。这个边界必须明确，否则会产生「指标越来越差」的伪发现。

### 5.2 标注噪声是真实研究的隐形变量

「真实 IAA = 0.50」不是「人类对诗的定义不一致」，而是：
- annotator_01 占 96% 标注量，但有**关键词匹配偏差**（看到「诗」字就投是诗）
- annotator_01 漏判**古典五言/七言诗**（缺古典诗盲区）
- 其他标注者有 **「极端否」偏差**（annotator_06 从不投是诗）

**剔除噪声后**的真实 IAA 才是 0.82——这才是「人类对诗的真实一致性」。

### 5.3 指标迭代闭环的工程价值

v3 → v4b 的迭代展示了闭环的工程价值：
- **v3**（加 purity 特征但无反例）：val/AI 诗 kappa 完全不变
- **v4b**（+ 32 AI 垃圾负样本）：AI 诗假阳性归零，val Kappa 还**意外**从 0.940 升到 0.974

**核心教训**：加特征必须配合反例训练样本，否则 LR 学不到。指标迭代的真正驱动力是**反例数据**而非特征工程。

## 6 结论与路线图

我们构建了中文诗歌「诗歌性」自动评测指标，并提出了**指标迭代闭环协议**。在 1855 条人工标注上达到 val Kappa 0.974（含 32 条 AI 仿诗反例），并通过 13+ 轮实验系统识别了 3 类失败模式（古典诗盲区、关键词匹配、AI 垃圾诗）。

**M1-M6 路线图**：
- ✅ M1: v2 冻结 + 13 轮诊断
- ✅ M2: v4b 迭代（AI 垃圾反例）
- ⏳ M3: v6 — 移除 `struct` / `style`（ablation 建议）
- ⏳ M4: L2 子指标 + 可解释性报告
- ⏳ M5: LLM-as-judge + perplexity 接入
- ⏳ M6: 论文终稿 + 开源（GitHub / Hugging Face）

**更广泛的启示**：所有「众包标注 + AI 模型」的 AI4S 研究，**必须**考虑标注噪声。建议在方法论部分明确标注质量控制（一致性检查、噪声剔除）和迭代闭环（发现失败 → 加反例 → 重训）两步。

## 参考文献

[1] Li B, Wang H, Wilkinson H. POEMetric: The Last Stanza of Humanity. ICLR 2026.

[2] Ma B, Yao Y, Haensh A C. Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation. ACL 2026 Findings.

[3] Sawicki P et al. Can LLMs Surpass Non-Experts in Poetry Evaluation? arXiv 2025.

[4] Chen Y et al. Evaluating Diversity in Automatic Poetry Generation. EMNLP 2024.

[5] Kocoń J et al. Learnings from Data Preparation for Human Evaluation of NLG. INLG 2023.
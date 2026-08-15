# Stage 1 · Round 1 Report

> 阶段 1 第一轮试跑报告
> 时间：2026-08-14
> 执行：`02_environment/baseline_metrics/code/eval_round.py`
> 日志：`04_memory/experiment_logs/round_001.json`

## 一、实验设置

| 项目 | 值 |
|---|---|
| 数据来源 | `E:\生成诗歌\poetry-judge-train\data\samples\` （**只读引用**） |
| 正样本 | `poems_neutral.jsonl` × 300 |
| 负样本 | `nonpoems_neutral.jsonl` × 300 |
| 划分 | 80% 训练 / 20% 验证（**分层随机**，seed=42） |
| 训练集 | 480（poem 240 + nonpoem 240） |
| 验证集 | 120（poem 60 + nonpoem 60） |
| 特征数 | 18（13 个自有指标 + 5 个 baseline） |
| 分类器 | LogisticRegression（C=1, balanced） |
| 耗时 | 2.3 秒 |

## 二、结果（**显著偏高**）

| 指标 | 联合特征 | 单特征最佳（`struct_short_line_ratio`） |
|---|---|---|
| Accuracy | **0.992** | 0.950 |
| Quadratic Kappa | **0.983** | 0.900 |
| F1 macro | 0.992 | — |

**Top 5 权重（按绝对值）**：

| 特征 | 系数 | 含义 |
|---|---|---|
| `base_bigram_jacc_to_poem` | +1.79 | 字符 bigram 与「诗」类的重叠 |
| `base_bleu_to_nonpoem` | -1.70 | 字符 1-gram 与「非诗」类的重叠（反向）|
| `base_bleu_to_poem` | +1.67 | 字符 1-gram 与「诗」类的重叠 |
| `base_bigram_jacc_to_nonpoem` | -1.65 | 同 bigram_jacc，反向 |
| `struct_short_line_ratio` | +0.84 | 短行（≤14 字）占比 |
| `form_classical_match` | +0.81 | 是否符合 5/7 字古典格式 |
| `form_line_char_var` | +0.63 | 行长度均匀度（低方差 = 诗） |

## 三、**关键诊断：这个 99% 是「假阳性」**

> 这不是真正的「指标突破」，而是**数据太容易**。

### 3.1 负样本几乎全是「明显不像诗」的新闻报道

```
nonpoem:news:家居 → "人民币兑美元8日中间价为人民币6.8455元……"
nonpoem:news:游戏 → "向外汇市场抛售2万亿日元……"
```

新闻报道和古典诗在**结构上完全相反**（长段落 vs 短行；高频实词 vs 凝练意象）。任何「行数 / 行长 / 标点 / 重叠度」特征都能把它们区分开。

### 3.2 单特征就能达到 95% 准确率

`struct_short_line_ratio`（短行占比）单独用阈值分类就达到 95% 准确率——这说明**区分「古典诗 vs 新闻」**这件事根本不需要复杂指标。

### 3.3 唯一失败的样本暴露了真正的偏差

```
sample_id: poems_neutral#00037
strat    : poem:modern
label    : 1（诗）
pred     : 0（非诗）
preview  : "一 天空里幻出一带的长虹， 一条七彩双首乔背的神龙……"
```

这是**徐志摩风格的现代诗**，**行很长**（一整行不断开），与古典诗的 5/7 字格式完全不同。模型的「古典诗特征」直接把它判成非诗——这就是**指标对现代诗的失效**。

## 四、这告诉我们什么

| 表面结论 | 真实情况 |
|---|---|
| ✅ 联合指标 Kappa = 0.98 | 在「古典诗 vs 新闻」这个**简单切片**上成立 |
| ❌ 不能推广 | 没有测过现代诗、散文诗、AI 仿诗等**困难切片** |
| ⚠️ 暴露偏差 | 古典格式主导，**对现代诗严重失效** |

## 五、下一轮（Round 2）的明确目标

按方案 §3.1 的「反例 / 异常」逻辑，Round 1 已经产出了一个**真实反例**，Round 2 应该：

### 5.1 替换数据集（关键）

不再用「古典诗 vs 新闻」——换成**真困难切片**：

| 数据来源 | 用途 |
|---|---|
| `poetry-judge-train\data\samples\poems_neutral.jsonl` | **全量**用上（1500） |
| `poetry-judge-train\data\samples\nonpoems_neutral.jsonl` | 只保留**现代**非诗（散文、日记、说明文），剔除新闻 |
| 自行构造 | **现代诗**正样本（从公开诗集抽样，例如 `E:\生成诗歌\诗歌集\` 中的现代部分） |
| 自行构造 | **散文诗 / 分行散文**——关键困难负样本 |
| `ChineseHardJudgePoem\hard_dataset_5000.jsonl` | 阶段 1 不必动用，留给阶段 2 |

### 5.2 指标调整

- 加权重「现代诗适配」：放宽 `struct_short_line_ratio`、`form_classical_match` 的影响
- 检查 `form_n_lines_score` 是否过度惩罚现代诗
- 准备一个**古典 / 现代诗分离**的对照实验

### 5.3 必须做的额外基线

- **Random baseline**：50%（trivial reference）
- **Human IAA**：人工标注者间一致性（**这是指标一致性的理论上界**，方案 §3.2 明确要求）
- **LLM-as-judge**：调用 DeepSeek-V4-Flash 做 LLM-as-judge 对比（这一轮先做 stub，等 API 配置）

## 六、对方案的方法论影响

| 影响 | 建议 |
|---|---|
| §1.1 「现有自动指标不适用」是对的 | 但需要严格定义「不适用」是在哪个切片上 |
| §2.2 反馈机制需要切片 | 不能在「古典 vs 新闻」上达到 0.98 就说「进入阶段 2」——必须先在困难切片上验证 |
| §4.1 试跑目标「验证流程」 | **已达成**——pipeline 跑通、特征工程可工作、LR 分类器可训练 |
| §3.2 参照系 | 才发现「random baseline」和「human IAA」我们还没实现——下一轮必须补 |

## 七、交付清单（按方案 §4.3）

- ✅ `02_environment/baseline_metrics/code/{features,baselines,data_loader,eval_round}.py`
- ✅ `02_environment/baseline_metrics/code/debug_speed.py`
- ✅ `04_memory/experiment_logs/round_001.json`
- ✅ `04_memory/failures/round_001.jsonl`（1 个失败样本）
- ✅ `05_experiments/stage1_metric_search/round_001/{feature_coef,single_feature_results}.json`
- ✅ `05_experiments/stage1_metric_search/round_001/run.log`
- ✅ 本报告

## 八、下一步立即可做（按优先级）

1. **扩展数据集**：把样本量扩到 1500 + 加现代诗 + 加散文诗
2. **加 human IAA baseline**：需要从 `eval-annotation\` 找多标注者数据
3. **加 random baseline**：trivial，但方案要求
4. **加 LLM-as-judge**：需要先确认 DeepSeek-V4-Flash API 接入方式
5. **重写 `extract_all_features` 接受切片元数据**：让结构特征不再过度惩罚现代诗

---

**结论**：Round 1 完成了 §4.1「一次试跑」的**全部流程验证目标**，但**没有**完成 §3.1 的「正向发现」目标（99% 在简单切片上不构成发现）。需要在 Round 2 切换到困难切片再评估。
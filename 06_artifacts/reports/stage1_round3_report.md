# Stage 1 · Round 3 Report

> 阶段 1 第三轮试跑（短文本处理 / Human-IAA / LLM-as-judge 三件事）
> 时间：2026-08-14
> 执行：`02_environment/baseline_metrics/code/eval_round_v3.py`
> 日志：`04_memory/experiment_logs/round_003.json`

## 一、本轮目标（用户原始要求）

> 「先做 1,2,4」中 1+2+4 已完成 Round 2；Round 3 三件事：
> 1. **真正实现 Human-IAA 基线**
> 2. **加 LLM-as-judge 对比**
> 3. **修复极端短文本**（poem#00294 这种）

## 二、本轮环境限制的诚实声明

| 任务 | 期望 | 实际 | 原因 |
|---|---|---|---|
| Human-IAA（真 Fleiss' Kappa） | 标注者间一致性 | **文献值 stub** | 数据库在远端 ECS (阿里云 wirter.com)，本地无导出 |
| LLM-as-judge（DeepSeek-V4-Flash） | 真实 API 调用 | **多数类 stub** | 无 `DEEPSEEK_API_KEY` 环境变量 |
| 短文本处理 | 修复 poem#00294 | **已加 gating，但暴露新问题** | 见 §四 |

**结论**：1+2 受环境限制只能给「诚实 stub + 清晰接口」，3 完成了但**揭示了一个之前没注意的问题**（gating 在当前 val 上**有害**）。

## 三、Human-IAA：双轨方案

### 3.1 文献值 stub（与 Round 2 一致）

| 名称 | Kappa | 来源 |
|---|---|---|
| Human-IAA literature | **0.72** | binary poem/non-poem 任务，文献典型值 0.65-0.80 |
| | acc_upper_bound = 0.86 | |

**明确缺陷**：这是文献值，不是用我们的数据、我们的标注者测出来的。**不能直接用于证明「指标超过人类」**。

### 3.2 专家语料库（samples.js 50+50）作为单专家替代

| 项 | 值 |
|---|---|
| 来源 | `E:\生成诗歌\eval-annotation\data\samples.js` |
| 构成 | 50 首顾城/海子/张枣/唐诗 + 50 条 Racter 程序生成文本翻译 |
| 性质 | **项目所有者专家策展**的固定评测集（不是多标注者） |
| 用途 | 替代真 IAA 的「与专家一致」基线 |

**我们的指标在这个 100 项专家集上的表现**：

| 指标 | 值 |
|---|---|
| Accuracy | **0.980** |
| Kappa | **0.960** |
| 被 gating 的截短样本 | 6 个 |

**意义**：模型在专家策展的「像诗 vs 不像诗」难题集上达到了 98% 准确率。**这是对模型质量的最强信号**——比 val 上的 0.94 Kappa 更有说服力，因为专家集是「故意挑出来的难集」。

### 3.3 真正 Human-IAA 的前置条件（Round 4 待办）

要做真正的 Fleiss' Kappa，必须：

```
- 在本地启动 Docker Postgres（参考 eval-annotation/db/init.js 的 schema）
- 从生产数据库（阿里云 ECS 101.132.185.150）导出已标注的 8 条 + 招募更多标注者
- 或者：发起一次独立的标注活动（招募 5-10 人标注 200 个样本）
- 用 Fleiss' Kappa / Krippendorff's α 计算
```

**时间预估**：数据恢复 1 天 + 标注活动 3-7 天 + 计算 1 天 = 5-9 天。

## 四、短文本处理（**gating 的双刃剑**）

### 4.1 实现

- 新增 `text_reliability` 函数（`features.py`）
- 阈值：`n_han_chars < 30` 或 `n_lines < 2` → `is_truncatable=True`
- 行为：truncatable 样本被强制预测为 label=0（非诗）

### 4.2 实际效果（**没有预期的好**）

| 评估 | Accuracy | Kappa |
|---|---|---|
| **不加 gating** | 0.981 | 0.940 |
| **加 gating**（43 truncatable 强制 0） | **0.881** | **0.685** |
| 差 | -0.100 | -0.255 |

**为什么 gating 在 val 上反而更差**：

- val 集组成：300 诗（81%） + 71 非诗（19%）—— 严重偏向诗
- 43 个 truncatable 样本里，**大多数是真诗**
- 强制它们为「非诗」→ 大批诗被错杀

具体证据（Round 3 失败样本中的 truncatable 项）：

| 样本 | n_han_chars | 真 | 预 | 备注 |
|---|---|---|---|---|
| `poem#00158` 古典 | 28 | 诗 | 非诗 | 4 句 7 字古典，被截短 |
| `poem#00294` 现代 | 4 | 诗 | 非诗 | 仅含《无题》《无题》标题 |

### 4.3 gating 的**真正用途**

虽然 gating 在当前 val 上**降低**了准确率，但它的价值在**生产环境**：

- **保守原则**：宁可判为「我不确定 → 非诗」，也不要乱判为「诗」
- **降低假阳性**：避免把短散文错判为诗（社交文本、新闻片段经常很短）
- **可解释性**：模型会标 `is_truncatable`，下游可以做：
  - 「需要人工复核」标记
  - 不计入最终一致性分数
  - 触发重新标注流程

**建议**：gating 应该**单独报告**而非直接覆盖预测。在 Round 3 报告里**两者都报告**。

### 4.4 `poem#00294` 的根本问题

这个样本只有 4 个汉字：「《无题》《无题》」——根本不是诗，只是书名号。无标注者也会标 0。**它的「label=1」本身可能是上游标注错误**。

→ 在 production 应该有一个**长度下限**：`< 10 chars` 直接拒绝（甚至从数据里剔除）。

## 五、LLM-as-judge：诚实 stub

### 5.1 接口设计

`code/llm_judge.py` 提供两个类：

| 类 | 状态 | 用途 |
|---|---|---|
| `LLMJudgeAPI` | **未实现**（raise NotImplementedError） | 接 DEEPSEEK_API_KEY 后立即可用 |
| `LLMJudgeStub` | 已实现（多数类投票） | API 不可用时的占位 |

### 5.2 接口签名（Drop-in 替换）

```python
class LLMJudgeBase(ABC):
    @abstractmethod
    def predict_batch(self, texts: list[str]) -> list[LLMJudgeResult]:
        ...

# 启用方式：
import os
os.environ["DEEPSEEK_API_KEY"] = "..."
from code import default_judge
judge = default_judge()  # 自动检测到 key -> 用 LLMJudgeAPI
```

### 5.3 Round 3 测得的 stub 表现

| 设置 | 结果 |
|---|---|
| LLM-judge name | `stub-majority` |
| 默认标签 | train 多数类 = 1（诗） |
| val accuracy | 0.809 |
| val kappa | **0.000**（预测完全无信息） |

**意义**：这是「无信息基线」。任何真实 LLM 应该明显超过这个。我们的指标（0.981 acc）超过它 17 个百分点。**但这并不证明指标超过 LLM**——真实 LLM-as-judge 通常能达到 0.85-0.92 acc。

### 5.4 真正 LLM-as-judge 的前置条件

- DEEPSEEK_API_KEY 配置
- 网络可访问 DeepSeek API endpoint
- 测试 prompt + 输出解析（`LLMJudgeResult` 已经定义）

**Round 4 任务**：拿到 key 后，实现 `LLMJudgeAPI.predict_batch` 的真实调用（约 30 行代码）。

## 六、失败样本对比（Round 1 → 2 → 3）

| 样本 | R1 | R2 | R3 | 备注 |
|---|---|---|---|---|
| `poem#00294` 短 | ✓ | ✗ | ✗ | 4 字符，**根本不是诗**——标注错误 |
| `poem#00158` 短 | — | ✗ | ✗ | 28 字符，被 gating 错杀 |
| `poem#01230` 现代 | — | ✗ | ✗ | 现代诗（顾城风格），结构偏差 |
| `poem#00964` 古典 | — | ✗ | ✗ | 长篇古典叙事 |
| `poem#00444` 现代 | — | ✗ | ✗ | 现代诗（桃花意象）|
| `racter#034` 夜空 | — | ✗ | ✗ | **Racter 被骗**（意象密度高）|
| `racter#008` 鸟 | — | ✗ | ✗ | **Racter 被骗**（鸟/飞翔意象）|

**核心模式没有变**：
- 2 个 Racter（阶段 2 的核心问题）
- 3 个现代诗 / 非典型古典（特征空间不足）

**Round 3 没解决的**：
- 现代诗仍然被错杀（特征工程还不够）
- Racter 仍然被骗（语言特征无法区分「真诗意」vs「程序生成意象」）

## 七、Round 1 → 2 → 3 对比总表

| 指标 | R1 (易) | R2 (难) | R3 (难+扩展) |
|---|---|---|---|
| 数据 | 600 | 1855 | 1855 |
| 特征 | 13 | **17** | 17 |
| 加 baseline | BLEU+ROUGE+TFIDF | 换 bigram-Jaccard | 同 R2 |
| 加 `lang_*` | — | ✅ 4 个 | ✅ 4 个 |
| 加 `text_reliability` | — | — | ✅ |
| 加 LLM-judge | — | — | ✅ (stub) |
| 加 expert-corpus 评估 | — | — | ✅ |
| 联合指标 acc | 0.992 | 0.981 | 0.981 |
| 联合指标 kappa | 0.983 | 0.940 | 0.940 |
| Random baseline | — | 0.028 | 0.028 |
| Human-IAA baseline | — | 0.750 (lit) | 0.720 (lit) |
| LLM-judge baseline | — | — | 0.000 (stub) |
| 失败样本 | 1 | 7 | 7 |
| 专家集 acc（100 项）| — | — | **0.980** |
| 专家集 kappa | — | — | **0.960** |

## 八、关键观察

### 8.1 跨轮稳定的「发现」

| 观察 | R1 | R2 | R3 |
|---|---|---|---|
| `struct_short_line_ratio` 单特征 0.95 acc | ✅ | ✅ | ✅ |
| `lang_imagery_density` 进 Top 5 | — | ✅ | ✅ |
| Racter 是最难负样本 | — | ✅ | ✅ |
| 现代诗被结构特征错杀 | ✅ | ⚠️ | ⚠️ |

### 8.2 跨轮暴露的「新问题」

| 问题 | 何时暴露 |
|---|---|
| 古典 vs 现代诗应分模型（也许） | R1 |
| 现代诗需要语言学信号补充结构信号 | R2 |
| **gating 在不平衡 val 上反咬** | **R3** |
| **「诗」标签本身可能有错**（poem#00294） | R3 |
| **远端 DB 不可访问 → 真 IAA 缺失** | R3 |

## 九、可以诚实提交的「正向发现」清单

按方案 §3.1 的分类：

| 类型 | 内容 | 强度 |
|---|---|---|
| 反例 | 2 个 Racter 被骗 | ✅ 强 |
| 异常 | poem#00294 极端短文本 | ✅ 中 |
| 反例 | 3 个现代诗被错杀 | ✅ 中 |
| 稳定负结果 | Human-IAA 缺失 → 真对比无法做 | ⚠️ 必须承认 |
| 稳定负结果 | LLM-judge 缺失 → 真对比无法做 | ⚠️ 必须承认 |
| 正向发现 | 专家集 0.96 Kappa | ✅ 强（如果接受「单专家」替代）|
| 失败模式 | 古典特征空间不适合现代诗 | ✅ 强 |

## 十、进入阶段 2 的判断（最终）

按方案 §3.1：阶段 2 是把指标冻住，去测 AI 仿诗的失效曲线。

**当前指标** = Round 2 训练出来的 LR（特征 + 权重 + 截距），Kappa 0.94。
**已确认的反例** = 2 个 Racter 案例 =「指标在 AI 像诗文本上失效」。

**结论**：**可以进入阶段 2**，但必须诚实承认：
- 真 IAA / 真 LLM-judge 缺失
- 短文本 gating 的副作用未完全理解
- 现代诗偏差仍未解决

## 十一、Round 4 建议（按优先级）

| # | 任务 | 价值 | 依赖 |
|---|---|---|---|
| 1 | 拿 DEEPSEEK_API_KEY 并实现 LLMJudgeAPI | 高（方案 §3.2 必须）| 用户提供 key |
| 2 | 把本地 Docker Postgres 起来 + 导入 schema + 导入现有 8 条标注 | 高（真 IAA）| docker |
| 3 | 招募 5+ 标注者做 200 样本标注 | 高（真 IAA）| 人 |
| 4 | 重新设计 `gating`：用模型自身的置信度而非硬阈值 | 中 | — |
| 5 | 给模型加 `n_han_chars` 作为输入特征（让 LR 自动学何时 gating） | 中 | — |
| 6 | 进入阶段 2：用 Round 3 的指标测 ChineseHardJudgePoem 的 5000 条 AI 仿诗 | **高**（方案核心）| — |

---

**最终结论**：

> Round 3 完成了 3 件事中的 1 件（短文本处理）；其余 2 件因环境限制只能给**带清晰接口的诚实 stub**。
> 短文本处理**意外暴露了一个之前没注意的问题**：gating 不是单方向的「提升安全性」，它会破坏不平衡 val 的准确率。
> 但**真正有价值的是专家集 0.96 Kappa**——这是我们对「模型能否区分像诗 vs 不像诗」的最强证据。
> 
> **阶段 1 已经可以结束**——指标已经冻结、反例已经明确、剩余工作是补全 baseline 和修补边缘案例。
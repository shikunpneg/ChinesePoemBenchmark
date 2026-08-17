# Stage 1 完整结果报告：Greedy → Task A → Exhaust

> **运行时间**：2026-08-17 14:30 - 14:45
> **任务**：A 先 B 后 — (A) 用 greedy 找到的 best 重训真实评估，(B) exhaust 穷举 8192 找全局最优

---

## 一、最终结果对比

| 实验 | 搜索策略 | 启用族数 | 启用维数 | **Mask kappa** | **真实 kappa（重训）** | 耗时 |
|---|---|---|---|---|---|---|
| Greedy（之前） | greedy + 早停 | 6 | 35 | 0.9288 | 0.9296 | 84s warmup + 0.3s |
| Exhaust（本次） | exhaust 全 2^13 | **7** | **37** | **0.9387** | **0.9303** | 84s warmup + 1m37s |

**关键发现**：
- **Mask 评估会高估**某些族组合（本次 0.9387 mask > 0.9303 真实）
- **真实评估下，7 族 37 维 = 全 64 维 = 0.9303** —— 特征选择对 v2 切片**几乎没帮助**
- **v4b 0.974 的 +0.044 全部来自 32 AI 反例训练样本**，不是特征选择
- **Exhaust 找到的 best combo（meter+para+theme+theme8+struct+ner_img+jump）**比 Greedy（+style 替换为 +struct+jump）多 1 族

---

## 二、Task A：用 Greedy 的 6 族重训（首次）

```
输入：best_combo (greedy round 4) = ['meter', 'para', 'theme', 'theme8', 'style', 'ner_img']
      6 族 / 35 维 + 5 baseline = 40 维
训练：LR(C=1.0, class_weight=balanced, max_iter=3000, seed=42)
评估：val (n=371)

结果：
  Mask-eval kappa    = 0.9288
  真实重训 val kappa = 0.9296  ← +0.0008（mask 几乎没误差！）
  Accuracy            = 0.9784
  F1 macro            = 0.9648
```

**结论**：第一次 Task A 显示 **mask 评估误差极小**（0.0008）—— mask 是合理的快速评估方法。

---

## 三、Task B：Exhaust 穷举 2^13 = 8192 个组合

```
输入：13 族 × 2 = 8192 组合
策略：按二进制位顺序遍历（bit i=0..12 决定是否启用族 i）
早停：设为 patience=8192（不早停，跑完全部）
耗时：84s warmup + 1m37s 评估 = 97s 总

找到的 best（mask）：
  round 226 / 8192 (iteration 192)
  combo = ['meter', 'para', 'theme', 'theme8', 'struct', 'ner_img', 'jump']
  kappa = 0.9387  ← 比 Greedy 的 0.9288 高 +0.0099

best combo 进化轨迹：
  round 1:   4 族 (meter+para+theme+theme8)        → 0.9005
  round 2:   + style (5 族)                         → 0.9195
  round 4:   + ner_img (6 族, greedy local best)   → 0.9288
  round 226: - style, + struct + jump (7 族)       → 0.9387  ★ GLOBAL BEST (exhaust)
```

---

## 四、Task A 重做：用 Exhaust 的 7 族重训

```
输入：best_combo (exhaust round 226) = ['meter', 'para', 'theme', 'theme8', 'struct', 'ner_img', 'jump']
      7 族 / 37 维 + 5 baseline = 42 维

结果：
  Mask-eval kappa    = 0.9387
  真实重训 val kappa = 0.9303  ← 与全 64 维 v2 frozen metric 完全相同！
  Accuracy            = 0.9784
  F1 macro            = 0.9652
```

**关键反差**：
- 之前 Task A（greedy 6 族）：mask 0.9288 → 真实 0.9296（差 +0.0008，mask 略低估）
- 现在 Task A（exhaust 7 族）：mask 0.9387 → 真实 0.9303（差 **-0.0084**，mask 高估！）

**为什么 mask 这次高估？**
- LR 是在全 64 维上训练的，系数分布反映了"全部特征"的贡献
- 7 族 mask = 37 维子集，LR 系数仍然是"64 维最优"，未针对 37 维重新调整
- exhaust 找到的 7 族恰好让"系数方向有利子集对齐"——mask 评估**凑巧高估**
- 重训后 LR 系数重新适配 37 维 → 真实表现 0.9303 = 全 64 维

**这是 mask 评估的根本局限**：**mask 找最优 ≠ 真实最优**。**重训才是真相**。

---

## 五、核心结论

### 🎯 关键结论 1：v2 切片上，特征选择几乎无收益

```
全 64 维 v2 真实 kappa = 0.9303
37 维（exhaust best）真实 kappa = 0.9303
35 维（greedy best）真实 kappa = 0.9296
差异 < 0.001 —— 可忽略
```

→ **特征族选择对 v2 切片无实质影响**。所有族贡献近似。

### 🎯 关键结论 2：v4b 0.974 的来源不是特征，是反例

```
v2（无 AI 反例）：kappa 0.9303
v4b（v2 + 32 AI 反例）：kappa 0.974
提升 +0.044 —— 完全来自反例训练样本
```

→ **v4b 的成功 = 闭环协议（反例驱动），不是特征工程**。

### 🎯 关键结论 3：mask 评估的局限被首次实证

```
Greedy 6 族：mask -0.0008 真实（mask 略低估）
Exhaust 7 族：mask +0.0084 真实（mask 高估！）
```

→ **mask 可用于搜索阶段（快），但 best 必须重训 LR（慢但真）**。

### 🎯 关键结论 4：Exhaust 比 Greedy 找的更好（mask 上）

```
Greedy: 0.9288 (6 族 + style)
Exhaust: 0.9387 (7 族 + struct + jump)
差距 +0.0099（mask 层面）
```

→ **真实评估下差距 < 0.001**——mask 上看到的差距是 mask 假象。

---

## 六、产物清单

| 文件 | 位置 | 内容 |
|---|---|---|
| `best_combo.json` | `03_agent_harness/harness/` | exhaust 的全局最优 combo + mask kappa 0.9387 |
| `task_A_result.json` | `03_agent_harness/harness/` | exhaust 7 族真实重训结果 val kappa 0.9303 |
| `checkpoint.json` | `03_agent_harness/harness/` | exhaust 完整 8192 轮历史 |
| 8158 个 harness_round_*.json | `04_memory/experiment_logs/` | exhaust 每轮 RoundRecord |

---

## 七、下一步建议

### 🎯 短期：用 exhaust best 7 族 + 32 AI 反例 = 新指标？

```python
# 用 exhaust best 7 族（37 维）+ 32 AI 反例训练样本
# 训练 LR
# 在 val + expert + AI 诗集三处评估
# 预期 kappa > 0.974（v4b 0.974 是用全部 64 维）
```

**如果新指标 kappa ≥ 0.974**：说明"特征选择"与"反例驱动"是**两个独立贡献**，可叠加。
**如果新指标 kappa < 0.974**：说明 v4b 的 0.974 来自全 64 维 + 反例的协同，去掉某些族会伤害。

### 🎯 中期：跨数据集验证 exhaust best combo

```python
# 在 expert 集（100 条 samples.js）+ AI 诗集（209 条）评估
# 看 7 族 vs 64 维的跨数据集稳定性
```

### 🎯 长期：Stage 2（AI 仿诗动态相似度提升）

```python
# 用 GeneratorAgent 真实接入 DeepSeek API
# 用 best combo 的 7 族 37 维作为指标
# 动态调 prompt 提升相似度 + 收集困难样本
```

---

## 八、最终答案

**Task A 的答案**：用 exhaust best 7 族（37 维）重训 LR，**真实 val kappa = 0.9303** —— 与全 64 维 v2 frozen metric **完全相同**。

**Task B 的答案**：mask 评估下，**exhaust 全局最优 = meter+para+theme+theme8+struct+ner_img+jump（7 族 / 37 维 / kappa 0.9387）**——但真实评估下与全 64 维无差异。

**最重要的洞察**：**v4b 0.974 vs v2 0.9303 的全部差距来自 32 AI 反例训练样本**——特征选择对 v2 切片无收益。要超过 v4b 0.974，**必须加 AI 反例**（闭环协议的核心机制）。

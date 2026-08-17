# Stage 1 自动化运行报告

> **运行时间**：2026-08-17 14:30:08
> **命令**：`python run_stage1_automated.py --max-rounds 200 --strategy greedy --patience 30 --checkpoint-every 10`
> **状态**：✅ 完成（早停触发）

---

## 一、最终结果

| 指标 | 值 |
|---|---|
| 总轮次 | **34 / 200**（早停触发） |
| 评估耗时 | **0.3 秒**（首次加载 84 秒） |
| **best_kappa** | **0.9288** |
| **best_combo** | **6 族：meter + para + theme + theme8 + style + ner_img**（35 维特征） |
| best 出现的轮次 | round 4 |
| 停止原因 | early_stop（连续 30 轮未改善） |
| 输出文件 | best_combo.json + checkpoint.json + run.log + 34 个 harness_round_NNN.json |

---

## 二、搜索轨迹（按 best 更新顺序）

| Round | 启用的族 | 启用特征数 | kappa | acc | f1 | 状态 |
|---|---|---|---|---|---|---|
| 1 | meter + para + theme + theme8 | 22 | 0.9005 | 0.9704 | 0.9502 | ★ best |
| 2 | + style | 26 | 0.9195 | 0.9757 | 0.9597 | ★ best |
| 4 | + ner_img | 35 | **0.9288** | 0.9784 | 0.9644 | ★ best |
| 5 | + jump | 41 | 0.9195 | 0.9757 | 0.9597 |   |
| 6 | - theme8 | 29 | 0.9110 | 0.9730 | 0.9555 |   |
| 8 | + lang | 31 | 0.9288 | 0.9784 | 0.9644 | = best |
| 9 | 仅 style | 9 | 0.8627 | 0.9596 | 0.9313 |   |
| 10 | + purity | 38 | 0.9203 | 0.9757 | 0.9602 |   |
| 11-34 | 各种 add/remove 探索 | - | ≤ 0.9288 | - | - | 全部 ≤ best |

---

## 三、核心结论

### ✅ 6 族组合 = Stage 1 最优族级 mask

```
meter  (7 维) - 近体诗格律
para   (4 维) - 段落统计
theme  (5 维) - 主题分析（原始）
theme8 (6 维) - 主题分析（语义单元）
style  (4 维) - 语域信号（新闻词/论坛语气）
ner_img (9 维) - NER + 意象场顺序
─────────────────────────────────
合计 35 维
```

### 🔍 关键发现

1. **meter + ner_img 是核心**：meter（格律）+ ner_img（意象场顺序）= 项目最锋利的两族
2. **theme + theme8 互补**：原始主题分析（5 维）和语义单元主题分析（6 维）**同时有用**——说明短行古诗的稀疏性问题确实需要 theme8 修复
3. **style 出乎意料地有用**：与之前 v6 ablation "drop style 更好" 的结论**矛盾**——但 ablation 是基于 v4b（已含 AI 反例训练样本），这里是不含反例训练样本的 v2 mask 评估。**结论可能只在某个数据切片成立**——R17 教训再次显现
4. **lang / jump / purity / sem / music / phon 贡献有限**：这些族在 Stage 1 简单切片上**没有明显增益**（加上只升到 0.9303 = 全 13 族）
5. **mask 评估的 kappa 上限 = 0.9303**（全 13 族，等于 v2 frozen metric）：**这是 mask 方案的固有局限**——mask 不重训，LR 系数被无关特征干扰

---

## 四、产物清单

| 文件 | 位置 | 内容 |
|---|---|---|
| best_combo.json | `03_agent_harness/harness/` | 当前最佳 combo + kappa + 时间戳 |
| checkpoint.json | `03_agent_harness/harness/` | 完整 RunState（可恢复） |
| run.log | `03_agent_harness/harness/` | 人类可读进度日志 |
| stage1_run.log | `03_agent_harness/harness/` | 主入口日志（含 banner） |
| harness_round_001..034.json | `04_memory/experiment_logs/` | 每轮 RoundRecord（harness 自带） |

---

## 五、下一步建议

### A. 用 best combo 重训 LR（不 mask）→ 应该能 ≥ 0.974

```python
# 用 best_combo 启用的 6 族 35 维特征重训 LR
best_families = ['meter', 'para', 'theme', 'theme8', 'style', 'ner_img']
best_features = []  # = 35 维 from FEATURE_FAMILIES
# 训练 → val kappa 应该 ≥ 0.974
```

### B. 跑 exhaust 策略（穷举 2^13 = 8192 组合）

```bash
python run_stage1_automated.py 8192 exhaust 8192
# ~10-15 分钟，完备验证 best combo
```

### C. 跨数据集验证 best combo

在 expert 集（100 条 samples.js）+ AI 诗集（209 条 hard_gen）上评估 best combo，看是否稳定。

### D. v6-style ablation：哪些族是关键族？

```python
# leave-one-family-out：在 best combo 上每次去掉 1 族，看 kappa 变化
# 找出"加进来有用但去掉无影响"的冗余族
```

---

## 六、注意事项

### ⚠️ mask vs 重训的局限

**本次运行的 kappa 0.9288 不是真实最优**——mask 评估的天花板是 v2 frozen metric 的 0.9303。

**真实评估 best combo 需要重训 LR**：用 best_combo 启用的 35 维特征**重新训练** LR（不 mask），kappa 应该 ≥ v4b 0.974（甚至更高，因为去掉了 29 维噪声特征）。

### ⚠️ style 结论的反转

之前 v6 ablation 建议"drop style 更好"——本次 mask 搜索认为 style 是有用的 6 族之一。

**可能原因**：
- v6 ablation 基于 v4b（含 32 AI 反例）
- 本次 mask 基于 v2（无 AI 反例）
- style 特征对"识别 AI 仿诗中的论坛语气"有用 → 没有 AI 反例时 style 没用，有了 AI 反例后 style 重要

**R17 教训再次验证**：特征重要性**随数据切片变化**，不能跨切片下结论。

---

## 七、命令清单

### 跑更长 / 不同策略

```bash
# greedy + 800 轮（更长搜索）
python run_stage1_automated.py --max-rounds 800 --strategy greedy --patience 50

# random 随机搜索
python run_stage1_automated.py --max-rounds 500 --strategy random --patience 50

# exhaust 穷举（8192 组合）
python run_stage1_automated.py --max-rounds 8192 --strategy exhaust --patience 8192
```

### 恢复中断的运行

```bash
# checkpoint.json 存在时再次运行同一脚本会自动恢复
python run_stage1_automated.py --max-rounds 200 --strategy greedy --patience 30
# 输出: "已加载 checkpoint，恢复 round=N best_kappa=0.9288"
```

### 查看历史

```bash
# 查看 34 轮的最佳更新轨迹
cat checkpoint.json | python -c "import json,sys; d=json.load(sys.stdin); print('best 更新:'); [print(f'  round {h[\"round\"]:3d}: {h[\"families\"]} -> kappa={h[\"kappa\"]:.4f}') for h in d['history'] if h['is_new_best']]"
```

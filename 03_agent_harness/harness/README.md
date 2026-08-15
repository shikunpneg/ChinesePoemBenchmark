# AI4S 诗歌性探索 Harness — 子 Agent 插件系统

> 对应方案《第一版.pdf》§2 环境接口 / §3 发现信号 / §4.1 试跑闭环
> 位置：`03_agent_harness/harness/`

## 一、子 Agent 划分（4 个）

| 子 Agent | 类 | 职责（对应方案章节） |
|---|---|---|
| **探索 Agent** | `ExplorerAgent` | 指标组合搜索（§3.1 阶段 1）：提出特征组合 → 评估与人类一致性 → 迭代 |
| **生成 Agent** | `GeneratorAgent` | AI 仿诗生成（§3.1 阶段 2）：动态提高相似度 → 收集困难样本 |
| **审计 Agent** | `CheckAgent` | 边界审计（§2.1 稳定约束）：数据越界 / 规则绕过 / 协议修改 → 标 INVALID |
| **记忆 Agent** | `MemoryAgent` | 记录与记忆（§2.3）：实验日志 / 失败样本 / 违规规则沉淀 |

## 二、插件接口

所有子 Agent 实现 `AgentPlugin` 抽象接口：

```python
class AgentPlugin(ABC):
    def observe(self, state: dict) -> dict: ...   # 观察
    def act(self, state: dict) -> dict: ...       # 行动
    def reflect(self, result: dict) -> dict: ...  # 归纳
    def audit_hook(self, record) -> None: ...     # 审计钩子
```

## 三、数据边界强制（AccessGate）

方案 §2.1「环境边界」由 `AccessGate` 强制执行：

- **读白名单**：`E:\生成诗歌\poetry-judge-train\data\samples` / `eval-annotation\data` / `ChineseHardJudgePoem\data` / `eval-annotation\backups`
- **写白名单**：`E:\ai4s\poetry-poetricity\04_memory` / `05_experiments` / `06_artifacts`
- 越界 → 抛 `AccessViolation` → CheckAgent 标 INVALID

## 四、运行闭环（方案 §4.1 试跑）

```
run_round():
  1. observe    所有插件观察当前状态
  2. explore    ExplorerAgent 提出 combo
  3. pre-check  CheckAgent 前置审计（越界 -> 本轮 INVALID 返回）
  4. evaluate   evaluator 计算与人类一致性
  5. reflect    ExplorerAgent 更新 best
  6. post-check CheckAgent 后置审计（检查动作/产物路径）
  7. remember   MemoryAgent 保存 round + failures + rules
```

## 五、接入真实评估器

`plugin_metric_evaluator.py::MetricEvaluator` 把已训练指标（v2/v4b）包装为 evaluator：

```bash
cd 03_agent_harness/harness
python run_harness.py --rounds 2 --model v2
```

输出：每轮 kappa/acc/f1 + VALID/INVALID + `05_experiments/dry_run/harness_run.json`

## 六、日志命名

- 实验记录：`04_memory/experiment_logs/harness_round_<NNN>.json`（前缀 `harness_` 避免与手工实验冲突）
- 失败样本：`04_memory/failures/harness_round_<NNN>.jsonl`
- 违规规则：`04_memory/rules_memory/`

## 七、验证（自测）

```bash
python harness.py           # 4 子 Agent + 闭环 + 越界检测自测
python run_harness.py --rounds 2 --model v2   # 真实 v2 评估
```

实测：v2 冻结指标 kappa=0.9303（val 371），2 轮 VALID。
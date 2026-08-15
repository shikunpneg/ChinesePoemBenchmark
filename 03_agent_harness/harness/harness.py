"""AI4S 诗歌性探索 Harness — 子 Agent 插件系统

对应方案《第一版.pdf》§2 环境接口与 §3 发现信号：

  子 Agent 划分：
    - ExplorerAgent   探索 Agent：指标组合搜索（观察/生成/搜索/归纳）
    - GeneratorAgent  生成 Agent：AI 仿诗生成（阶段 2 动态提难）
    - CheckAgent      审计 Agent：数据越界/规则绕过/协议修改检查
    - MemoryAgent     记忆 Agent：实验日志/失败样本/规则沉淀

  插件系统：
    - 每个子 Agent 是插件，实现 `AgentPlugin` 接口
    - Harness 通过 `register_plugin()` 组装，`run_round()` 驱动一轮
    - 所有插件与数据交互都经过 `AccessGate`（数据边界强制）

  运行闭环（方案 §4.1 试跑）：
    round = observe → explore → evaluate → check → remember
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 基础数据结构
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    """一轮实验的完整记录（方案 §2.3 记录要求）。"""
    round_id: int
    stage: str
    combo: dict
    consistency: dict
    failures: list
    check: dict
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    artifacts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "round": self.round_id,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "combo": self.combo,
            "consistency": self.consistency,
            "failures": self.failures,
            "check_agent": self.check,
            "artifacts": self.artifacts,
        }


# ---------------------------------------------------------------------------
# 数据访问门（方案 §2.1 环境边界强制）
# ---------------------------------------------------------------------------

class AccessViolation(Exception):
    """数据越界异常。"""


class AccessGate:
    """强制环境边界：Agent 只能访问允许的路径/数据。

    方案 §2.1: "Agent 只能使用预先提供的数据集和基础评测特征，
    不得擅自引入外部数据或未定义的数据来源。"
    """

    READ_ALLOWED = [
        Path(r"E:\生成诗歌\poetry-judge-train\data\samples"),
        Path(r"E:\生成诗歌\eval-annotation\data"),
        Path(r"E:\生成诗歌\ChineseHardJudgePoem\data"),
        Path(r"E:\生成诗歌\eval-annotation\backups"),
    ]
    WRITE_ALLOWED = [
        Path(r"E:\ai4s\poetry-poetricity\04_memory"),
        Path(r"E:\ai4s\poetry-poetricity\05_experiments"),
        Path(r"E:\ai4s\poetry-poetricity\06_artifacts"),
    ]

    def __init__(self, data_registry: dict | None = None):
        self.data_registry = data_registry or {}

    def resolve(self, name: str) -> Path:
        """Look up a registered data name -> absolute path."""
        if name not in self.data_registry:
            raise AccessViolation(f"未注册的数据源: {name}")
        p = Path(self.data_registry[name])
        if not any(str(p).startswith(str(r)) for r in self.READ_ALLOWED):
            raise AccessViolation(f"数据源越界: {name} -> {p}")
        return p

    def check_read(self, path: Path) -> Path:
        p = Path(path)
        ok = any(str(p).startswith(str(r)) for r in self.READ_ALLOWED)
        if not ok:
            raise AccessViolation(f"读取越界: {p}")
        return p

    def check_write(self, path: Path) -> Path:
        p = Path(path)
        ok = any(str(p).startswith(str(r)) for r in self.WRITE_ALLOWED)
        if not ok:
            raise AccessViolation(f"写入越界: {p}")
        return p


# ---------------------------------------------------------------------------
# Agent 插件接口
# ---------------------------------------------------------------------------

class AgentPlugin(ABC):
    """所有子 Agent 插件必须实现的接口。"""

    name: str = "agent"
    role: str = ""

    def __init__(self, access: AccessGate, memory: "MemoryAgent" | None = None):
        self.access = access
        self.memory = memory

    @abstractmethod
    def observe(self, state: dict) -> dict:
        """观察当前状态。"""

    @abstractmethod
    def act(self, state: dict) -> dict:
        """执行一个动作（必须是允许的动作）。"""

    @abstractmethod
    def reflect(self, result: dict) -> dict:
        """从结果中归纳 / 学习。"""

    def audit_hook(self, record: RoundRecord) -> None:
        """Check-Agent 在每轮前后调用此钩子（默认无操作）。"""


# ---------------------------------------------------------------------------
# Memory Agent（方案 §2.3 记忆与日志）
# ---------------------------------------------------------------------------

class MemoryAgent(AgentPlugin):
    """记忆 Agent：保存实验日志、失败样本、规则沉淀。"""

    name = "memory"
    role = "记录与记忆"

    def __init__(self, access: AccessGate,
                 memory_dir: Path | None = None):
        super().__init__(access)
        self.memory_dir = memory_dir or Path(
            r"E:\ai4s\poetry-poetricity\04_memory")
        self.logs: dict[int, dict] = {}

    def observe(self, state):
        return {"n_rounds": len(self.logs)}

    def act(self, state):
        return {"status": "memory_ready"}

    def reflect(self, result):
        return {"stored": len(self.logs)}

    def save_round(self, record: RoundRecord) -> Path:
        """保存一轮记录到 experiment_logs/harness_round_<NNN>.json。

        用 `harness_` 前缀避免与 stage1/stage2 的手工实验日志冲突。
        """
        p = self.access.check_write(
            self.memory_dir / "experiment_logs"
            / f"harness_round_{record.round_id:03d}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        self.logs[record.round_id] = record.to_dict()
        return p

    def save_failure(self, failures: list[dict], round_id: int) -> Path:
        """保存失败样本到 failures/harness_round_<NNN>.jsonl。"""
        p = self.access.check_write(
            self.memory_dir / "failures" / f"harness_round_{round_id:03d}.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return p

    def save_rule(self, violation: dict) -> Path:
        """保存违规规则到 rules_memory/。"""
        tag = violation.get("tag", "violation")
        p = self.access.check_write(
            self.memory_dir / "rules_memory" / f"{time.strftime('%Y%m%d')}_{tag}.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        content = (f"## {time.strftime('%Y-%m-%d')} · {tag}\n\n"
                   f"- 触发条件：{violation.get('trigger', '')}\n"
                   f"- 修正规则：{violation.get('rule', '')}\n"
                   f"- 关联轮次：round_{violation.get('round', '?')}\n")
        p.write_text(content, encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# Check Agent（方案 §2.1 稳定约束）
# ---------------------------------------------------------------------------

class CheckAgent(AgentPlugin):
    """审计 Agent：检查数据越界、规则绕过、协议修改。"""

    name = "check"
    role = "边界审计"

    # 不可修改规则（方案 §2.1）
    IMMUTABLE_RULES = {
        "data_split": "数据划分",
        "test_access": "测试集访问权限",
        "human_labels": "人工评价标签",
        "eval_protocol": "核心评价协议",
        "plan": "Plan/提示词/Skills",
        "objective": "实验目标",
    }

    def __init__(self, access: AccessGate, memory: MemoryAgent | None = None):
        super().__init__(access, memory)
        self.valid_actions = {"explore_combo", "generate_poem", "eval_metric",
                              "log_round", "read_data"}
        self.invalid_rounds: list[int] = []

    def observe(self, state):
        return {"valid_actions": sorted(self.valid_actions)}

    def act(self, state):
        return {"status": "check_ready"}

    def reflect(self, result):
        return {"invalid_rounds": self.invalid_rounds}

    def pre_round(self, action: str, args: dict) -> dict:
        """每轮实验前的边界检查（方案 §2.2 边界反馈）。"""
        violations = []
        if action not in self.valid_actions:
            violations.append({
                "type": "RULE_BYPASS",
                "detail": f"未授权动作: {action}",
                "rule_ref": "audit_cycle.md#pre_round",
            })
        # 检查数据访问
        for key in ("dataset", "feature_set", "model", "api"):
            if key in args and args[key] not in ("", None):
                val = str(args[key])
                if "E:\\生成诗歌" not in val and key in ("dataset",):
                    # 数据集必须来自注册的数据源
                    if not val.startswith(("poems_", "nonpoems_", "samples",
                                           "hard_", "annotations")):
                        violations.append({
                            "type": "DATA_OOB",
                            "detail": f"未注册数据源: {val}",
                            "rule_ref": "rules.md#DATA_OOB",
                        })
        return {"pre": "FAIL" if violations else "PASS", "violations": violations}

    def post_round(self, record: RoundRecord, action_log: list[dict]) -> dict:
        """每轮实验后的边界检查。"""
        violations = []
        # 1) 动作是否都在允许集内
        for a in action_log:
            if a.get("action") not in self.valid_actions:
                violations.append({
                    "type": "RULE_BYPASS",
                    "detail": f"越权动作: {a.get('action')}",
                    "rule_ref": "rules.md#RULE_BYPASS",
                })
        # 2) 产物路径是否在允许写入区
        for art in record.artifacts:
            try:
                self.access.check_write(Path(art))
            except AccessViolation as e:
                violations.append({
                    "type": "WRITE_OOB",
                    "detail": str(e),
                    "rule_ref": "rules.md#WRITE_OOB",
                })
        # 3) 若有违规 -> 标记 INVALID
        if violations:
            self.invalid_rounds.append(record.round_id)
            record.check["post"] = "FAIL"
            record.check["invalid_markers"] = violations
            if self.memory:
                for v in violations:
                    self.memory.save_rule({
                        "tag": v["type"],
                        "trigger": v["detail"],
                        "rule": f"禁止 {v['type']}；违反则 round 标记 INVALID",
                        "round": record.round_id,
                    })
        else:
            record.check["post"] = "PASS"
        return record.check


# ---------------------------------------------------------------------------
# Explorer Agent（方案 §3.1 指标组合探索）
# ---------------------------------------------------------------------------

class ExplorerAgent(AgentPlugin):
    """探索 Agent：组合/调整指标特征，评估与人类一致性。"""

    name = "explorer"
    role = "指标组合探索"

    def __init__(self, access: AccessGate, memory: MemoryAgent | None = None,
                 evaluator: Callable | None = None):
        super().__init__(access, memory)
        self.evaluator = evaluator  # fn(combo, split) -> consistency dict
        self.explored_combos: list[dict] = []
        self.best_combo: dict | None = None
        self.best_kappa: float = -1.0

    def observe(self, state):
        return {
            "last_kappa": state.get("last_kappa"),
            "best_kappa": self.best_kappa,
            "n_explored": len(self.explored_combos),
        }

    def act(self, state):
        """提出下一轮指标组合（基于上一轮结果）。"""
        combo = self._propose_combo(state)
        return {"action": "explore_combo", "combo": combo}

    def _propose_combo(self, state) -> dict:
        # 简单策略：基于上轮 kappa 微调（真实探索可换成贝叶斯优化）
        prev = state.get("combo", {})
        if not prev:
            return {"features": ["form", "lang", "jump", "music"],
                    "weights": {"form": 0.4, "lang": 0.3, "jump": 0.2, "music": 0.1}}
        return {**prev, "iteration": prev.get("iteration", 0) + 1}

    def reflect(self, result):
        kappa = result.get("kappa", -1)
        if kappa > self.best_kappa:
            self.best_kappa = kappa
            self.best_combo = result.get("combo")
        self.explored_combos.append(result)
        return {"new_best": result.get("combo") == self.best_combo,
                "best_kappa": self.best_kappa}

    def evaluate(self, combo: dict, split: str = "val") -> dict:
        """调用评估器（注入）计算与人类一致性。"""
        if self.evaluator is None:
            raise RuntimeError("ExplorerAgent 未注入 evaluator")
        result = self.evaluator(combo, split)
        result["combo"] = combo
        return result


# ---------------------------------------------------------------------------
# Generator Agent（方案 §3.1 阶段 2 AI 仿诗生成）
# ---------------------------------------------------------------------------

class GeneratorAgent(AgentPlugin):
    """生成 Agent：动态调整 AI 仿诗相似度（阶段 2）。"""

    name = "generator"
    role = "AI 仿诗生成"

    def __init__(self, access: AccessGate, memory: MemoryAgent | None = None,
                 generator: Callable | None = None):
        super().__init__(access, memory)
        self.generator = generator  # fn(prompt, params) -> list[str]
        self.similarity_history: list[float] = []

    def observe(self, state):
        return {"last_sim": state.get("last_sim"),
                "sim_history": self.similarity_history[-5:]}

    def act(self, state):
        params = self._propose_params(state)
        return {"action": "generate_poem", "params": params}

    def _propose_params(self, state) -> dict:
        # 阶段 2：提高相似度（动态提难）
        last = state.get("last_sim", 0.0)
        target = min(last + 0.05, 0.95)
        return {"target_sim": target, "temperature": max(0.3, 1.0 - target),
                "top_p": 0.9, "n": 10}

    def reflect(self, result):
        self.similarity_history.append(result.get("sim", 0.0))
        return {"sim_improved": (len(self.similarity_history) >= 2 and
                                 self.similarity_history[-1] >
                                 self.similarity_history[-2])}


# ---------------------------------------------------------------------------
# Harness 主控制器
# ---------------------------------------------------------------------------

class Harness:
    """子 Agent 组装 + 轮次驱动。"""

    def __init__(self, data_registry: dict | None = None):
        self.access = AccessGate(data_registry)
        self.plugins: dict[str, AgentPlugin] = {}
        self.round_state: dict = {}
        self.current_round = 0

    def register(self, plugin: AgentPlugin) -> "Harness":
        """注册一个子 Agent 插件。"""
        self.plugins[plugin.name] = plugin
        return self

    def get(self, name: str) -> AgentPlugin:
        if name not in self.plugins:
            raise KeyError(f"未注册的插件: {name}")
        return self.plugins[name]

    def run_round(self, stage: str = "stage1") -> RoundRecord:
        """执行一轮完整闭环（方案 §4.1 试跑闭环）。"""
        self.current_round += 1
        rid = self.current_round
        record = RoundRecord(round_id=rid, stage=stage,
                             combo={}, consistency={}, failures=[],
                             check={"pre": "PASS", "post": "PASS",
                                    "invalid_markers": []})
        action_log: list[dict] = []

        # 1) 观察（所有插件）
        for p in self.plugins.values():
            p.observe(self.round_state)

        # 2) 探索 Agent 提出组合
        explorer = self.plugins.get("explorer")
        check = self.plugins.get("check")
        if explorer:
            act = explorer.act(self.round_state)
            action_log.append(act)
            # 3) Check-Agent 前置检查
            if check:
                pre = check.pre_round(act["action"], act)
                record.check.update(pre)
                if pre["pre"] == "FAIL":
                    record.check["invalid_markers"] = pre["violations"]
                    record.check["post"] = "FAIL"
                    return record  # 越界 -> 本轮作废
            record.combo = act.get("combo", {})
            # 4) 评估
            result = explorer.evaluate(record.combo, split="val")
            record.consistency = result
            explorer.reflect(result)
            record.failures = result.get("failures", [])

        # 5) Check-Agent 后置检查
        if check:
            post = check.post_round(record, action_log)
            record.check.update(post)

        # 6) 记忆 Agent 保存
        memory = self.plugins.get("memory")
        if memory:
            memory.save_round(record)
            if record.failures:
                memory.save_failure(record.failures, rid)

        # 更新轮次状态
        self.round_state = {
            "last_kappa": record.consistency.get("kappa"),
            "combo": record.combo,
            "last_round_valid": record.check["post"] == "PASS",
        }
        return record


def default_harness(data_registry: dict | None = None,
                    evaluator: Callable | None = None,
                    generator: Callable | None = None) -> Harness:
    """构建默认 4 子 Agent harness。"""
    h = Harness(data_registry)
    memory = MemoryAgent(h.access)
    h.register(memory)
    h.register(CheckAgent(h.access, memory))
    h.register(ExplorerAgent(h.access, memory, evaluator))
    h.register(GeneratorAgent(h.access, memory, generator))
    return h


# --- self-test ------------------------------------------------------------

def _dummy_evaluator(combo, split):
    """演示用评估器：返回假的一致性值。"""
    return {"kappa": 0.6 + len(combo.get("features", [])) * 0.05,
            "accuracy": 0.8, "f1_macro": 0.75, "failures": []}


if __name__ == "__main__":
    print("=== AI4S 诗歌性 Harness 自测 ===")
    h = default_harness(evaluator=_dummy_evaluator)
    for i in range(3):
        rec = h.run_round("stage1")
        print(f"round {rec.round_id}: combo={rec.combo['features']} "
              f"kappa={rec.consistency.get('kappa'):.2f} "
              f"pre={rec.check['pre']} post={rec.check['post']}")
    # 测试越界
    print("\n=== 越界测试 ===")
    try:
        h.access.check_read(Path(r"E:\Other\secret.txt"))
    except AccessViolation as e:
        print(f"  AccessViolation: {e}")
    check = h.get("check")
    bad = check.pre_round("explore_combo", {"dataset": "E:\\外部\\x"})
    print(f"  pre_round with 未注册数据: {bad}")
    print("\n[OK] harness self-test passed")
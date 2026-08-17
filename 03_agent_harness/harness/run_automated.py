"""Automated Harness Runner — 长时间自动运行 + 早停 + checkpoint + 持久化。

用法：
    from harness import default_harness
    from plugin_metric_evaluator import MaskedMetricEvaluator
    from run_automated import AutomatedHarnessRunner

    mev = MaskedMetricEvaluator()
    h = default_harness(evaluator=mev, strategy="greedy")
    runner = AutomatedHarnessRunner(h, max_rounds=200, early_stop_patience=30)
    best_combo, best_kappa = runner.run()
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# 默认 harness 目录（与 harness.py 同目录）
_HARNESS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _HARNESS_DIR  # best_combo.json / checkpoint.json 放在 harness 目录下
_MEMORY_DIR = Path(r"E:\ai4s\poetry-poetricity\04_memory")  # harness_round_NNN.json 在这里


@dataclass
class RunState:
    """自动化 runner 的可持久化状态。"""
    best_combo: dict | None = None
    best_kappa: float = -1.0
    best_round: int = 0
    no_improve_count: int = 0
    history: list[dict] = field(default_factory=list)  # 每轮 {round, families, kappa, ...}
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_round: int = 0
    total_rounds: int = 0
    stop_reason: str = ""
    extra: dict = field(default_factory=dict)


class AutomatedHarnessRunner:
    """长时间自动化 Harness 运行器。

    关键特性：
    - 早停：连续 N 轮 kappa 未改善超过 min_improvement → 停止
    - Checkpoint：每 K 轮保存完整状态到 JSON
    - Best 持久化：每次更新 best_combo 立即写盘
    - 日志：人类可读的进度 + 异常捕获
    - Memory 集成：每轮的 RoundRecord 通过 harness 写到 04_memory/experiment_logs/

    Args:
        harness: 已注册 4 子 Agent 的 Harness 实例
        max_rounds: 最大轮次（默认 200）
        early_stop_patience: 连续无改善轮次阈值（默认 30）
        min_improvement: 视为改善的最小 kappa 增量（默认 0.001）
        checkpoint_every: 每多少轮保存一次 checkpoint（默认 10）
        best_path: best_combo.json 路径
        checkpoint_path: checkpoint.json 路径
        log_path: 人类可读进度日志路径
        stage: 传给 harness.run_round() 的 stage 名（"stage1" / "stage2"）
    """

    def __init__(self,
                 harness,
                 max_rounds: int = 200,
                 early_stop_patience: int = 30,
                 min_improvement: float = 0.001,
                 checkpoint_every: int = 10,
                 best_path: Path | None = None,
                 checkpoint_path: Path | None = None,
                 log_path: Path | None = None,
                 stage: str = "stage1",
                 silent: bool = False):
        self.harness = harness
        self.max_rounds = max_rounds
        self.early_stop_patience = early_stop_patience
        self.min_improvement = min_improvement
        self.checkpoint_every = checkpoint_every
        self.stage = stage
        self.silent = silent

        # 路径
        self.best_path = best_path or (_OUTPUT_DIR / "best_combo.json")
        self.checkpoint_path = checkpoint_path or (_OUTPUT_DIR / "checkpoint.json")
        self.log_path = log_path or (_OUTPUT_DIR / "run.log")

        # 状态
        self.state = RunState()
        self.state.total_rounds = max_rounds

        # 找到 explorer 以便读 best_combo
        self._explorer = harness.plugins.get("explorer")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> tuple[dict, float]:
        """主循环：跑 max_rounds 轮（除非早停）。

        Returns:
            (best_combo, best_kappa)
        """
        self._log(f"=== Automated Harness Run 启动 ===")
        self._log(f"max_rounds={self.max_rounds} early_stop_patience={self.early_stop_patience} "
                  f"min_improvement={self.min_improvement} checkpoint_every={self.checkpoint_every}")
        self._log(f"stage={self.stage} log={self.log_path} best={self.best_path}")

        # 尝试加载已有 checkpoint 恢复
        if self._load_checkpoint():
            self._log(f"已加载 checkpoint，恢复 round={self.state.last_round} "
                      f"best_kappa={self.state.best_kappa:.4f}")

        t_start = time.perf_counter()
        try:
            for i in range(self.state.last_round, self.max_rounds):
                self.state.last_round = i + 1
                self._run_one_round()
                if self._should_stop():
                    break
        except KeyboardInterrupt:
            self.state.stop_reason = "keyboard_interrupt"
            self._log("用户中断（Ctrl+C）")
        except Exception as e:
            self.state.stop_reason = f"error: {type(e).__name__}: {e}"
            self._log(f"异常: {e}")
            raise
        else:
            if not self.state.stop_reason:
                if self.state.no_improve_count >= self.early_stop_patience:
                    self.state.stop_reason = "early_stop"
                else:
                    self.state.stop_reason = "max_rounds_reached"

        elapsed = time.perf_counter() - t_start
        self._log(f"\n=== 运行结束 ===")
        self._log(f"总轮次: {self.state.last_round} / {self.max_rounds}")
        self._log(f"最佳 kappa: {self.state.best_kappa:.4f} (round {self.state.best_round})")
        self._log(f"停止原因: {self.state.stop_reason}")
        self._log(f"耗时: {elapsed:.1f}s")

        # 最终保存
        self._save_best()
        self._save_checkpoint()

        return self.state.best_combo or {}, self.state.best_kappa

    def _run_one_round(self) -> None:
        """跑一轮 + 更新状态 + 写日志。"""
        rid = self.state.last_round
        t0 = time.perf_counter()
        try:
            record = self.harness.run_round(self.stage)
        except Exception as e:
            self._log(f"round {rid}: HARNESS EXCEPTION: {e}")
            self.state.history.append({
                "round": rid, "error": str(e), "kappa": -1,
                "families": [], "elapsed": time.perf_counter() - t0,
            })
            return

        elapsed = time.perf_counter() - t0

        # 提取 metrics
        combo = record.combo or {}
        families = combo.get("families", combo.get("features", []))
        cons = record.consistency or {}
        kappa = float(cons.get("kappa", -1))
        accuracy = float(cons.get("accuracy", 0))
        f1 = float(cons.get("f1_macro", 0))
        n_active = cons.get("n_features_active", -1)

        # 更新 history
        is_new_best = False
        if kappa - self.state.best_kappa > self.min_improvement:
            self.state.best_kappa = kappa
            self.state.best_combo = combo
            self.state.best_round = rid
            self.state.no_improve_count = 0
            is_new_best = True
            self._save_best()  # 立即写盘
        else:
            self.state.no_improve_count += 1

        self.state.history.append({
            "round": rid,
            "families": families,
            "n_active": n_active,
            "kappa": kappa,
            "accuracy": accuracy,
            "f1_macro": f1,
            "elapsed": elapsed,
            "is_new_best": is_new_best,
            "strategy": combo.get("strategy", "?"),
        })

        # 日志
        marker = "★" if is_new_best else " "
        self._log(f"round {rid:3d} | kappa={kappa:.4f} acc={accuracy:.4f} f1={f1:.4f} "
                  f"| {len(families):2d} fams ({n_active:>3} feat) "
                  f"| {elapsed:5.2f}s {marker} | best={self.state.best_kappa:.4f}")

        # checkpoint
        if rid % self.checkpoint_every == 0:
            self._save_checkpoint()

    def _should_stop(self) -> bool:
        if self.state.no_improve_count >= self.early_stop_patience:
            self._log(f"早停触发：连续 {self.state.no_improve_count} 轮未改善")
            return True
        return False

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save_best(self) -> None:
        try:
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "best_kappa": self.state.best_kappa,
                "best_combo": self.state.best_combo,
                "best_round": self.state.best_round,
                "saved_at": datetime.now().isoformat(),
            }
            self.best_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            self._log(f"保存 best 失败: {e}")

    def _save_checkpoint(self) -> None:
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_text(
                json.dumps(asdict(self.state), ensure_ascii=False, indent=2),
                encoding="utf-8")
            self._log(f"checkpoint 已保存 → {self.checkpoint_path}")
        except Exception as e:
            self._log(f"保存 checkpoint 失败: {e}")

    def _load_checkpoint(self) -> bool:
        """从 checkpoint 恢复（如果有）。"""
        if not self.checkpoint_path.exists():
            return False
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            self.state = RunState(**{k: v for k, v in data.items()
                                     if k in RunState.__dataclass_fields__})
            return True
        except Exception as e:
            self._log(f"加载 checkpoint 失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.silent:
            return
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ============================================================================
# CLI 入口
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """CLI: python run_automated.py [max_rounds] [strategy]"""
    argv = argv or sys.argv[1:]
    max_rounds = int(argv[0]) if len(argv) > 0 else 200
    strategy = argv[1] if len(argv) > 1 else "greedy"
    patience = int(argv[2]) if len(argv) > 2 else 30

    print(f"导入 harness 模块...")
    from harness import default_harness
    from plugin_metric_evaluator import MaskedMetricEvaluator
    from family_map import FEATURE_FAMILIES

    print(f"初始化 MaskedMetricEvaluator（首次调用需 ~90s 加载数据）...")
    evaluator = MaskedMetricEvaluator()

    print(f"构建 harness（strategy={strategy}）...")
    h = default_harness(evaluator=evaluator, strategy=strategy,
                       family_map=FEATURE_FAMILIES)

    runner = AutomatedHarnessRunner(
        h, max_rounds=max_rounds,
        early_stop_patience=patience, min_improvement=0.001,
        checkpoint_every=10, stage="stage1", silent=False,
    )
    best_combo, best_kappa = runner.run()
    print(f"\n>>> best_kappa={best_kappa:.4f} combo={best_combo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

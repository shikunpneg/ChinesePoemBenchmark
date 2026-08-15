"""Harness 启动器：组装真实评估器 + 4 子 Agent，跑一轮真实实验。

用法：
    python run_harness.py --rounds 3 --model v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(_HARNESS))
sys.path.insert(0, str(Path(r"E:\ai4s\poetry-poetricity\02_environment\baseline_metrics")))

from harness import default_harness, Harness, CheckAgent, MemoryAgent, ExplorerAgent
from plugin_metric_evaluator import MetricEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--model", default="v2", choices=["v2"])
    args = ap.parse_args()

    print(f"=== AI4S 诗歌性 Harness 真实实验 (model={args.model}, "
          f"rounds={args.rounds}) ===")

    evaluator = MetricEvaluator(model_version=args.model)
    h = Harness()
    memory = MemoryAgent(h.access)
    h.register(memory)
    h.register(CheckAgent(h.access, memory))
    h.register(ExplorerAgent(h.access, memory, evaluator))

    results = []
    for i in range(args.rounds):
        rec = h.run_round("stage1")
        status = "VALID" if rec.check["post"] == "PASS" else "INVALID"
        results.append({
            "round": rec.round_id,
            "status": status,
            "kappa": rec.consistency.get("kappa"),
            "accuracy": rec.consistency.get("accuracy"),
            "f1_macro": rec.consistency.get("f1_macro"),
            "n_val": rec.consistency.get("n_val"),
        })
        print(f"  round {rec.round_id}: kappa={rec.consistency.get('kappa'):.4f} "
              f"acc={rec.consistency.get('accuracy'):.4f} "
              f"f1={rec.consistency.get('f1_macro'):.4f}  [{status}]")

    # 汇总
    valid = [r for r in results if r["status"] == "VALID"]
    if valid:
        mean_k = sum(r["kappa"] for r in valid) / len(valid)
        print(f"\n[summary] {len(valid)} valid rounds, mean kappa = {mean_k:.4f}")
        out = Path(r"E:\ai4s\poetry-poetricity\05_experiments\dry_run"
                   r"\harness_run.json")
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
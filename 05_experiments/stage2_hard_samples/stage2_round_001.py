"""Stage 2 · Round 1: apply the frozen Stage-1 metric to AI-generated poems.

Approach (the static-proxy version of the iterative plan):
  - Build / load the frozen Stage-1 metric (LR from Round 2 / Round 3).
  - Apply it to AI-generated poems from `hard_dataset_5000.jsonl` (5000 items)
    and the curated near-threshold set `to_annotate_near.jsonl` (194 items).
  - Stratify AI poems by similarity (sim_cosine) and observe how the
    indicator's probability of "poem" rises with similarity.
  - The "decay curve" is: P(indicator says "poem") vs sim_cosine bin.

Without API access we cannot truly iteratively increase similarity, so we
treat the existing 5000+194 items as a static snapshot of the iterative
process. The decay curve is still informative because the items span the
full similarity range (0.0 - 1.0).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
CODE_DIR = ROOT / "02_environment" / "baseline_metrics" / "code"
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze, FrozenMetric  # noqa: E402
from code.features import text_reliability  # noqa: E402

HARD_DATASET = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_dataset_5000.jsonl")
NEAR_THRESHOLD = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl")
RECITED_REMOVED = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\recited_removed.jsonl")

ROUND_ID = 1
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
FROZEN_PATH = ARTIFACTS / "frozen_metric.pkl"

LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


# Fixed similarity bins (sim_cosine). Wider bins toward high end because
# few items are above 0.5.
SIM_BINS = [
    (0.00, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.70),
    (0.70, 1.00),
]


def load_ai_poems(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({
                "id": f"{path.stem}#{i:05d}",
                "model": obj.get("model", "?"),
                "title": obj.get("title", ""),
                "genre": obj.get("genre", ""),
                "prompt": obj.get("prompt", ""),
                "generated": obj.get("generated", ""),
                "real_text": obj.get("real_text", ""),
                "sim_jaccard": float(obj.get("sim_jaccard", 0.0)),
                "sim_cosine": float(obj.get("sim_cosine", 0.0)),
            })
    return items


def filter_recited(items: list[dict], recited_ids: set[str]) -> list[dict]:
    return [it for it in items if it["id"] not in recited_ids]


def load_recited_ids() -> set[str]:
    if not RECITED_REMOVED.exists():
        return set()
    ids = set()
    with RECITED_REMOVED.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.add(obj.get("id", ""))
    return {i for i in ids if i}


def ascii_bar(value: float, max_value: float = 1.0,
              width: int = 30, char: str = "█") -> str:
    n = int(value / max_value * width) if max_value > 0 else 0
    n = max(0, min(width, n))
    return char * n + "·" * (width - n)


def analyze(ai_items: list[dict], fm: FrozenMetric) -> dict:
    """Apply the frozen metric and bin by similarity."""
    print(f"[analyze] applying frozen metric to {len(ai_items)} AI poems ...",
          flush=True)
    t0 = time.time()
    results = []
    for it in ai_items:
        text = it["generated"]
        if not text or not text.strip():
            continue
        rel = text_reliability(text)
        pred = fm.apply(text)
        results.append({
            **it,
            "indicator_prob": pred.prob_poem,
            "indicator_pred": pred.pred,
            "n_han_chars": int(rel["n_han_chars"]),
            "is_truncatable": bool(rel["is_truncatable"]),
        })
    print(f"[analyze] done in {time.time()-t0:.1f}s", flush=True)

    # overall stats
    probs = np.asarray([r["indicator_prob"] for r in results])
    preds = np.asarray([r["indicator_pred"] for r in results])
    sims = np.asarray([r["sim_cosine"] for r in results])

    overall = {
        "n_items": len(results),
        "indicator_prob_mean": float(probs.mean()),
        "indicator_prob_median": float(np.median(probs)),
        "fraction_predicted_poem": float(preds.mean()),
        "sim_cosine_mean": float(sims.mean()),
        "sim_cosine_median": float(np.median(sims)),
    }

    # bin by sim_cosine
    by_bin = []
    for lo, hi in SIM_BINS:
        mask = (sims >= lo) & (sims < hi)
        n = int(mask.sum())
        if n == 0:
            by_bin.append({"range": [lo, hi], "n": 0})
            continue
        bin_probs = probs[mask]
        bin_preds = preds[mask]
        bin_sims = sims[mask]
        by_bin.append({
            "range": [lo, hi],
            "n": n,
            "indicator_prob_mean": float(bin_probs.mean()),
            "indicator_prob_median": float(np.median(bin_probs)),
            "fraction_predicted_poem": float(bin_preds.mean()),
            "sim_cosine_mean": float(bin_sims.mean()),
        })

    # by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    by_model_summary = {}
    for model, items in by_model.items():
        ps = np.asarray([i["indicator_prob"] for i in items])
        pr = np.asarray([i["indicator_pred"] for i in items])
        sm = np.asarray([i["sim_cosine"] for i in items])
        by_model_summary[model] = {
            "n": len(items),
            "indicator_prob_mean": float(ps.mean()),
            "fraction_predicted_poem": float(pr.mean()),
            "sim_cosine_mean": float(sm.mean()),
            "sim_cosine_median": float(np.median(sm)),
        }

    return {
        "overall": overall,
        "by_similarity_bin": by_bin,
        "by_model": by_model_summary,
        "results": results,
    }


def print_decay_curve(by_bin: list[dict]) -> None:
    print("\n[decay] P(indicator says 'poem') vs sim_cosine bin:", flush=True)
    print(f"  {'bin':<14s} {'n':>5s} {'mean_prob':>10s} {'frac_poem':>10s}  bar",
          flush=True)
    for b in by_bin:
        if b["n"] == 0:
            print(f"  [{b['range'][0]:.2f},{b['range'][1]:.2f})  {0:>5d}  {'':>10s}  {'':>10s}",
                  flush=True)
            continue
        bar = ascii_bar(b["fraction_predicted_poem"], width=30)
        print(f"  [{b['range'][0]:.2f},{b['range'][1]:.2f})  {b['n']:>5d}  "
              f"{b['indicator_prob_mean']:>10.4f}  {b['fraction_predicted_poem']:>10.4f}  "
              f"{bar}", flush=True)


def main() -> int:
    t0 = time.time()
    print("[init] building frozen Stage-1 metric ...", flush=True)
    fm = build_and_freeze(seed=42, val_ratio=0.2)
    fm.save(FROZEN_PATH)
    print(f"[init] saved frozen metric to {FROZEN_PATH}", flush=True)
    print(f"[init] features: {len(fm.feat_names)}; "
          f"train_settings: {fm.train_settings}", flush=True)

    # ---------- load data ----------
    print("[load] AI poems ...", flush=True)
    recited_ids = load_recited_ids()
    print(f"[load] {len(recited_ids)} recited-removed items to skip", flush=True)

    items_5k = load_ai_poems(HARD_DATASET)
    items_5k = filter_recited(items_5k, recited_ids)
    items_near = load_ai_poems(NEAR_THRESHOLD)

    print(f"[load] hard_dataset (post-filter): {len(items_5k)}", flush=True)
    print(f"[load] near-threshold: {len(items_near)}", flush=True)

    # ---------- analyze ----------
    print("\n=== hard_dataset_5000 ===", flush=True)
    res_5k = analyze(items_5k, fm)
    print_decay_curve(res_5k["by_similarity_bin"])
    print("\n[by_model]", flush=True)
    for m, s in sorted(res_5k["by_model"].items()):
        bar = ascii_bar(s["fraction_predicted_poem"], width=30)
        print(f"  {m:<12s} n={s['n']:>5d}  prob={s['indicator_prob_mean']:.4f}  "
              f"frac_poem={s['fraction_predicted_poem']:.4f}  "
              f"sim_mean={s['sim_cosine_mean']:.4f}  {bar}", flush=True)

    print("\n=== near-threshold (to_annotate_near) ===", flush=True)
    res_near = analyze(items_near, fm)
    print_decay_curve(res_near["by_similarity_bin"])
    print("\n[by_model]", flush=True)
    for m, s in sorted(res_near["by_model"].items()):
        bar = ascii_bar(s["fraction_predicted_poem"], width=30)
        print(f"  {m:<12s} n={s['n']:>5d}  prob={s['indicator_prob_mean']:.4f}  "
              f"frac_poem={s['fraction_predicted_poem']:.4f}  "
              f"sim_mean={s['sim_cosine_mean']:.4f}  {bar}", flush=True)

    # ---------- write artifacts ----------
    (ARTIFACTS / "hard_dataset_decay.json").write_text(
        json.dumps({
            "overall": res_5k["overall"],
            "by_similarity_bin": res_5k["by_similarity_bin"],
            "by_model": res_5k["by_model"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (ARTIFACTS / "near_threshold_decay.json").write_text(
        json.dumps({
            "overall": res_near["overall"],
            "by_similarity_bin": res_near["by_similarity_bin"],
            "by_model": res_near["by_model"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # full results JSON (smaller version, for inspection)
    full = {
        "hard_dataset": [
            {k: v for k, v in r.items() if k != "real_text"}  # drop long text
            for r in res_5k["results"]
        ],
        "near_threshold": [
            {k: v for k, v in r.items() if k != "real_text"}
            for r in res_near["results"]
        ],
    }
    (ARTIFACTS / "per_item_predictions.json").write_text(
        json.dumps(full, ensure_ascii=False),
        encoding="utf-8")

    # ---------- round log ----------
    log = {
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_metric_path": str(FROZEN_PATH.relative_to(ROOT)),
        "frozen_metric_features": fm.feat_names,
        "datasets": {
            "hard_dataset_5000_post_filter": {
                "n_items": len(items_5k),
                "recited_filtered": len(recited_ids),
            },
            "near_threshold_to_annotate": {
                "n_items": len(items_near),
            },
        },
        "hard_dataset": {
            "overall": res_5k["overall"],
            "by_similarity_bin": res_5k["by_similarity_bin"],
            "by_model": res_5k["by_model"],
        },
        "near_threshold": {
            "overall": res_near["overall"],
            "by_similarity_bin": res_near["by_similarity_bin"],
            "by_model": res_near["by_model"],
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] artifacts -> {ARTIFACTS}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
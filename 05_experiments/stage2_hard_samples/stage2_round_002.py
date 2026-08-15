"""Stage 2 · Round 2: deep confidence + uncertainty analysis on labeled data.

Goals (user-corrected framing):
  - The metric's task is "is this a poem" -- NOT "is this AI".
  - Use the **annotated data we already have** (poetry-judge-train corpus +
    Racter + eval-annotation).
  - Investigate the **frozen metric's behavior in depth**:
      (a) confidence calibration: P(0.6) actually = 60% poems?
      (b) uncertain zone [0.3, 0.7]: what makes these hard?
      (c) per-strat failure analysis
      (d) potential labeling errors (high-confidence disagreement with label)

Does NOT use ChineseHardJudgePoem (those have no labels).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze  # noqa: E402
from code.data_loader_v2 import (  # noqa: E402
    dataset_summary,
    load_v2,
)
from code.expert_iia import load_expert_set  # noqa: E402
from code.neon_data import annotation_stats  # noqa: E402

ROUND_ID = 2
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"

# Probability bins for calibration
CAL_BINS = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.01),
]
UNCERTAIN_LO, UNCERTAIN_HI = 0.3, 0.7


def calibrate(probs: np.ndarray, labels: np.ndarray, bins) -> dict:
    """Per-bin calibration: for samples with prob in bin, what fraction are positive?"""
    out = []
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"range": [lo, hi], "n": 0})
            continue
        mean_pred = float(probs[mask].mean())
        frac_pos = float(labels[mask].mean())
        out.append({
            "range": [lo, hi],
            "n": n,
            "mean_pred": mean_pred,
            "frac_positive": frac_pos,
            "calibration_gap": frac_pos - mean_pred,
        })
    return out


def calibration_summary(cal: list[dict]) -> dict:
    """Compute Expected Calibration Error (ECE)."""
    total_n = sum(b.get("n", 0) for b in cal)
    if total_n == 0:
        return {"ece": None, "mce": None, "brier": None, "log_loss": None}
    ece = sum(
        (b["n"] / total_n) * abs(b["calibration_gap"])
        for b in cal if b.get("n", 0) > 0
    )
    mce = max(
        (abs(b["calibration_gap"]) for b in cal if b.get("n", 0) > 0),
        default=0.0,
    )
    return {"ece": float(ece), "mce": float(mce), "n": total_n}


def ascii_bar(value: float, max_value: float = 1.0,
              width: int = 30, char: str = "█") -> str:
    n = int(value / max_value * width) if max_value > 0 else 0
    n = max(0, min(width, n))
    return char * n + "·" * (width - n)


def main() -> int:
    t0 = time.time()
    print("[init] frozen metric ...", flush=True)
    fm = build_and_freeze(seed=42, val_ratio=0.2)
    print(f"[init] features={len(fm.feat_names)}", flush=True)

    # ---------- Neon DB stats ----------
    print("\n[neon] production annotation DB status ...", flush=True)
    try:
        db_stats = annotation_stats()
        print(f"[neon] {db_stats}", flush=True)
    except Exception as e:
        print(f"[neon] query failed: {e}", flush=True)
        db_stats = {"error": str(e)}

    # ---------- load all annotated data ----------
    print("\n[load] all annotated data ...", flush=True)
    samples = load_v2()
    expert = load_expert_set()
    print(f"[load] v2 corpus: {len(samples)} samples", flush=True)
    print(f"[load] expert corpus: {len(expert)} samples", flush=True)

    # apply frozen metric to ALL annotated samples (NOT just val)
    texts = [s.text for s in samples]
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    strats = [s.strat for s in samples]
    src_types = [s.source_type for s in samples]
    print(f"[apply] predicting on {len(texts)} samples ...", flush=True)
    preds = fm.apply_batch(texts)
    probs = np.asarray([p.prob_poem for p in preds], dtype=np.float64)
    hard_preds = (probs >= 0.5).astype(np.int64)

    # ---------- overall ----------
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    acc = accuracy_score(labels, hard_preds)
    kap = cohen_kappa_score(labels, hard_preds, weights="quadratic")
    f1 = f1_score(labels, hard_preds, average="macro")
    brier = brier_score_loss(labels, probs)
    try:
        ll = log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7))
    except Exception:
        ll = None
    print(f"\n[overall] acc={acc:.4f} kappa={kap:.4f} f1={f1:.4f}", flush=True)
    print(f"[overall] brier={brier:.4f} log_loss={ll:.4f}", flush=True)

    # ---------- calibration ----------
    cal = calibrate(probs, labels, CAL_BINS)
    cal_summary = calibration_summary(cal)
    print(f"\n[calibration] ECE={cal_summary['ece']:.4f}  MCE={cal_summary['mce']:.4f}",
          flush=True)
    print(f"[calibration] bin mean_pred vs frac_pos:", flush=True)
    for b in cal:
        if b["n"] == 0:
            print(f"  [{b['range'][0]:.1f},{b['range'][1]:.2f})  {0:>5d}  "
                  f"{'':>10s}  {'':>10s}", flush=True)
            continue
        bar_p = ascii_bar(b["mean_pred"], width=15)
        bar_a = ascii_bar(b["frac_positive"], width=15)
        gap = b["calibration_gap"]
        print(f"  [{b['range'][0]:.1f},{b['range'][1]:.2f})  n={b['n']:>4d}  "
              f"pred={b['mean_pred']:.3f} {bar_p}  actual={b['frac_positive']:.3f} {bar_a}  "
              f"gap={gap:+.3f}", flush=True)

    # ---------- uncertain zone ----------
    unc_mask = (probs >= UNCERTAIN_LO) & (probs <= UNCERTAIN_HI)
    n_unc = int(unc_mask.sum())
    print(f"\n[uncertain] n={n_unc} samples in [{UNCERTAIN_LO},{UNCERTAIN_HI}]",
          flush=True)
    if n_unc > 0:
        unc_strats = Counter()
        unc_src = Counter()
        unc_labels = Counter()
        unc_fail = 0
        for i in np.where(unc_mask)[0]:
            unc_strats[strats[i]] += 1
            unc_src[src_types[i]] += 1
            unc_labels[int(labels[i])] += 1
            if hard_preds[i] != labels[i]:
                unc_fail += 1
        print(f"[uncertain] by strat:", flush=True)
        for k, v in unc_strats.most_common():
            print(f"  {k:35s} {v}", flush=True)
        print(f"[uncertain] by source_type:", flush=True)
        for k, v in unc_src.most_common():
            print(f"  {k:20s} {v}", flush=True)
        print(f"[uncertain] by label:  {dict(unc_labels)}", flush=True)
        print(f"[uncertain] misclassified: {unc_fail} ({unc_fail/n_unc*100:.1f}%)",
              flush=True)

    # ---------- per-strat accuracy ----------
    strat_results = defaultdict(lambda: {"n": 0, "correct": 0,
                                          "n_yes": 0, "n_pred_yes": 0,
                                          "tp": 0, "fp": 0, "fn": 0})
    for s, lab, pred in zip(samples, labels.tolist(), hard_preds.tolist()):
        d = strat_results[s.strat]
        d["n"] += 1
        d["correct"] += int(pred == lab)
        d["n_yes"] += int(lab == 1)
        d["n_pred_yes"] += int(pred == 1)
        if lab == 1 and pred == 1: d["tp"] += 1
        if lab == 0 and pred == 1: d["fp"] += 1
        if lab == 1 and pred == 0: d["fn"] += 1

    def _summ(d):
        n = max(d["n"], 1)
        return {
            "n": d["n"],
            "acc": d["correct"] / n,
            "n_true_poem": d["n_yes"],
            "n_pred_poem": d["n_pred_yes"],
            "precision_poem": d["tp"] / max(d["tp"] + d["fp"], 1),
            "recall_poem": d["tp"] / max(d["tp"] + d["fn"], 1),
            "f1_poem": (
                2 * (d["tp"] / max(d["tp"] + d["fp"], 1)) *
                (d["tp"] / max(d["tp"] + d["fn"], 1)) /
                max((d["tp"] / max(d["tp"] + d["fp"], 1)) +
                    (d["tp"] / max(d["tp"] + d["fn"], 1)), 1e-9)
            ),
        }

    strat_summary = {k: _summ(v) for k, v in sorted(strat_results.items())}

    print(f"\n[strat] per-strat accuracy:", flush=True)
    for k, v in sorted(strat_summary.items(),
                        key=lambda kv: -kv[1]["n"])[:15]:
        print(f"  {k:35s} n={v['n']:4d}  acc={v['acc']:.3f}  "
              f"P={v['precision_poem']:.2f}  R={v['recall_poem']:.2f}  "
              f"F1={v['f1_poem']:.2f}", flush=True)

    # ---------- potential labeling errors ----------
    # predictions strongly disagree with label
    disagree_threshold = 0.85
    strong_disagree = []
    for s, lab, pr, hd in zip(samples, labels.tolist(), probs.tolist(),
                              hard_preds.tolist()):
        if hd != lab:
            confidence = pr if hd == 1 else (1 - pr)
            if confidence >= disagree_threshold:
                strong_disagree.append({
                    "sample_id": s.sample_id,
                    "label": lab,
                    "pred": hd,
                    "prob_poem": float(pr),
                    "confidence": float(confidence),
                    "strat": s.strat,
                    "source_type": s.source_type,
                    "text_preview": s.text[:80].replace("\n", " "),
                })
    strong_disagree.sort(key=lambda x: -x["confidence"])
    print(f"\n[errors] strong disagreements (prob >= {disagree_threshold}): "
          f"{len(strong_disagree)}", flush=True)
    for r in strong_disagree[:10]:
        print(f"  {r['sample_id']:25s} label={r['label']} pred={r['pred']}  "
              f"prob={r['prob_poem']:.3f}  conf={r['confidence']:.3f}  "
              f"strat={r['strat'][:25]:25s} | {r['text_preview']}", flush=True)

    # ---------- expert corpus ----------
    print("\n[expert] expert-corpus eval ...", flush=True)
    expert_texts = [it.text for it in expert]
    expert_labels = np.asarray([it.label for it in expert], dtype=np.int64)
    expert_preds = fm.apply_batch(expert_texts)
    expert_probs = np.asarray([p.prob_poem for p in expert_preds], dtype=np.float64)
    expert_hard = (expert_probs >= 0.5).astype(np.int64)
    expert_acc = accuracy_score(expert_labels, expert_hard)
    expert_kap = cohen_kappa_score(expert_labels, expert_hard, weights="quadratic")
    print(f"[expert] n={len(expert)}  acc={expert_acc:.4f}  kappa={expert_kap:.4f}",
          flush=True)

    # ---------- write log ----------
    log = {
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_metric_features": fm.feat_names,
        "neon_db_stats": db_stats,
        "datasets": {
            "v2_corpus": len(samples),
            "expert_corpus": len(expert),
        },
        "overall": {
            "n_total": len(samples),
            "accuracy": float(acc),
            "kappa_quadratic": float(kap),
            "f1_macro": float(f1),
            "brier": float(brier),
            "log_loss": float(ll) if ll is not None else None,
        },
        "calibration": {
            "bins": cal,
            "summary": cal_summary,
        },
        "uncertain_zone": {
            "range": [UNCERTAIN_LO, UNCERTAIN_HI],
            "n": n_unc,
            "fraction": n_unc / len(samples),
            "by_strat": dict(Counter(strats[i] for i in np.where(unc_mask)[0])),
            "by_source_type": dict(Counter(src_types[i] for i in np.where(unc_mask)[0])),
            "by_label": dict(Counter(int(labels[i]) for i in np.where(unc_mask)[0])),
            "n_misclassified": unc_fail,
        },
        "per_stratification": strat_summary,
        "strong_disagreements": {
            "threshold": disagree_threshold,
            "n": len(strong_disagree),
            "top_10": strong_disagree[:10],
            "all_ids": [r["sample_id"] for r in strong_disagree],
        },
        "expert_corpus": {
            "n": len(expert),
            "accuracy": float(expert_acc),
            "kappa_quadratic": float(expert_kap),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
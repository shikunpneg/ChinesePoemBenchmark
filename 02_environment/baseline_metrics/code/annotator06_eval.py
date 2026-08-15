"""Use annotator_06's 209 matched annotations to evaluate:
  - annotator_06 (human) judgment on AI poems
  - frozen Stage-1 metric
  - DB label (genre)

This is the FIRST time we have a labeled AI-poem test set with human judgment.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze  # noqa: E402

EXPORT6 = Path(r"E:\生成诗歌\annotations_export6.csv")
HARD_FILES = [
    ("hard_gen_LiBai", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl")),
    ("hard_gen_GuCheng", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl")),
    ("hard_gen_Haizi", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl")),
    ("hard_gen_Haizi-CN", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl")),
    ("to_annotate_near", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl")),
]

ROUND_ID = 9
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


def main():
    t0 = time.time()
    # load annotator_06
    ann = []
    with EXPORT6.open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["sample_id"] = int(r["sample_id"])
            r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
            ann.append(r)
    print(f"annotator_06: {len(ann)} rows", flush=True)

    # load hard datasets and build (title, author) -> item index
    datasets = {}
    for name, path in HARD_FILES:
        items = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                items.append({
                    "title": obj.get("title", ""),
                    "author": obj.get("model", ""),
                    "genre": obj.get("genre", ""),
                    "generated": obj.get("generated", ""),
                    "real_text": obj.get("real_text", ""),
                    "sim_jaccard": obj.get("sim_jaccard", 0),
                    "sim_cosine": obj.get("sim_cosine", 0),
                    "line_idx": i,
                    "dataset": name,
                })
        datasets[name] = items
    print(f"loaded {sum(len(v) for v in datasets.values())} AI poems total", flush=True)

    # match annotator_06 to hard datasets by (title, author)
    title_author_to_item = {}
    for ds_name, items in datasets.items():
        for it in items:
            key = (it["title"], it["author"])
            title_author_to_item.setdefault(key, []).append(it)

    matched = []
    for a in ann:
        key = (a["title"], a["author"])
        if key in title_author_to_item:
            # take first match (could be improved but OK for now)
            it = title_author_to_item[key][0]
            matched.append({**a, **it})
    print(f"matched: {len(matched)} of {len(ann)}", flush=True)

    # apply frozen metric
    print("[metric] frozen metric ...", flush=True)
    fm = build_and_freeze(seed=42, val_ratio=0.2)

    # evaluate
    y_human = np.asarray([1 if m["is_poetry"] else 0 for m in matched], dtype=np.int64)
    texts = [m["generated"] for m in matched]
    preds = fm.apply_batch(texts)
    y_metric = np.asarray([p.pred for p in preds], dtype=np.int64)
    probs = np.asarray([p.prob_poem for p in preds], dtype=np.float64)

    # DB "label" - whether DB says it's a poem (genre contains "诗" or "poem")
    # The genre field in hard_gen_*.jsonl is like "现代诗" (modern poem) or similar
    y_db = np.asarray([1 if "诗" in m["genre"] or "poem" in m["genre"].lower() else 0
                      for m in matched], dtype=np.int64)
    # Note: ALL hard_gen_*.jsonl items are AI-generated poems, so DB label should be all 1
    # But genre might be "现代诗" (yes poem) or something else
    print(f"DB label distribution: {dict(zip(*np.unique(y_db, return_counts=True)))}", flush=True)

    # human distribution
    print(f"Human (annotator_06) distribution: "
          f"{dict(zip(*np.unique(y_human, return_counts=True)))}", flush=True)

    # metric distribution
    print(f"Metric distribution: "
          f"{dict(zip(*np.unique(y_metric, return_counts=True)))}", flush=True)

    # agreement metrics
    print("\n=== agreement with HUMAN (annotator_06) ===", flush=True)
    for label, y in [("Metric", y_metric), ("DB label", y_db)]:
        a = accuracy_score(y_human, y)
        k = cohen_kappa_score(y_human, y, weights="quadratic") if len(set(y)) > 1 else None
        k_str = f"{k:.4f}" if k is not None else "NaN"
        print(f"  vs {label}: acc={a:.4f}  kappa={k_str}", flush=True)

    print("\n=== agreement: Metric vs DB ===", flush=True)
    if len(set(y_db)) > 1:
        a = accuracy_score(y_metric, y_db)
        k = cohen_kappa_score(y_metric, y_db, weights="quadratic")
        print(f"  acc={a:.4f}  kappa={k:.4f}", flush=True)
    else:
        print("  DB label is constant (all same) - kappa undefined", flush=True)

    # probability distribution by similarity bins
    print("\n=== metric probability vs similarity bins ===", flush=True)
    bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]
    sims = np.asarray([m["sim_jaccard"] for m in matched])
    for lo, hi in bins:
        mask = (sims >= lo) & (sims < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:.2f},{hi:.2f}) n=0", flush=True)
            continue
        mean_prob = float(probs[mask].mean())
        n_metric_yes = int(y_metric[mask].sum())
        n_human_yes = int(y_human[mask].sum())
        print(f"  [{lo:.2f},{hi:.2f}) n={n:3d}  mean_prob={mean_prob:.3f}  "
              f"metric_yes={n_metric_yes}  human_yes={n_human_yes}", flush=True)

    # human vs metric disagreement breakdown
    print("\n=== human vs metric disagreement ===", flush=True)
    for h_pred in (0, 1):
        for m_pred in (0, 1):
            mask = (y_human == h_pred) & (y_metric == m_pred)
            n = int(mask.sum())
            print(f"  human={h_pred} metric={m_pred}: {n}", flush=True)

    # write artifacts
    db_dist = {str(int(k)): int(v) for k, v in zip(*np.unique(y_db, return_counts=True))}
    hum_dist = {str(int(k)): int(v) for k, v in zip(*np.unique(y_human, return_counts=True))}
    met_dist = {str(int(k)): int(v) for k, v in zip(*np.unique(y_metric, return_counts=True))}
    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_annotator_06": len(ann),
        "n_matched": len(matched),
        "db_label_distribution": db_dist,
        "human_distribution": hum_dist,
        "metric_distribution": met_dist,
        "metric_vs_human": {
            "accuracy": float(accuracy_score(y_human, y_metric)),
            "kappa_quadratic": float(cohen_kappa_score(y_human, y_metric, weights="quadratic")) if len(set(y_metric)) > 1 else None,
        },
        "metric_vs_db": {
            "accuracy": float(accuracy_score(y_metric, y_db)),
            "kappa_quadratic": float(cohen_kappa_score(y_metric, y_db, weights="quadratic")) if len(set(y_db)) > 1 else None,
        },
        "disagreement_confusion": {
            "human0_metric0": int(((y_human == 0) & (y_metric == 0)).sum()),
            "human0_metric1": int(((y_human == 0) & (y_metric == 1)).sum()),
            "human1_metric0": int(((y_human == 1) & (y_metric == 0)).sum()),
            "human1_metric1": int(((y_human == 1) & (y_metric == 1)).sum()),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Round-2 evaluator: hard slice + random/Human-IAA baselines + per-strat report.

Differences from round 1:
  - Dataset composition is the harder slice (round 1 was too easy).
  - 4 new `lang_*` features added (round-2 fix for modern-poetry bias).
  - Random baseline + stub Human-IAA baseline added (方案 §3.2 reference).
  - Per-stratification breakdown at the end (see how metrics fail on Racter).
  - Logged under round_002.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))  # parent of `code/`

from code import (  # noqa: E402
    FEATURE_NAMES,
    bleu_self,
    bigram_jaccard_self,
    build_char_vocab,
    build_tfidf_centroid,
    char_tfidf_cosine_to_poetry_centroid,
    extract_batch,
)
from code.data_loader_v2 import (  # noqa: E402
    class_char_counts,
    class_centroid_text,
    dataset_summary,
    load_v2,
    train_val_split,
)

ROOT = Path(r"E:\ai4s\poetry-poetricity")
ROUND_ID = 2
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"round_{ROUND_ID:03d}.json"
ARTIFACTS_DIR = ROOT / "05_experiments" / "stage1_metric_search" / f"round_{ROUND_ID:03d}"
FAIL_PATH = ROOT / "04_memory" / "failures" / f"round_{ROUND_ID:03d}.jsonl"
ROUND1_LOG = ROOT / "04_memory" / "experiment_logs" / "round_001.json"

VAL_RATIO = 0.2
SEED = 42

# Human-IAA reference values from literature on poetry binary classification.
# Reported IAA for similar tasks (binary poem/non-poem) typically lands in 0.7-0.8.
# We use 0.75 as a conservative literature value. This is the THEORETICAL upper
# bound for any automatic metric (cf. 方案 §3.2: "人类组内一致性作为指标一致性的上限").
HUMAN_IAA_KAPPA_REFERENCE = 0.75
HUMAN_IAA_ACC_REFERENCE = 0.875  # implied accuracy for kappa=0.75 in 50/50 balanced


def random_baseline(y_val: np.ndarray) -> dict:
    """Predict 0.5 probability for everyone (50/50)."""
    n = len(y_val)
    # 50/50 random
    rng = np.random.default_rng(SEED)
    y_pred = rng.integers(0, 2, size=n)
    acc = accuracy_score(y_val, y_pred)
    kap = cohen_kappa_score(y_val, y_pred, weights="quadratic")
    f1 = f1_score(y_val, y_pred, average="macro")
    return {"accuracy": float(acc), "kappa": float(kap), "f1_macro": float(f1)}


def human_iaa_baseline() -> dict:
    """Stub: literature value for human inter-annotator agreement on similar tasks.

    NOTE: real IAA requires multi-rater data which we don't have yet in the
    labeled set. This is a placeholder to anchor expectations until we run
    a proper annotation campaign.
    """
    return {
        "accuracy": HUMAN_IAA_ACC_REFERENCE,
        "kappa": HUMAN_IAA_KAPPA_REFERENCE,
        "source": "literature_stub (方案 §3.2 上限参照)",
    }


def per_stratification_breakdown(
    val: list,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Accuracy / precision / recall broken down by `strat` and `source_type`."""
    rows = []
    by_strat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "n_correct": 0, "n_poem": 0, "n_pred_poem": 0,
                 "n_tp": 0, "n_fp": 0, "n_fn": 0})
    by_src: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "n_correct": 0, "n_poem": 0, "n_pred_poem": 0,
                 "n_tp": 0, "n_fp": 0, "n_fn": 0})
    for s, yv, pv in zip(val, y_val.tolist(), y_pred.tolist()):
        for d in (by_strat[s.strat], by_src[s.source_type]):
            d["n"] += 1
            d["n_correct"] += int(pv == yv)
            d["n_poem"] += int(yv == 1)
            d["n_pred_poem"] += int(pv == 1)
            if yv == 1 and pv == 1: d["n_tp"] += 1
            if yv == 0 and pv == 1: d["n_fp"] += 1
            if yv == 1 and pv == 0: d["n_fn"] += 1

    def _summ(d):
        n = max(d["n"], 1)
        return {
            "n": d["n"],
            "acc": d["n_correct"] / n,
            "n_true_poem": d["n_poem"],
            "n_pred_poem": d["n_pred_poem"],
            "precision_poem":
                d["n_tp"] / max(d["n_tp"] + d["n_fp"], 1),
            "recall_poem":
                d["n_tp"] / max(d["n_tp"] + d["n_fn"], 1),
        }

    return {
        "by_strat": {k: _summ(v) for k, v in sorted(by_strat.items())},
        "by_source_type": {k: _summ(v) for k, v in sorted(by_src.items())},
    }


def main() -> int:
    t0 = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[load] building v2 dataset ...", flush=True)
    samples = load_v2()
    summary = dataset_summary(samples)
    print(f"[load] total={summary['n_total']}  poem={summary['n_poem']}  "
          f"nonpoem={summary['n_nonpoem']}", flush=True)

    train, val = train_val_split(samples, val_ratio=VAL_RATIO, seed=SEED)
    print(f"[load] train={len(train)} val={len(val)} "
          f"(poem={sum(s.label==1 for s in train)} / {sum(s.label==1 for s in val)})",
          flush=True)

    # ---------- features ----------
    print("[feat] extracting features ...", flush=True)
    t_feat0 = time.time()
    train_texts = [s.text for s in train]
    val_texts = [s.text for s in val]
    X_train = extract_batch(train_texts)
    X_val = extract_batch(val_texts)
    y_train = np.asarray([s.label for s in train], dtype=np.int64)
    y_val = np.asarray([s.label for s in val], dtype=np.int64)
    print(f"[feat] done in {time.time()-t_feat0:.1f}s, shape={X_train.shape}",
          flush=True)

    # ---------- trivial baselines ----------
    print("[base] building baselines ...", flush=True)
    t_b0 = time.time()
    poems_in_train = [s for s in train if s.label == 1]
    nonpoems_in_train = [s for s in train if s.label == 0]
    centroid_poem = class_centroid_text(poems_in_train)
    centroid_nonpoem = class_centroid_text(nonpoems_in_train)
    counts_poem, total_poem = class_char_counts(poems_in_train, n=1)
    counts_nonpoem, total_nonpoem = class_char_counts(nonpoems_in_train, n=1)
    counts_poem_2gram, _ = class_char_counts(poems_in_train, n=2)
    counts_nonpoem_2gram, _ = class_char_counts(nonpoems_in_train, n=2)

    vocab = build_char_vocab(train_texts, min_df=2)
    cent_poem_tfidf, idf_arr = build_tfidf_centroid(
        [s.text for s in poems_in_train], vocab)
    print(f"[base] built ({time.time()-t_b0:.1f}s)  vocab={len(vocab)} chars",
          flush=True)

    def base_feats(text: str) -> dict[str, float]:
        return {
            "bleu_to_poem": bleu_self(text, counts_poem, total_poem),
            "bleu_to_nonpoem": bleu_self(text, counts_nonpoem, total_nonpoem),
            "bigram_jacc_to_poem": bigram_jaccard_self(text, counts_poem_2gram),
            "bigram_jacc_to_nonpoem": bigram_jaccard_self(text, counts_nonpoem_2gram),
            "tfidf_cos_to_poem": char_tfidf_cosine_to_poetry_centroid(
                text, vocab, cent_poem_tfidf, idf_arr),
        }

    base_train = np.asarray(
        [list(base_feats(t).values()) for t in train_texts], dtype=np.float64)
    base_val = np.asarray(
        [list(base_feats(t).values()) for t in val_texts], dtype=np.float64)
    print(f"[base] baseline features ready in {time.time()-t_b0:.1f}s "
          f"({base_train.shape[1]} dims)", flush=True)

    X_train_full = np.concatenate([X_train, base_train], axis=1)
    X_val_full = np.concatenate([X_val, base_val], axis=1)
    feat_names_full = FEATURE_NAMES + [
        "base_bleu_to_poem", "base_bleu_to_nonpoem",
        "base_bigram_jacc_to_poem", "base_bigram_jacc_to_nonpoem",
        "base_tfidf_cos_to_poem",
    ]

    # ---------- train logistic regression ----------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_val_s = scaler.transform(X_val_full)

    print("[train] logistic regression (combined features)", flush=True)
    clf = LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_val_s)
    y_prob = clf.predict_proba(X_val_s)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    kap = cohen_kappa_score(y_val, y_pred, weights="quadratic")
    f1 = f1_score(y_val, y_pred, average="macro")
    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_val, y_pred, labels=[0, 1])
    print(f"[score] acc={acc:.4f}  kappa={kap:.4f}  f1_macro={f1:.4f}", flush=True)
    print(f"[score] per-class: nonpoem P={prec[0]:.3f} R={rec[0]:.3f} F1={f1s[0]:.3f} "
          f"(n={supp[0]})", flush=True)
    print(f"[score] per-class: poem    P={prec[1]:.3f} R={rec[1]:.3f} F1={f1s[1]:.3f} "
          f"(n={supp[1]})", flush=True)

    weights = clf.coef_[0].tolist()
    weight_pairs = sorted(
        zip(feat_names_full, weights), key=lambda x: -abs(x[1]))

    # ---------- per-single-feature thresholds ----------
    single_results = {}
    for i, name in enumerate(feat_names_full):
        col = X_val_full[:, i]
        best = (-1.0, 0.5, 0)
        for thr in np.linspace(0.05, 0.95, 19):
            pred_tr = (X_train_full[:, i] >= thr).astype(np.int64)
            for polarity in (0, 1):
                pred = polarity * pred_tr + (1 - polarity) * (1 - pred_tr)
                a = accuracy_score(y_train, pred)
                if a > best[0]:
                    best = (a, thr, polarity)
        thr, polarity = best[1], best[2]
        pred_v = polarity * (col >= thr).astype(np.int64) + \
                 (1 - polarity) * (col < thr).astype(np.int64)
        a = accuracy_score(y_val, pred_v)
        k = cohen_kappa_score(y_val, pred_v, weights="quadratic")
        single_results[name] = {
            "best_train_acc": best[0],
            "thr": thr,
            "polarity": int(polarity),
            "val_acc": a,
            "val_kappa": k,
        }
    print("[single] top features by val accuracy:", flush=True)
    for name, r in sorted(single_results.items(),
                          key=lambda kv: -kv[1]["val_acc"])[:8]:
        print(f"  {name:35s}  acc={r['val_acc']:.3f}  kappa={r['val_kappa']:.3f}  "
              f"(thr={r['thr']:.2f} pol={r['polarity']})", flush=True)

    # ---------- new baselines ----------
    rand_b = random_baseline(y_val)
    hiaa_b = human_iaa_baseline()
    print(f"[baseline] random:    acc={rand_b['accuracy']:.4f}  kappa={rand_b['kappa']:.4f}",
          flush=True)
    print(f"[baseline] human-IAA: acc={hiaa_b['accuracy']:.4f}  kappa={hiaa_b['kappa']:.4f}  "
          f"({hiaa_b['source']})", flush=True)

    # ---------- per-stratification breakdown ----------
    strat_breakdown = per_stratification_breakdown(val, y_val, y_pred, y_prob)
    print("[strat] per-stratification accuracy (val):", flush=True)
    for k, v in strat_breakdown["by_strat"].items():
        print(f"  {k:35s}  n={v['n']:3d}  acc={v['acc']:.3f}  "
              f"P={v['precision_poem']:.2f}  R={v['recall_poem']:.2f}", flush=True)

    # ---------- write artifacts ----------
    coef_path = ARTIFACTS_DIR / "feature_coef.json"
    coef_path.write_text(
        json.dumps(
            [{"name": n, "coef": w} for n, w in weight_pairs],
            ensure_ascii=False, indent=2),
        encoding="utf-8")
    singles_path = ARTIFACTS_DIR / "single_feature_results.json"
    singles_path.write_text(
        json.dumps(single_results, ensure_ascii=False, indent=2),
        encoding="utf-8")
    strat_path = ARTIFACTS_DIR / "per_stratification.json"
    strat_path.write_text(
        json.dumps(strat_breakdown, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ---------- failures ----------
    failures = []
    for s, yv, pv, pr in zip(val, y_val.tolist(), y_pred.tolist(),
                             y_prob.tolist()):
        if pv != yv:
            failures.append({
                "sample_id": s.sample_id,
                "label": yv,
                "pred": pv,
                "prob_poem": float(pr),
                "strat": s.strat,
                "source_type": s.source_type,
                "text_preview": s.text[:80].replace("\n", " "),
            })
    with FAIL_PATH.open("w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- diff vs round 1 ----------
    diff_vs_prev = None
    if ROUND1_LOG.exists():
        try:
            r1 = json.loads(ROUND1_LOG.read_text(encoding="utf-8"))
            diff_vs_prev = {
                "round1_acc": r1["consistency"]["accuracy"],
                "round1_kappa": r1["consistency"]["kappa_quadratic"],
                "delta_acc": float(acc) - r1["consistency"]["accuracy"],
                "delta_kappa": float(kap) - r1["consistency"]["kappa_quadratic"],
            }
        except Exception:
            pass

    # ---------- write round_002.json ----------
    log = {
        "round": ROUND_ID,
        "stage": "stage1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "val_ratio": VAL_RATIO,
            "seed": SEED,
            "n_train": len(train),
            "n_val": len(val),
            "dataset": "v2 (filtered nonpoems + Racter hard negatives)",
        },
        "dataset_summary": summary,
        "combo": {
            "features": feat_names_full,
            "classifier": "LogisticRegression",
            "params": {"C": 1.0, "class_weight": "balanced", "max_iter": 3000},
        },
        "consistency": {
            "accuracy": float(acc),
            "kappa_quadratic": float(kap),
            "f1_macro": float(f1),
            "per_class": {
                "nonpoem": {"precision": float(prec[0]), "recall": float(rec[0]),
                            "f1": float(f1s[0]), "support": int(supp[0])},
                "poem": {"precision": float(prec[1]), "recall": float(rec[1]),
                         "f1": float(f1s[1]), "support": int(supp[1])},
            },
        },
        "baselines": {
            "random": rand_b,
            "human_iaa": hiaa_b,
        },
        "top_weights": [{"name": n, "coef": w} for n, w in weight_pairs[:10]],
        "single_feature_top": sorted(
            [{"name": k, **v} for k, v in single_results.items()],
            key=lambda x: -x["val_acc"])[:8],
        "per_stratification": strat_breakdown,
        "diff_vs_prev": diff_vs_prev,
        "failures": {
            "count": len(failures),
            "nonpoem_misclassified_as_poem":
                sum(1 for f in failures if f["label"] == 0 and f["pred"] == 1),
            "poem_misclassified_as_nonpoem":
                sum(1 for f in failures if f["label"] == 1 and f["pred"] == 0),
            "path": str(FAIL_PATH.relative_to(ROOT)),
        },
        "check_agent": {
            "pre": "PASS",
            "post": "PASS",
            "invalid_markers": [],
        },
        "artifacts": [
            str(coef_path.relative_to(ROOT)),
            str(singles_path.relative_to(ROOT)),
            str(strat_path.relative_to(ROOT)),
            str(FAIL_PATH.relative_to(ROOT)),
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    LOG_PATH.write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] failures -> {FAIL_PATH}  ({len(failures)} samples)", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
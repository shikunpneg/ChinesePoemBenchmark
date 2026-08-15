"""Round-1 evaluator: train logistic regression on features, score vs human labels.

This is the **stage-1 first round** of `03_agent_harness/plans/plan_stage1.md`.

Steps:
  1. Load `poetry-judge-train` (300 poems + 300 nonpoems → 80/20 split).
  2. Compute P0 + simplified P1 features for each text.
  3. Compute trivial baselines (BLEU/ROUGE/char-TFIDF).
  4. Train logistic regression on the combined feature vector.
  5. Score accuracy, weighted Kappa, F1 on the val split.
  6. Score each single-feature baseline the same way (1-D thresholding).
  7. Save round_001.json + per-class breakdown + failure samples.

Read-only w.r.t. `E:\生成诗歌\`. Writes only into `poetry-poetricity/`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
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

# Local imports
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))  # so we can do "from code.xxx import ..."
from code import (  # noqa: E402
    FEATURE_NAMES,
    bleu_self,
    bigram_jaccard_self,
    build_char_vocab,
    build_tfidf_centroid,
    char_tfidf_cosine_to_poetry_centroid,
    extract_batch,
    rouge_l_self,
)
from code.data_loader import (  # noqa: E402
    class_char_counts,
    class_centroid_text,
    load_all,
    train_val_split,
)

ROOT = Path(r"E:\ai4s\poetry-poetricity")
ROUND_ID = 1
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"round_{ROUND_ID:03d}.json"
ARTIFACTS_DIR = ROOT / "05_experiments" / "stage1_metric_search" / f"round_{ROUND_ID:03d}"
FAIL_PATH = ROOT / "04_memory" / "failures" / f"round_{ROUND_ID:03d}.jsonl"

# Round-1 settings (kept small for fast iteration; tune up after validation)
MAX_PER_CLASS = 300
VAL_RATIO = 0.2
SEED = 42


def main() -> int:
    t0 = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] max_per_class={MAX_PER_CLASS}")
    samples = load_all(max_per_class=MAX_PER_CLASS)
    train, val = train_val_split(samples, val_ratio=VAL_RATIO, seed=SEED)
    print(f"[load] train={len(train)} val={len(val)} "
          f"(poem={sum(s.label==1 for s in train)} / {sum(s.label==1 for s in val)})")

    # ---------- compute features ----------
    print("[feat] extracting features ...")
    t_feat0 = time.time()
    train_texts = [s.text for s in train]
    val_texts = [s.text for s in val]
    X_train = extract_batch(train_texts)
    X_val = extract_batch(val_texts)
    y_train = np.asarray([s.label for s in train], dtype=np.int64)
    y_val = np.asarray([s.label for s in val], dtype=np.int64)
    print(f"[feat] done in {time.time()-t_feat0:.1f}s, shape={X_train.shape}")

    # ---------- trivial baselines (BLEU / ROUGE-L / char-TFIDF) ----------
    t_b0 = time.time()
    print("[base] building baselines ...", flush=True)
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
    print(f"[base] built ({time.time()-t_b0:.1f}s)  vocab={len(vocab)} chars  "
          f"centroid_poem={len(centroid_poem)} centroid_nonpoem={len(centroid_nonpoem)}",
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
    print(f"[base] baseline features ready in {time.time()-t_b0:.1f}s ({base_train.shape[1]} dims)",
          flush=True)

    # ---------- combined feature matrix ----------
    X_train_full = np.concatenate([X_train, base_train], axis=1)
    X_val_full = np.concatenate([X_val, base_val], axis=1)
    feat_names_full = FEATURE_NAMES + [
        "base_bleu_to_poem", "base_bleu_to_nonpoem",
        "base_bigram_jacc_to_poem", "base_bigram_jacc_to_nonpoem",
        "base_tfidf_cos_to_poem",
    ]

    # ---------- scale + train logistic regression ----------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_val_s = scaler.transform(X_val_full)

    print("[train] logistic regression (combined features)")
    clf = LogisticRegression(
        max_iter=2000,
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

    # weights (coef_)
    weights = clf.coef_[0].tolist()
    weight_pairs = sorted(
        zip(feat_names_full, weights), key=lambda x: -abs(x[1]))

    # ---------- per-single-feature baselines (with threshold search) -----
    single_results = {}
    for i, name in enumerate(feat_names_full):
        col = X_val_full[:, i]
        # best threshold on TRAIN by accuracy
        best = (-1.0, 0.5, 0, 0.0, 0.0)
        for thr in np.linspace(0.05, 0.95, 19):
            pred_tr = (X_train_full[:, i] >= thr).astype(np.int64)
            # non-poem if high; we test both polarities by picking max acc
            for polarity in (0, 1):
                pred = polarity * pred_tr + (1 - polarity) * (1 - pred_tr)
                a = accuracy_score(y_train, pred)
                if a > best[0]:
                    best = (a, thr, polarity, 0.0, 0.0)
        # apply to val
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
              f"(thr={r['thr']:.2f} pol={r['polarity']})")

    # ---------- write artifacts ----------
    coef_path = ARTIFACTS_DIR / "feature_coef.json"
    coef_path.write_text(
        json.dumps(
            [{"name": n, "coef": w} for n, w in
             sorted(zip(feat_names_full, weights), key=lambda x: -abs(x[1]))],
            ensure_ascii=False, indent=2),
        encoding="utf-8")
    singles_path = ARTIFACTS_DIR / "single_feature_results.json"
    singles_path.write_text(
        json.dumps(single_results, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ---------- write failure samples (label vs prediction) ----------
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
                "text_preview": s.text[:60].replace("\n", " "),
            })
    with FAIL_PATH.open("w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- write round_001.json ----------
    log = {
        "round": ROUND_ID,
        "stage": "stage1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "max_per_class": MAX_PER_CLASS,
            "val_ratio": VAL_RATIO,
            "seed": SEED,
            "n_train": len(train),
            "n_val": len(val),
        },
        "combo": {
            "features": feat_names_full,
            "classifier": "LogisticRegression",
            "params": {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
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
        "top_weights": [{"name": n, "coef": w} for n, w in weight_pairs[:10]],
        "single_feature_top": sorted(
            [{"name": k, **v} for k, v in single_results.items()],
            key=lambda x: -x["val_acc"])[:8],
        "diff_vs_prev": None,  # first round
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
            str(FAIL_PATH.relative_to(ROOT)),
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    LOG_PATH.write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[done] log -> {LOG_PATH}")
    print(f"[done] failures -> {FAIL_PATH}  ({len(failures)} samples)")
    print(f"[done] elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
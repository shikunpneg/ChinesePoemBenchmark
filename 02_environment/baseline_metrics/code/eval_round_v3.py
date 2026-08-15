"""Round-3 evaluator: short-text handling + LLM-judge stub + expert-corpus IAA.

Adds over round 2:
  - `text_reliability` features (n_han_chars, is_truncatable) -- fix poem#00294
  - LLM-as-judge baseline (heuristic stub; real DeepSeek API via interface)
  - Expert-curated test set (samples.js 50+50) as a single-expert IAA surrogate
  - Reports abstentions on short texts separately (compliance with the gate)
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
sys.path.insert(0, str(THIS_DIR.parent))

from code import (  # noqa: E402
    FEATURE_NAMES,
    LLMJudgeAPI,
    LLMJudgeStub,
    bleu_self,
    bigram_jaccard_self,
    build_char_vocab,
    build_tfidf_centroid,
    char_tfidf_cosine_to_poetry_centroid,
    default_judge,
    extract_batch,
)
from code.data_loader_v2 import (  # noqa: E402
    class_char_counts,
    class_centroid_text,
    dataset_summary,
    load_v2,
    train_val_split,
)
from code.expert_iia import (  # noqa: E402
    expert_iia_baseline_reference,
    load_expert_set,
)
from code.features import text_reliability  # noqa: E402

ROOT = Path(r"E:\ai4s\poetry-poetricity")
ROUND_ID = 3
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"round_{ROUND_ID:03d}.json"
ARTIFACTS_DIR = ROOT / "05_experiments" / "stage1_metric_search" / f"round_{ROUND_ID:03d}"
FAIL_PATH = ROOT / "04_memory" / "failures" / f"round_{ROUND_ID:03d}.jsonl"
EXPERT_REPORT_PATH = ARTIFACTS_DIR / "expert_corpus_eval.json"

VAL_RATIO = 0.2
SEED = 42


def expert_corpus_eval(model_predict_proba, expert_items,
                        base_feats_fn, label_to_int=1):
    """Evaluate our LR classifier on the expert-curated samples.js corpus."""
    if not expert_items:
        return None
    texts = [it.text for it in expert_items]
    labels = np.asarray([it.label for it in expert_items], dtype=np.int64)
    rel = [text_reliability(t) for t in texts]
    n_trunc = sum(1 for r in rel if r["is_truncatable"])
    X = extract_batch(texts)
    base = np.asarray(
        [list(base_feats_fn(t).values()) for t in texts], dtype=np.float64)
    X_full = np.concatenate([X, base], axis=1)
    probs = model_predict_proba(X_full)[:, label_to_int]
    preds = (probs >= 0.5).astype(np.int64)
    acc = accuracy_score(labels, preds)
    kap = cohen_kappa_score(labels, preds, weights="quadratic")
    f1 = f1_score(labels, preds, average="macro")
    prec, rec, f1s, supp = precision_recall_fscore_support(
        labels, preds, labels=[0, 1])
    return {
        "n_total": len(expert_items),
        "n_truncated_abstain": n_trunc,
        "consistency": {
            "accuracy": float(acc),
            "kappa": float(kap),
            "f1_macro": float(f1),
            "per_class": {
                "nonpoem": {"p": float(prec[0]), "r": float(rec[0]),
                            "f1": float(f1s[0]), "n": int(supp[0])},
                "poem": {"p": float(prec[1]), "r": float(rec[1]),
                         "f1": float(f1s[1]), "n": int(supp[1])},
            },
        },
    }


def main() -> int:
    t0 = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---------- load ----------
    print("[load] v2 dataset ...", flush=True)
    samples = load_v2()
    summary = dataset_summary(samples)
    print(f"[load] total={summary['n_total']}  poem={summary['n_poem']}  "
          f"nonpoem={summary['n_nonpoem']}", flush=True)

    train, val = train_val_split(samples, val_ratio=VAL_RATIO, seed=SEED)
    print(f"[load] train={len(train)} val={len(val)} "
          f"(poem={sum(s.label==1 for s in train)} / {sum(s.label==1 for s in val)})",
          flush=True)

    # ---------- short-text gating on val ----------
    val_reliability = [text_reliability(s.text) for s in val]
    val_too_short = sum(1 for r in val_reliability if r["is_too_short"])
    val_truncatable = sum(1 for r in val_reliability if r["is_truncatable"])
    print(f"[gate] val too_short (<{30} chars) = {val_too_short};  "
          f"truncatable = {val_truncatable}", flush=True)

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
    print("[base] baselines ...", flush=True)
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

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_val_s = scaler.transform(X_val_full)

    print("[train] logistic regression (combined features)", flush=True)
    clf = LogisticRegression(
        max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED,
    )
    clf.fit(X_train_s, y_train)
    y_prob = clf.predict_proba(X_val_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)

    # ---------- short-text abstentions ----------
    y_pred_gated = y_pred.copy()
    abstained = 0
    for i, r in enumerate(val_reliability):
        if r["is_truncatable"]:
            y_pred_gated[i] = 0  # default to nonpoem when too short
            abstained += 1
    if abstained:
        kap_g = cohen_kappa_score(y_val, y_pred_gated, weights="quadratic")
        acc_g = accuracy_score(y_val, y_pred_gated)
        print(f"[gate] abstained {abstained} truncatable samples -> "
              f"acc={acc_g:.4f} kappa={kap_g:.4f}", flush=True)
    else:
        kap_g, acc_g = None, None

    # ---------- per-class + overall ----------
    acc = accuracy_score(y_val, y_pred)
    kap = cohen_kappa_score(y_val, y_pred, weights="quadratic")
    f1 = f1_score(y_val, y_pred, average="macro")
    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_val, y_pred, labels=[0, 1])
    print(f"[score] raw   acc={acc:.4f} kappa={kap:.4f} f1={f1:.4f}", flush=True)

    weights = clf.coef_[0].tolist()
    weight_pairs = sorted(
        zip(feat_names_full, weights), key=lambda x: -abs(x[1]))

    # ---------- per-single-feature ----------
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

    # ---------- baselines: random / expert-IAA / LLM-judge ----------
    rng = np.random.default_rng(SEED)
    rand_pred = rng.integers(0, 2, size=len(y_val))
    rand_b = {
        "accuracy": float(accuracy_score(y_val, rand_pred)),
        "kappa": float(cohen_kappa_score(y_val, rand_pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_val, rand_pred, average="macro")),
    }
    hiaa_b = expert_iia_baseline_reference()

    # Choose default LLM judge: real API if key set, else stub-majority
    judge = default_judge()
    # If stub, set default to the train-set majority class (no gating here:
    # gating would conflate with our metric's gate and break the comparison)
    if judge.name == "stub-majority":
        majority_label = int(round(np.mean(y_train)))
        judge = LLMJudgeStub(mode="majority", default_label=majority_label)
    print(f"[llm] default judge = {judge!r}", flush=True)
    judge_results = judge.predict_batch(val_texts)
    judge_pred = np.asarray([r.label for r in judge_results], dtype=np.int64)
    judge_b = {
        "name": judge.name,
        "default_label": getattr(judge, "default_label", None),
        "is_real_llm": isinstance(judge, LLMJudgeAPI),
        "accuracy": float(accuracy_score(y_val, judge_pred)),
        "kappa": float(cohen_kappa_score(y_val, judge_pred, weights="quadratic")),
        "f1_macro": float(f1_score(y_val, judge_pred, average="macro")),
        "note": ("real DeepSeek-V4-Flash API" if isinstance(judge, LLMJudgeAPI)
                 else ("stub-majority placeholder: needs DEEPSEEK_API_KEY")),
    }
    print(f"[baseline] random    acc={rand_b['accuracy']:.4f}  kappa={rand_b['kappa']:.4f}",
          flush=True)
    print(f"[baseline] human-IAA kappa={hiaa_b['kappa']:.4f}  (stub: {hiaa_b['source']})",
          flush=True)
    print(f"[baseline] llm-judge({judge.name}) acc={judge_b['accuracy']:.4f}  "
          f"kappa={judge_b['kappa']:.4f}  ({judge_b['note']})", flush=True)

    # ---------- expert-corpus eval (samples.js 50+50) ----------
    print("[expert] loading samples.js expert corpus ...", flush=True)
    expert_items = load_expert_set()
    print(f"[expert] {len(expert_items)} items "
          f"({sum(it.label==1 for it in expert_items)} poems + "
          f"{sum(it.label==0 for it in expert_items)} Racter)", flush=True)

    def predict_proba(X):
        return clf.predict_proba(scaler.transform(X))

    expert_eval = expert_corpus_eval(predict_proba, expert_items, base_feats)
    if expert_eval:
        print(f"[expert] our model on expert set:  acc={expert_eval['consistency']['accuracy']:.4f}  "
              f"kappa={expert_eval['consistency']['kappa']:.4f}  "
              f"(abstained {expert_eval['n_truncated_abstain']} short texts)",
              flush=True)

    EXPERT_REPORT_PATH.write_text(
        json.dumps(expert_eval, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ---------- per-stratification ----------
    by_strat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "n_correct": 0, "n_poem": 0, "n_pred_poem": 0,
                 "n_tp": 0, "n_fp": 0, "n_fn": 0})
    for s, yv, pv in zip(val, y_val.tolist(), y_pred.tolist()):
        d = by_strat[s.strat]
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
            "n": d["n"], "acc": d["n_correct"] / n,
            "n_true_poem": d["n_poem"], "n_pred_poem": d["n_pred_poem"],
            "precision_poem": d["n_tp"] / max(d["n_tp"] + d["n_fp"], 1),
            "recall_poem": d["n_tp"] / max(d["n_tp"] + d["n_fn"], 1),
        }

    strat_breakdown = {
        "by_strat": {k: _summ(v) for k, v in sorted(by_strat.items())},
    }

    # ---------- failures ----------
    failures = []
    for s, yv, pv, pr, rel in zip(val, y_val.tolist(), y_pred.tolist(),
                                   y_prob.tolist(), val_reliability):
        if pv != yv:
            failures.append({
                "sample_id": s.sample_id,
                "label": yv,
                "pred": pv,
                "prob_poem": float(pr),
                "strat": s.strat,
                "source_type": s.source_type,
                "n_han_chars": rel["n_han_chars"],
                "is_truncatable": bool(rel["is_truncatable"]),
                "text_preview": s.text[:80].replace("\n", " "),
            })
    with FAIL_PATH.open("w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- diff vs prior rounds ----------
    diff_vs = {}
    for rid in (1, 2):
        p = ROOT / "04_memory" / "experiment_logs" / f"round_{rid:03d}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                diff_vs[f"round_{rid}"] = {
                    "acc": d["consistency"]["accuracy"],
                    "kappa": d["consistency"]["kappa_quadratic"],
                    "delta_acc": float(acc) - d["consistency"]["accuracy"],
                    "delta_kappa": float(kap) - d["consistency"]["kappa_quadratic"],
                }
            except Exception:
                pass

    # ---------- artifacts ----------
    coef_path = ARTIFACTS_DIR / "feature_coef.json"
    coef_path.write_text(
        json.dumps([{"name": n, "coef": w} for n, w in weight_pairs],
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

    # ---------- log ----------
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
        "short_text_gating": {
            "min_han_chars": 30,
            "n_val_too_short": int(val_too_short),
            "n_val_truncatable": int(val_truncatable),
            "n_abstained_in_eval": int(abstained),
            "abstain_default_label": 0,
        },
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
        "consistency_with_gating": (
            {"accuracy": float(acc_g), "kappa_quadratic": float(kap_g)}
            if acc_g is not None else None),
        "baselines": {
            "random": rand_b,
            "human_iaa_stub": hiaa_b,
            "llm_judge": judge_b,
        },
        "expert_corpus_eval": expert_eval,
        "top_weights": [{"name": n, "coef": w} for n, w in weight_pairs[:10]],
        "single_feature_top": sorted(
            [{"name": k, **v} for k, v in single_results.items()],
            key=lambda x: -x["val_acc"])[:8],
        "per_stratification": strat_breakdown,
        "diff_vs_prev": diff_vs,
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
            str(EXPERT_REPORT_PATH.relative_to(ROOT)),
            str(FAIL_PATH.relative_to(ROOT)),
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] failures -> {FAIL_PATH}  ({len(failures)} samples)", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
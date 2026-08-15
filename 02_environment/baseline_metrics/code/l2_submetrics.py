"""L2 sub-metrics: train one LR per feature family, then meta-LR to combine.

This directly answers scheme §1.3:
  "中文诗歌性自动评测指标的设计与验证... 检验权重在专业组/非专业组标注下的差异"
i.e. "which feature families does human judgment most depend on?"

Pipeline:
  1. For each family (form, struct, jump, lang, purity, style, music):
     - train a family-only LR on the same data
     - evaluate on val split
  2. Meta-LR: combine the 7 family scores + 5 baseline scores
  3. Report per-family importance: drop each family, retrain, measure kappa drop
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import extract_batch, build_and_freeze, FEATURE_NAMES  # noqa: E402
from code.baselines import (  # noqa: E402
    bleu_self, bigram_jaccard_self, build_char_vocab,
    build_tfidf_centroid, char_tfidf_cosine_to_poetry_centroid,
)
from code.data_loader_v2 import (  # noqa: E402
    class_char_counts, load_v2, train_val_split,
)

ROUND_ID = 13
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


FAMILY_FEATURES = {
    "form":   [f for f in FEATURE_NAMES if f.startswith("form_")],
    "struct": [f for f in FEATURE_NAMES if f.startswith("struct_")],
    "jump":   [f for f in FEATURE_NAMES if f.startswith("jump_")],
    "lang":   [f for f in FEATURE_NAMES if f.startswith("lang_")],
    "purity": [f for f in FEATURE_NAMES if f.startswith("purity_")],
    "style":  [f for f in FEATURE_NAMES if f.startswith("style_")],
    "music":  [f for f in FEATURE_NAMES if f.startswith("music_")],
}
BASELINE_NAMES = [
    "base_bleu_to_poem", "base_bleu_to_nonpoem",
    "base_bigram_jacc_to_poem", "base_bigram_jacc_to_nonpoem",
    "base_tfidf_cos_to_poem",
]


def class_char_counts_from_texts(texts, n):
    df, total = Counter(), 0
    for t in texts:
        han = "".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams); total += len(grams)
    return df, total


def build_features(samples, extra_pos=None, extra_neg=None, seed=42):
    """Build (X_own, X_base, y) arrays for the full set of training samples."""
    train, _ = train_val_split(samples, val_ratio=0.2, seed=seed)
    texts = [s.text for s in train]
    labels = [s.label for s in train]
    if extra_pos:
        texts += extra_pos
        labels += [1] * len(extra_pos)
    if extra_neg:
        texts += extra_neg
        labels += [0] * len(extra_neg)
    X_own = extract_batch(texts)
    y = np.asarray(labels, dtype=np.int64)

    poems = [t for t, l in zip(texts, labels) if l == 1]
    nonpoems = [t for t, l in zip(texts, labels) if l == 0]
    cp, tp = class_char_counts_from_texts(poems, 1)
    cn, tn = class_char_counts_from_texts(nonpoems, 1)
    cp2, _ = class_char_counts_from_texts(poems, 2)
    cn2, _ = class_char_counts_from_texts(nonpoems, 2)
    vocab = build_char_vocab(texts, min_df=2)
    cent, idf = build_tfidf_centroid(poems, vocab)
    X_base = np.asarray([
        [
            bleu_self(t, cp, tp), bleu_self(t, cn, tn),
            bigram_jaccard_self(t, cp2), bigram_jaccard_self(t, cn2),
            char_tfidf_cosine_to_poetry_centroid(t, vocab, cent, idf),
        ]
        for t in texts
    ], dtype=np.float64)
    return X_own, X_base, y, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn


def fit_lr(X, y, seed=42):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(X_s, y)
    return clf, scaler


def prob(clf, scaler, X):
    return clf.predict_proba(scaler.transform(X))[:, 1]


def main():
    t0 = time.time()
    print("[load] base samples ...", flush=True)
    samples = load_v2()
    _, val = train_val_split(samples, val_ratio=0.2, seed=42)
    val_texts = [s.text for s in val]
    y_val = np.asarray([s.label for s in val], dtype=np.int64)

    # also: add 32 AI garbage negatives (same as v4b)
    import csv, json
    EXPORT6 = Path(r"E:\生成诗歌\annotations_export6.csv")
    HARD_FILES = [
        Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl"),
        Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl"),
        Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl"),
        Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl"),
        Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl"),
    ]
    ann = []
    with EXPORT6.open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["sample_id"] = int(r["sample_id"])
            r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
            ann.append(r)
    title_author = {}
    for path in HARD_FILES:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = (obj.get("title", ""), obj.get("model", ""))
                title_author.setdefault(key, []).append(obj)
    matched = []
    for a in ann:
        key = (a["title"], a["author"])
        if key in title_author:
            matched.append({**a, **title_author[key][0]})
    ai_neg = [m["generated"] for m in matched if not m["is_poetry"]]
    print(f"[load] AI garbage negatives: {len(ai_neg)}", flush=True)

    # ----- 1) fit v4b baseline + sub-metrics on the same training data -----
    print("\n[fit] training v4b baseline + 7 sub-metrics ...", flush=True)
    X_own, X_base, y, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn = \
        build_features(samples, extra_neg=ai_neg, seed=42)
    n = len(y)
    print(f"  train: n={n} (with {len(ai_neg)} AI neg added)", flush=True)

    # v4b baseline
    v4b_X = np.concatenate([X_own, X_base], axis=1)
    v4b_clf, v4b_scaler = fit_lr(v4b_X, y)
    v4b_clf.__class__ = LogisticRegression  # ensure picklable
    v4b_coef = v4b_clf.coef_[0].tolist()

    # family index maps
    fam_idx = {fam: [FEATURE_NAMES.index(f) for f in feats]
               for fam, feats in FAMILY_FEATURES.items()}
    base_idx = list(range(len(FEATURE_NAMES),
                         len(FEATURE_NAMES) + len(BASELINE_NAMES)))

    # 1a) each family alone + baselines
    family_models = {}
    family_val_probs = {}
    for fam, idx in fam_idx.items():
        if not idx:
            continue
        X_fam = np.concatenate([X_own[:, idx], X_base], axis=1)
        clf, sc = fit_lr(X_fam, y)
        family_models[fam] = (clf, sc)
    # 1b) baselines alone
    clf_b, sc_b = fit_lr(X_base, y)

    # 1c) v2 baseline (no AI negs, original feature set)
    samples0 = load_v2()
    X_own0, X_base0, y0, *_ = build_features(samples0, seed=42)
    X_v2 = np.concatenate([X_own0, X_base0], axis=1)
    v2_clf, v2_scaler = fit_lr(X_v2, y0)

    # ----- 2) validate each model on val -----
    print("\n[eval] on val split (each family alone + combined) ...", flush=True)
    Xv_own, Xv_base = extract_batch(val_texts), None
    # need to recompute val base features using v4b corpus stats
    # For simplicity, we build the val base features using the v4b corpus
    def base_feats_for(texts, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn):
        return np.asarray([
            [
                bleu_self(t, cp, tp), bleu_self(t, cn, tn),
                bigram_jaccard_self(t, cp2), bigram_jaccard_self(t, cn2),
                char_tfidf_cosine_to_poetry_centroid(t, vocab, cent, idf),
            ]
            for t in texts
        ], dtype=np.float64)
    Xv_base = base_feats_for(val_texts, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn)
    Xv_own0, Xv_base0 = extract_batch(val_texts), base_feats_for(
        val_texts, *build_features(samples0, seed=42)[3:])

    # v4b val
    v4b_val = (v4b_clf.predict_proba(v4b_scaler.transform(
        np.concatenate([Xv_own, Xv_base], axis=1)))[:, 1] >= 0.5).astype(int)
    v2_val = (v2_clf.predict_proba(v2_scaler.transform(
        np.concatenate([Xv_own0, Xv_base0], axis=1)))[:, 1] >= 0.5).astype(int)

    print(f"  v2 (no AI neg):       kappa={cohen_kappa_score(y_val, v2_val, weights='quadratic'):.4f}", flush=True)
    print(f"  v4b (with AI neg):    kappa={cohen_kappa_score(y_val, v4b_val, weights='quadratic'):.4f}", flush=True)

    family_results = {}
    for fam, (clf, sc) in family_models.items():
        idx = fam_idx[fam]
        Xv = np.concatenate([Xv_own[:, idx], Xv_base], axis=1)
        p = clf.predict_proba(sc.transform(Xv))[:, 1]
        preds = (p >= 0.5).astype(int)
        family_val_probs[fam] = p
        kap = cohen_kappa_score(y_val, preds, weights="quadratic")
        acc = accuracy_score(y_val, preds)
        family_results[fam] = {"acc": float(acc), "kappa": float(kap),
                                "n_features": len(idx)}
        print(f"  fam={fam:8s}  n_features={len(idx)}  acc={acc:.4f}  kappa={kap:.4f}", flush=True)
    # baselines alone
    p_b = clf_b.predict_proba(sc_b.transform(Xv_base))[:, 1]
    preds_b = (p_b >= 0.5).astype(int)
    kap_b = cohen_kappa_score(y_val, preds_b, weights="quadratic")
    family_results["baseline"] = {"acc": float(accuracy_score(y_val, preds_b)),
                                    "kappa": float(kap_b), "n_features": 5}
    print(f"  fam=baseline  n_features=5  acc={accuracy_score(y_val, preds_b):.4f}  kappa={kap_b:.4f}", flush=True)

    # ----- 3) L2 meta-LR: combine family scores as input to a final LR -----
    print("\n[meta] L2: stack family scores as meta-features ...", flush=True)
    # Build train-side family scores (use same training data)
    train, _ = train_val_split(samples, val_ratio=0.2, seed=42)
    train_texts = [s.text for s in train]
    train_y = np.asarray([s.label for s in train], dtype=np.int64)
    n_train = len(train_y)
    # also need ai_neg labels (all 0)
    if ai_neg:
        n_train += len(ai_neg)
        train_y_full = np.concatenate([train_y, np.zeros(len(ai_neg), dtype=np.int64)])
    else:
        train_y_full = train_y

    # train-side family scores: extract features for train + ai_neg
    train_all_texts = train_texts + list(ai_neg)
    Xt_own = extract_batch(train_all_texts)
    Xt_base = base_feats_for(train_all_texts, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn)

    train_family_probs = {}
    for fam, (clf, sc) in family_models.items():
        idx = fam_idx[fam]
        Xt = np.concatenate([Xt_own[:, idx], Xt_base], axis=1)
        train_family_probs[fam] = clf.predict_proba(sc.transform(Xt))[:, 1]
    p_b_train = clf_b.predict_proba(sc_b.transform(Xt_base))[:, 1]

    # meta-features: each family's prob + baseline prob
    fams = list(family_models.keys())
    Xt_meta = np.column_stack([train_family_probs[f] for f in fams] + [p_b_train])
    Xv_meta = np.column_stack([family_val_probs[f] for f in fams] + [p_b])

    meta_clf, meta_scaler = fit_lr(Xt_meta, train_y_full)
    meta_pred = (meta_clf.predict_proba(meta_scaler.transform(Xv_meta))[:, 1] >= 0.5).astype(int)
    meta_kap = cohen_kappa_score(y_val, meta_pred, weights="quadratic")
    meta_acc = accuracy_score(y_val, meta_pred)
    print(f"  L2 meta-LR: acc={meta_acc:.4f}  kappa={meta_kap:.4f}", flush=True)
    # family importance from meta-LR coefficients
    meta_coef = meta_clf.coef_[0].tolist()
    family_importance = dict(zip(fams + ["baseline"], meta_coef))

    # ----- 4) leave-one-family-out ablation -----
    print("\n[ablation] leave-one-family-out (v4b retrain, drop one family) ...", flush=True)
    Xt_full = np.concatenate([Xt_own, Xt_base], axis=1)
    ablation = {}
    for fam in list(FAMILY_FEATURES.keys()):
        if not fam_idx[fam]:
            continue
        keep_idx = [i for i in range(X_own.shape[1]) if i not in fam_idx[fam]]
        Xt_drop = np.concatenate([Xt_own[:, keep_idx], Xt_base], axis=1)
        Xv_drop_idx = keep_idx
        Xv_drop_own = Xv_own[:, Xv_drop_idx]
        Xv_drop = np.concatenate([Xv_drop_own, Xv_base], axis=1)
        clf, sc = fit_lr(Xt_drop, train_y_full)
        p = clf.predict_proba(sc.transform(Xv_drop))[:, 1]
        preds = (p >= 0.5).astype(int)
        kap = cohen_kappa_score(y_val, preds, weights="quadratic")
        ablation[fam] = float(kap)
        print(f"  drop {fam:8s}  kappa={kap:.4f}  (full={cohen_kappa_score(y_val, v4b_val, weights='quadratic'):.4f})", flush=True)
    # importance = full_kappa - drop_kappa
    full_kappa = float(cohen_kappa_score(y_val, v4b_val, weights="quadratic"))
    importance = {fam: full_kappa - kap for fam, kap in ablation.items()}
    print("\n[ablation] family importance (full - drop):", flush=True)
    for fam, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {fam:8s}  Δkappa = {imp:+.4f}", flush=True)

    # ----- 5) write log -----
    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "L2 sub-metric analysis (family-by-family ablation)",
        "family_features": {fam: feats for fam, feats in FAMILY_FEATURES.items()},
        "n_features_per_family": {fam: len(feats) for fam, feats in FAMILY_FEATURES.items()},
        "n_train": int(len(train_y_full)),
        "single_family_results": family_results,
        "v4b_full_kappa": full_kappa,
        "v2_baseline_kappa": float(cohen_kappa_score(y_val, v2_val, weights="quadratic")),
        "l2_meta_kappa": float(meta_kap),
        "l2_meta_acc": float(meta_acc),
        "family_importance_from_meta_coef": family_importance,
        "ablation_importance": importance,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
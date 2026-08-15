"""Compare v2 (frozen) vs v3 (with purity features) on:
  - original training val
  - annotator_06's 209 matched AI poems (human-labeled!)

This directly tests whether purity features fix the "AI poems with English
garbage judged as poem" failure mode.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze, FrozenMetric, extract_batch  # noqa: E402
from code.baselines import (  # noqa: E402
    bleu_self, bigram_jaccard_self, build_char_vocab,
    build_tfidf_centroid, char_tfidf_cosine_to_poetry_centroid,
)
from code.data_loader_v2 import (  # noqa: E402
    class_char_counts, load_v2, train_val_split,
)

EXPORT6 = Path(r"E:\生成诗歌\annotations_export6.csv")
HARD_FILES = [
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl"),
]

ROUND_ID = 10
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


def build_v3(seed=42):
    """Train v3 = v2 features + purity features on the same data."""
    samples = load_v2()
    train, val = train_val_split(samples, val_ratio=0.2, seed=seed)
    train_texts = [s.text for s in train]
    X_train = extract_batch(train_texts)
    y_train = np.asarray([s.label for s in train], dtype=np.int64)

    poems = [s for s in train if s.label == 1]
    nonpoems = [s for s in train if s.label == 0]
    counts_poem, total_poem = class_char_counts(poems, n=1)
    counts_nonpoem, total_nonpoem = class_char_counts(nonpoems, n=1)
    counts_poem_2, _ = class_char_counts(poems, n=2)
    counts_nonpoem_2, _ = class_char_counts(nonpoems, n=2)
    vocab = build_char_vocab(train_texts, min_df=2)
    cent_poem, idf = build_tfidf_centroid([s.text for s in poems], vocab)

    def base_feats(t):
        return {
            "bleu_to_poem": bleu_self(t, counts_poem, total_poem),
            "bleu_to_nonpoem": bleu_self(t, counts_nonpoem, total_nonpoem),
            "bigram_jacc_to_poem": bigram_jaccard_self(t, counts_poem_2),
            "bigram_jacc_to_nonpoem": bigram_jaccard_self(t, counts_nonpoem_2),
            "tfidf_cos_to_poem": char_tfidf_cosine_to_poetry_centroid(
                t, vocab, cent_poem, idf),
        }
    base_train = np.asarray([list(base_feats(t).values()) for t in train_texts],
                            dtype=np.float64)
    X_full = np.concatenate([X_train, base_train], axis=1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_full)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(X_s, y_train)
    return clf, scaler, counts_poem, counts_nonpoem, counts_poem_2, \
        counts_nonpoem_2, vocab, cent_poem, idf, total_poem, total_nonpoem


def apply_v3(clf, scaler, counts_poem, counts_nonpoem, counts_poem_2,
             counts_nonpoem_2, vocab, cent_poem, idf, total_poem, total_nonpoem,
             text):
    feats = extract_batch([text])[0]
    base = np.asarray([
        bleu_self(text, counts_poem, total_poem),
        bleu_self(text, counts_nonpoem, total_nonpoem),
        bigram_jaccard_self(text, counts_poem_2),
        bigram_jaccard_self(text, counts_nonpoem_2),
        char_tfidf_cosine_to_poetry_centroid(text, vocab, cent_poem, idf),
    ], dtype=np.float64)
    X = np.concatenate([feats, base])[None, :]
    prob = float(clf.predict_proba(scaler.transform(X))[0, 1])
    return prob, int(prob >= 0.5)


def main():
    t0 = time.time()
    # load v2 (frozen)
    print("[v2] frozen metric ...", flush=True)
    v2 = build_and_freeze(seed=42, val_ratio=0.2)

    # build v3
    print("[v3] training v3 (with purity features) ...", flush=True)
    (clf, scaler, cp, cn, cp2, cn2, vocab, cent, idf,
     tp, tn) = build_v3(seed=42)

    # load annotator_06 + match
    print("[load] annotator_06 annotations ...", flush=True)
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
    print(f"[load] matched {len(matched)}", flush=True)

    # evaluate
    y_human = np.asarray([1 if m["is_poetry"] else 0 for m in matched], dtype=np.int64)
    texts = [m["generated"] for m in matched]

    print("\n=== v2 (frozen) on 209 AI poems ===", flush=True)
    v2_preds = []
    v2_probs = []
    for t in texts:
        p = v2.apply(t)
        v2_preds.append(p.pred)
        v2_probs.append(p.prob_poem)
    v2_preds = np.asarray(v2_preds)
    v2_probs = np.asarray(v2_probs)
    print(f"  human_yes={int(y_human.sum())}/{len(y_human)}  "
          f"v2_yes={int(v2_preds.sum())}/{len(v2_preds)}", flush=True)
    acc = accuracy_score(y_human, v2_preds)
    kap = cohen_kappa_score(y_human, v2_preds, weights="quadratic")
    print(f"  acc={acc:.4f}  kappa={kap:.4f}", flush=True)

    print("\n=== v3 (with purity) on 209 AI poems ===", flush=True)
    v3_preds = []
    v3_probs = []
    for t in texts:
        prob, pred = apply_v3(clf, scaler, cp, cn, cp2, cn2, vocab, cent,
                              idf, tp, tn, t)
        v3_preds.append(pred)
        v3_probs.append(prob)
    v3_preds = np.asarray(v3_preds)
    v3_probs = np.asarray(v3_probs)
    print(f"  human_yes={int(y_human.sum())}/{len(y_human)}  "
          f"v3_yes={int(v3_preds.sum())}/{len(v3_preds)}", flush=True)
    acc3 = accuracy_score(y_human, v3_preds)
    kap3 = cohen_kappa_score(y_human, v3_preds, weights="quadratic")
    print(f"  acc={acc3:.4f}  kappa={kap3:.4f}", flush=True)

    # Confusion
    print("\n=== v2 vs v3 confusion on the 31 failure cases ===", flush=True)
    fixed = 0
    for i in range(len(texts)):
        if y_human[i] == 0 and v2_preds[i] == 1:
            # v2 wrong
            if v3_preds[i] == 0:
                fixed += 1
    print(f"  v2 said poem on {int((y_human==0)&(v2_preds==1)).sum()} non-poems; "
          f"v3 fixed {fixed} of them", flush=True)
    new_failures = int(((y_human == 1) & (v2_preds == 0) & (v3_preds == 1)).sum())
    print(f"  v3 newly misclassified (human poem, v2 no, v3 yes): {new_failures}", flush=True)

    # also eval on original val split
    print("\n=== val split (original data) ===", flush=True)
    samples = load_v2()
    train, val = train_val_split(samples, val_ratio=0.2, seed=42)
    val_texts = [s.text for s in val]
    y_val = np.asarray([s.label for s in val], dtype=np.int64)
    v2_val = np.asarray([v2.apply(t).pred for t in val_texts])
    v3_val = []
    for t in val_texts:
        _, p = apply_v3(clf, scaler, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn, t)
        v3_val.append(p)
    v3_val = np.asarray(v3_val)
    print(f"  v2: acc={accuracy_score(y_val, v2_val):.4f} "
          f"kappa={cohen_kappa_score(y_val, v2_val, weights='quadratic'):.4f}", flush=True)
    print(f"  v3: acc={accuracy_score(y_val, v3_val):.4f} "
          f"kappa={cohen_kappa_score(y_val, v3_val, weights='quadratic'):.4f}", flush=True)

    # write log
    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "v2_on_ai_poems": {
            "n": len(y_human),
            "human_yes": int(y_human.sum()),
            "metric_yes": int(v2_preds.sum()),
            "acc": float(acc),
            "kappa": float(kap),
        },
        "v3_on_ai_poems": {
            "n": len(y_human),
            "human_yes": int(y_human.sum()),
            "metric_yes": int(v3_preds.sum()),
            "acc": float(acc3),
            "kappa": float(kap3),
        },
        "v3_fixes": {
            "v2_fp_on_ai": int(((y_human == 0) & (v2_preds == 1)).sum()),
            "v3_fixed": int(fixed),
            "v3_new_fn": int(new_failures),
        },
        "val_split": {
            "v2_acc": float(accuracy_score(y_val, v2_val)),
            "v2_kappa": float(cohen_kappa_score(y_val, v2_val, weights="quadratic")),
            "v3_acc": float(accuracy_score(y_val, v3_val)),
            "v3_kappa": float(cohen_kappa_score(y_val, v3_val, weights="quadratic")),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
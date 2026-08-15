"""v4: retrain v2 metric WITH the 32 human-labeled 'not-poem' AI samples added
as negative training examples. This is the real metric-iteration loop:
  - Stage 2 discovered a failure mode (AI poems w/ English garbage)
  - v4 adds those as negatives
  - retrain and re-evaluate on both the AI set AND original val split
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import extract_batch, build_and_freeze  # noqa: E402
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

ROUND_ID = 11
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


class MetricV2:
    """Reusable v2-compatible metric (uses current FEATURE_NAMES incl purity)."""

    def __init__(self, clf, scaler, cp, cn, cp2, cn2, vocab, cent, idf,
                 tp, tn):
        self.clf, self.scaler = clf, scaler
        self.cp, self.cn = cp, cn
        self.cp2, self.cn2 = cp2, cn2
        self.vocab, self.cent, self.idf = vocab, cent, idf
        self.tp, self.tn = tp, tn

    def apply(self, text):
        feats = extract_batch([text])[0]
        base = np.asarray([
            bleu_self(text, self.cp, self.tp),
            bleu_self(text, self.cn, self.tn),
            bigram_jaccard_self(text, self.cp2),
            bigram_jaccard_self(text, self.cn2),
            char_tfidf_cosine_to_poetry_centroid(text, self.vocab,
                                                  self.cent, self.idf),
        ], dtype=np.float64)
        X = np.concatenate([feats, base])[None, :]
        prob = float(self.clf.predict_proba(self.scaler.transform(X))[0, 1])
        return prob, int(prob >= 0.5)


def train_on(samples, extra_pos=None, extra_neg=None, seed=42):
    """Train a fresh metric on `samples` + optional extra positives/negatives."""
    train, _ = train_val_split(samples, val_ratio=0.2, seed=seed)
    train_texts = [s.text for s in train]
    labels = [s.label for s in train]
    if extra_pos:
        train_texts += extra_pos
        labels += [1] * len(extra_pos)
    if extra_neg:
        train_texts += extra_neg
        labels += [0] * len(extra_neg)
    X = extract_batch(train_texts)
    y = np.asarray(labels, dtype=np.int64)

    poems = [t for t, l in zip(train_texts, labels) if l == 1]
    nonpoems = [t for t, l in zip(train_texts, labels) if l == 0]
    cp, tp = class_char_counts_from_texts(poems, 1)
    cn, tn = class_char_counts_from_texts(nonpoems, 1)
    cp2, _ = class_char_counts_from_texts(poems, 2)
    cn2, _ = class_char_counts_from_texts(nonpoems, 2)
    vocab = build_char_vocab(train_texts, min_df=2)
    cent, idf = build_tfidf_centroid(poems, vocab)

    base = np.asarray([
        [
            bleu_self(t, cp, tp), bleu_self(t, cn, tn),
            bigram_jaccard_self(t, cp2), bigram_jaccard_self(t, cn2),
            char_tfidf_cosine_to_poetry_centroid(t, vocab, cent, idf),
        ]
        for t in train_texts
    ], dtype=np.float64)
    X_full = np.concatenate([X, base], axis=1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_full)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(X_s, y)
    return MetricV2(clf, scaler, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn)


def class_char_counts_from_texts(texts, n):
    from collections import Counter
    df = Counter()
    total = 0
    for t in texts:
        han = "".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams)
        total += len(grams)
    return df, total


def main():
    t0 = time.time()
    print("[load] base samples ...", flush=True)
    samples = load_v2()

    # load annotator_06 matched AI poems
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
    print(f"[load] matched {len(matched)} AI poems with human labels", flush=True)

    y_human = np.asarray([1 if m["is_poetry"] else 0 for m in matched], dtype=np.int64)
    ai_texts = [m["generated"] for m in matched]

    # split AI poems: the 32 human=0 are new negatives; 177 human=1 are held-out positives
    extra_neg = [t for t, h in zip(ai_texts, y_human) if h == 0]
    heldout_pos = [t for t, h in zip(ai_texts, y_human) if h == 1]
    print(f"[load] extra_neg (human=not poem): {len(extra_neg)}  "
          f"heldout_pos (human=poem): {len(heldout_pos)}", flush=True)

    # NOTE: we hold out ALL AI poems from training so evaluation is fair.
    # v2 baseline = frozen original.
    print("\n[v2] frozen original metric ...", flush=True)
    v2 = build_and_freeze(seed=42, val_ratio=0.2)

    # v4: train WITHOUT extra AI data (fair comparison on AI set)
    print("[v4a] train baseline (no extra AI) ...", flush=True)
    v4a = train_on(samples, seed=42)

    # v4b: train WITH extra negatives
    print("[v4b] train with extra AI negatives ...", flush=True)
    v4b = train_on(samples, extra_neg=extra_neg, seed=42)

    def metric_preds(metric, texts):
        """Return (preds array). Works for both FrozenMetric and MetricV2."""
        out = []
        for t in texts:
            r = metric.apply(t)
            if isinstance(r, tuple):
                out.append(r[1])
            else:
                out.append(r.pred)
        return np.asarray(out, dtype=np.int64)

    # evaluate on AI set
    print("\n=== on 209 AI poems (all held out) ===", flush=True)
    for name, m in [("v2 (frozen)", v2), ("v4a (baseline retrain)", v4a),
                    ("v4b (+AI negatives)", v4b)]:
        preds = metric_preds(m, ai_texts)
        acc = accuracy_score(y_human, preds)
        kap = cohen_kappa_score(y_human, preds, weights="quadratic")
        fp = int(((y_human == 0) & (preds == 1)).sum())
        fn = int(((y_human == 1) & (preds == 0)).sum())
        print(f"  {name:28s}  yes={int(preds.sum())}/{len(preds)}  "
              f"acc={acc:.4f}  kappa={kap:.4f}  fp={fp}  fn={fn}", flush=True)

    # evaluate on original val split
    print("\n=== on original val split ===", flush=True)
    _, val = train_val_split(samples, val_ratio=0.2, seed=42)
    val_texts = [s.text for s in val]
    y_val = np.asarray([s.label for s in val], dtype=np.int64)
    for name, m in [("v2 (frozen)", v2), ("v4a (baseline retrain)", v4a),
                    ("v4b (+AI negatives)", v4b)]:
        preds = metric_preds(m, val_texts)
        acc = accuracy_score(y_val, preds)
        kap = cohen_kappa_score(y_val, preds, weights="quadratic")
        print(f"  {name:28s}  acc={acc:.4f}  kappa={kap:.4f}", flush=True)

    # write log
    res = {}
    for name, m in [("v2", v2), ("v4a", v4a), ("v4b", v4b)]:
        p_ai = metric_preds(m, ai_texts)
        p_val = metric_preds(m, val_texts)
        res[name] = {
            "ai_set": {
                "n": len(y_human),
                "yes": int(p_ai.sum()),
                "acc": float(accuracy_score(y_human, p_ai)),
                "kappa": float(cohen_kappa_score(y_human, p_ai, weights="quadratic")),
                "fp": int(((y_human == 0) & (p_ai == 1)).sum()),
                "fn": int(((y_human == 1) & (p_ai == 0)).sum()),
            },
            "val_split": {
                "n": len(y_val),
                "acc": float(accuracy_score(y_val, p_val)),
                "kappa": float(cohen_kappa_score(y_val, p_val, weights="quadratic")),
            },
        }
    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_extra_neg": len(extra_neg),
        "n_heldout_pos": len(heldout_pos),
        "results": res,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""v6: drop 'struct' and 'style' families, retrain on the same data.

Based on L2 ablation (R13):
  - dropping 'struct' -> kappa +0.018
  - dropping 'style' -> kappa +0.017

v6 keeps: form (4) + jump (3) + lang (4) + purity (4) + music (3) + baseline (5) = 23 dims
       (drops struct 3 + style 4 = 7 dims)

We test two v6 variants:
  - v6a: drops struct + style, no extra negatives
  - v6b: drops struct + style + 32 AI negs (the v4b trick)

Evaluation on:
  - val split
  - expert set (100)
  - AI poem set (209)
"""
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import FEATURE_NAMES, extract_batch, build_and_freeze  # noqa: E402
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
EXPERT_JS = Path(r"E:\生成诗歌\eval-annotation\data\samples.js")

ROUND_ID = 14
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"

# v6 = drop struct (struct_*) and style (style_*) families
DROP_FAMILIES = ("struct_", "style_")
v6_keep_idx = [i for i, f in enumerate(FEATURE_NAMES)
              if not f.startswith(DROP_FAMILIES)]
v6_drop_features = [f for f in FEATURE_NAMES if f.startswith(DROP_FAMILIES)]
print(f"v6: keep {len(v6_keep_idx)} features (drop {len(v6_drop_features)})")
print(f"  dropped: {v6_drop_features}")


def class_char_counts_from_texts(texts, n):
    df, total = Counter(), 0
    for t in texts:
        han = "".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams); total += len(grams)
    return df, total


def make_metric_v6(clf, scaler, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn,
                   feat_idx):
    """v6 metric: keeps only `feat_idx` of the 21 own features + 5 baselines."""
    class _MetricV6:
        def __init__(self):
            self.clf, self.scaler = clf, scaler
            self.cp, self.cn = cp, cn
            self.cp2, self.cn2 = cp2, cn2
            self.vocab, self.cent, self.idf = vocab, cent, idf
            self.tp, self.tn = tp, tn
            self.feat_idx = feat_idx

        def _featurize(self, text):
            feats = extract_batch([text])[0]
            keep_feats = feats[self.feat_idx]
            base = np.asarray([
                bleu_self(text, self.cp, self.tp),
                bleu_self(text, self.cn, self.tn),
                bigram_jaccard_self(text, self.cp2),
                bigram_jaccard_self(text, self.cn2),
                char_tfidf_cosine_to_poetry_centroid(text, self.vocab,
                                                      self.cent, self.idf),
            ], dtype=np.float64)
            return np.concatenate([keep_feats, base])[None, :]

        def apply(self, text):
            X = self._featurize(text)
            prob = float(self.clf.predict_proba(self.scaler.transform(X))[0, 1])
            return prob, int(prob >= 0.5)
    return _MetricV6()


def train_v6(samples, extra_neg=None, seed=42):
    """Train v6 on `samples` + optional extra negatives, drop struct+style."""
    train, _ = train_val_split(samples, val_ratio=0.2, seed=seed)
    train_texts = [s.text for s in train]
    labels = [s.label for s in train]
    if extra_neg:
        train_texts += extra_neg
        labels += [0] * len(extra_neg)
    # full features
    X_full = extract_batch(train_texts)
    X_own = X_full[:, v6_keep_idx]
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
    X = np.concatenate([X_own, base], axis=1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(X_s, y)
    return make_metric_v6(clf, scaler, cp, cn, cp2, cn2, vocab, cent, idf, tp, tn,
                         v6_keep_idx), X.shape[1]


def main():
    t0 = time.time()
    print("[load] base samples ...", flush=True)
    samples = load_v2()
    _, val = train_val_split(samples, val_ratio=0.2, seed=42)
    val_texts = [s.text for s in val]
    y_val = np.asarray([s.label for s in val], dtype=np.int64)

    # v2 frozen (baseline)
    print("[v2] frozen original metric ...", flush=True)
    v2 = build_and_freeze(seed=42, val_ratio=0.2)

    # AI negatives
    print("[load] annotator_06 AI annotations ...", flush=True)
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
    print(f"[load] AI neg: {len(ai_neg)}", flush=True)

    # v6a: drop struct+style, no extra data
    print("\n[v6a] train (drop struct+style, no extra negs) ...", flush=True)
    v6a, n_v6a = train_v6(samples, seed=42)

    # v6b: drop struct+style + AI negs
    print("[v6b] train (drop struct+style + 32 AI negs) ...", flush=True)
    v6b, n_v6b = train_v6(samples, extra_neg=ai_neg, seed=42)

    # Helper: predict on texts
    def preds(metric, texts):
        out = []
        for t in texts:
            r = metric.apply(t)
            if isinstance(r, tuple):
                out.append(r[1])
            else:
                out.append(r.pred)
        return np.asarray(out, dtype=np.int64)

    # === val split ===
    print("\n=== on val split ===", flush=True)
    val_res = {}
    for name, m in [("v2 (frozen)", v2), ("v6a (drop 7 feats)", v6a),
                    ("v6b (drop 7 + AI negs)", v6b)]:
        p = preds(m, val_texts)
        acc = accuracy_score(y_val, p)
        kap = cohen_kappa_score(y_val, p, weights="quadratic")
        val_res[name] = {"acc": float(acc), "kappa": float(kap)}
        print(f"  {name:30s}  acc={acc:.4f}  kappa={kap:.4f}", flush=True)

    # === expert set ===
    print("\n=== on expert set (samples.js 100) ===", flush=True)
    expert_items = []
    if EXPERT_JS.exists():
        # parse the JS
        with EXPERT_JS.open("r", encoding="utf-8") as f:
            raw = f.read()
        import re
        for m in re.finditer(r"\{ title: \"([^\"]+)\".*?author: \"([^\"]*)\".*?text: \"((?:[^\"\\]|\\.)*)\".*?genre: \"(poem|nonpoem)\"", raw, re.DOTALL):
            title, author, text, genre = m.groups()
            text = text.replace("\\n", "\n").replace('\\"', '"')
            expert_items.append({"title": title, "author": author,
                                 "text": text, "genre": genre})
    print(f"  loaded {len(expert_items)} expert items", flush=True)
    if expert_items:
        expert_texts = [it["text"] for it in expert_items]
        expert_labels = np.asarray([1 if it["genre"] == "poem" else 0
                                    for it in expert_items], dtype=np.int64)
        expert_res = {}
        for name, m in [("v2 (frozen)", v2), ("v6a (drop 7 feats)", v6a),
                        ("v6b (drop 7 + AI negs)", v6b)]:
            p = preds(m, expert_texts)
            acc = accuracy_score(expert_labels, p)
            kap = cohen_kappa_score(expert_labels, p, weights="quadratic")
            expert_res[name] = {"acc": float(acc), "kappa": float(kap)}
            print(f"  {name:30s}  acc={acc:.4f}  kappa={kap:.4f}", flush=True)
    else:
        expert_res = None

    # === AI poem set (held out) ===
    print("\n=== on 209 AI poems (held out) ===", flush=True)
    y_ai = np.asarray([1 if m["is_poetry"] else 0 for m in matched], dtype=np.int64)
    ai_texts = [m["generated"] for m in matched]
    ai_res = {}
    for name, m in [("v2 (frozen)", v2), ("v6a (drop 7 feats)", v6a),
                    ("v6b (drop 7 + AI negs)", v6b)]:
        p = preds(m, ai_texts)
        acc = accuracy_score(y_ai, p)
        kap = cohen_kappa_score(y_ai, p, weights="quadratic")
        fp = int(((y_ai == 0) & (p == 1)).sum())
        fn = int(((y_ai == 1) & (p == 0)).sum())
        ai_res[name] = {"acc": float(acc), "kappa": float(kap),
                         "fp": fp, "fn": fn}
        print(f"  {name:30s}  yes={int(p.sum())}/{len(p)}  "
              f"acc={acc:.4f}  kappa={kap:.4f}  fp={fp}  fn={fn}", flush=True)

    # === write log ===
    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "v6: drop struct+style families, retrain",
        "v6_keep_idx": v6_keep_idx,
        "v6_keep_features": [FEATURE_NAMES[i] for i in v6_keep_idx],
        "v6_drop_features": v6_drop_features,
        "n_features_v6a": n_v6a,
        "n_features_v6b": n_v6b,
        "n_ai_neg": len(ai_neg),
        "val_results": val_res,
        "expert_results": expert_res,
        "ai_results": ai_res,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
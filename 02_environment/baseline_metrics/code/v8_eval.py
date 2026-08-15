"""v8: train with semantic-unit theme + bge-entity imagery similarity.

New vs v7:
  - theme8_* family (5): theme on semantic units (merged short lines)
  - img_* family now uses bge-small-zh entity similarity (v8)
  - 6 semantic features unchanged

Evaluation on val / expert / AI-poem sets.
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

ROUND_ID = 16
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


def class_char_counts_from_texts(texts, n):
    df, total = Counter(), 0
    for t in texts:
        han = "".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams); total += len(grams)
    return df, total


def train_metric(samples, extra_neg=None, seed=42, use_semantic=True):
    train, _ = train_val_split(samples, val_ratio=0.2, seed=seed)
    train_texts = [s.text for s in train]
    labels = [s.label for s in train]
    if extra_neg:
        train_texts += extra_neg
        labels += [0] * len(extra_neg)
    X_own = extract_batch(train_texts, use_semantic=use_semantic)
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
        [bleu_self(t, cp, tp), bleu_self(t, cn, tn),
         bigram_jaccard_self(t, cp2), bigram_jaccard_self(t, cn2),
         char_tfidf_cosine_to_poetry_centroid(t, vocab, cent, idf)]
        for t in train_texts
    ], dtype=np.float64)
    X = np.concatenate([X_own, base], axis=1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=seed)
    clf.fit(X_s, y)

    class _M:
        def __init__(s):
            s.clf, s.scaler = clf, scaler
            s.cp, s.cn, s.cp2, s.cn2 = cp, cn, cp2, cn2
            s.vocab, s.cent, s.idf = vocab, cent, idf
            s.tp, s.tn = tp, tn
        def apply(s, text):
            own = extract_batch([text], use_semantic=use_semantic)[0]
            b = np.asarray([
                bleu_self(text, s.cp, s.tp), bleu_self(text, s.cn, s.tn),
                bigram_jaccard_self(text, s.cp2), bigram_jaccard_self(text, s.cn2),
                char_tfidf_cosine_to_poetry_centroid(text, s.vocab, s.cent, s.idf),
            ], dtype=np.float64)
            X1 = np.concatenate([own, b])[None, :]
            p = float(s.clf.predict_proba(s.scaler.transform(X1))[0, 1])
            return p, int(p >= 0.5)
    return _M(), X.shape[1]


def load_expert():
    import re
    with EXPERT_JS.open("r", encoding="utf-8") as f:
        raw = f.read()
    items = []
    for m in re.finditer(r"\{ title: \"([^\"]+)\".*?author: \"([^\"]*)\".*?text: \"((?:[^\"\\]|\\.)*)\".*?genre: \"(poem|nonpoem)\"", raw, re.DOTALL):
        title, author, text, genre = m.groups()
        text = text.replace("\\n", "\n").replace('\\"', '"')
        items.append({"title": title, "author": author, "text": text, "genre": genre})
    return items


def main():
    t0 = time.time()
    print(f"[init] FEATURE_NAMES = {len(FEATURE_NAMES)} features", flush=True)
    samples = load_v2()
    _, val = train_val_split(samples, val_ratio=0.2, seed=42)
    val_texts = [s.text for s in val]
    y_val = np.asarray([s.label for s in val], dtype=np.int64)

    # v2 frozen baseline
    print("[v2] frozen ...", flush=True)
    v2 = build_and_freeze(seed=42, val_ratio=0.2)

    # AI negatives
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

    print("\n[v8a] train (all v8 features, no AI neg) ...", flush=True)
    v8a, n8a = train_metric(samples, seed=42, use_semantic=True)
    print(f"  [v8a] done, {n8a} dims, {time.time()-t0:.1f}s", flush=True)
    print("[v8b] train (+ AI neg) ...", flush=True)
    v8b, n8b = train_metric(samples, extra_neg=ai_neg, seed=42, use_semantic=True)
    print(f"  [v8b] done, {n8b} dims, {time.time()-t0:.1f}s", flush=True)

    def preds(metric, texts):
        out = []
        for t in texts:
            r = metric.apply(t)
            out.append(r[1] if isinstance(r, tuple) else r.pred)
        return np.asarray(out, dtype=np.int64)

    print("\n=== val split ===", flush=True)
    val_res = {}
    for name, m in [("v2 (frozen)", v2), ("v8a", v8a), ("v8b (+AI neg)", v8b)]:
        p = preds(m, val_texts)
        acc = accuracy_score(y_val, p)
        kap = cohen_kappa_score(y_val, p, weights="quadratic")
        val_res[name] = {"acc": float(acc), "kappa": float(kap)}
        print(f"  {name:20s}  acc={acc:.4f}  kappa={kap:.4f}", flush=True)

    print("\n=== expert set ===", flush=True)
    expert = load_expert()
    e_texts = [it["text"] for it in expert]
    e_labels = np.asarray([1 if it["genre"] == "poem" else 0 for it in expert])
    expert_res = {}
    for name, m in [("v2 (frozen)", v2), ("v8a", v8a), ("v8b (+AI neg)", v8b)]:
        p = preds(m, e_texts)
        acc = accuracy_score(e_labels, p)
        kap = cohen_kappa_score(e_labels, p, weights="quadratic")
        expert_res[name] = {"acc": float(acc), "kappa": float(kap)}
        print(f"  {name:20s}  acc={acc:.4f}  kappa={kap:.4f}", flush=True)

    print("\n=== AI poem set ===", flush=True)
    y_ai = np.asarray([1 if m["is_poetry"] else 0 for m in matched])
    ai_texts = [m["generated"] for m in matched]
    ai_res = {}
    for name, m in [("v2 (frozen)", v2), ("v8a", v8a), ("v8b (+AI neg)", v8b)]:
        p = preds(m, ai_texts)
        acc = accuracy_score(y_ai, p)
        kap = cohen_kappa_score(y_ai, p, weights="quadratic")
        fp = int(((y_ai == 0) & (p == 1)).sum())
        fn = int(((y_ai == 1) & (p == 0)).sum())
        ai_res[name] = {"acc": float(acc), "kappa": float(kap), "fp": fp, "fn": fn}
        print(f"  {name:20s}  yes={int(p.sum())}/{len(p)}  acc={acc:.4f}  "
              f"kappa={kap:.4f}  fp={fp}  fn={fn}", flush=True)

    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_features": len(FEATURE_NAMES) + 5,
        "v8_changes": ["theme8 semantic units", "img bge entity sim"],
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
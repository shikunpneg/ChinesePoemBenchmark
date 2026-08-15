"""Diagnose v6b expert set regression: which samples did it flip?"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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
    class_char_counts, load_v2 as _load_v2, train_val_split,
)
load_v2 = _load_v2  # avoid shadowing

EXPERT_JS = Path(r"E:\生成诗歌\eval-annotation\data\samples.js")
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / "stage2_round_014.json"

DROP = ("struct_", "style_")
v6_keep_idx = [i for i, f in enumerate(FEATURE_NAMES) if not f.startswith(DROP)]


def class_char_counts_from_texts(texts, n):
    from collections import Counter
    df, total = Counter(), 0
    for t in texts:
        han = "".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams); total += len(grams)
    return df, total


def main():
    # Load expert
    with EXPERT_JS.open("r", encoding="utf-8") as f:
        raw = f.read()
    items = []
    for m in re.finditer(r"\{ title: \"([^\"]+)\".*?author: \"([^\"]*)\".*?text: \"((?:[^\"\\]|\\.)*)\".*?genre: \"(poem|nonpoem)\"", raw, re.DOTALL):
        title, author, text, genre = m.groups()
        text = text.replace("\\n", "\n").replace('\\"', '"')
        items.append({"title": title, "author": author, "text": text, "genre": genre})
    print(f"loaded {len(items)} expert items")

    texts = [it["text"] for it in items]
    labels = np.asarray([1 if it["genre"] == "poem" else 0 for it in items])

    # v2 vs v6b
    v2 = build_and_freeze(seed=42, val_ratio=0.2)
    samples = load_v2()
    train, _ = train_val_split(samples, val_ratio=0.2, seed=42)
    train_texts = [s.text for s in train]
    # add AI neg
    log = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    print(f"v6_keep_idx: {v6_keep_idx}")
    print(f"v6 dropped: {[FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if i not in v6_keep_idx]}")

    # v2 preds
    v2_preds = []
    for t in texts:
        p = v2.apply(t)
        v2_preds.append(p.pred)
    v2_preds = np.asarray(v2_preds)
    # v6b preds (re-train)
    # need to retrain v6b
    train_texts_v6b = list(train_texts)
    labels_v6b = [s.label for s in train]
    # need to reconstruct AI neg
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
    train_texts_v6b += ai_neg
    labels_v6b += [0] * len(ai_neg)

    X_full = extract_batch(train_texts_v6b)
    X = X_full[:, v6_keep_idx]
    y = np.asarray(labels_v6b, dtype=np.int64)
    poems = [t for t, l in zip(train_texts_v6b, labels_v6b) if l == 1]
    nonpoems = [t for t, l in zip(train_texts_v6b, labels_v6b) if l == 0]
    cp, tp = class_char_counts_from_texts(poems, 1)
    cn, tn = class_char_counts_from_texts(nonpoems, 1)
    cp2, _ = class_char_counts_from_texts(poems, 2)
    cn2, _ = class_char_counts_from_texts(nonpoems, 2)
    vocab = build_char_vocab(train_texts_v6b, min_df=2)
    cent, idf = build_tfidf_centroid(poems, vocab)
    base = np.asarray([
        [bleu_self(t, cp, tp), bleu_self(t, cn, tn),
         bigram_jaccard_self(t, cp2), bigram_jaccard_self(t, cn2),
         char_tfidf_cosine_to_poetry_centroid(t, vocab, cent, idf)]
        for t in texts
    ], dtype=np.float64)
    X_eval_own = extract_batch(texts)[:, v6_keep_idx]
    X_eval = np.concatenate([X_eval_own, base], axis=1)
    # refit scaler on the v6b training
    X_train_combined = np.concatenate([
        extract_batch(train_texts_v6b)[:, v6_keep_idx],
        np.asarray([
            [bleu_self(t, cp, tp), bleu_self(t, cn, tn),
             bigram_jaccard_self(t, cp2), bigram_jaccard_self(t, cn2),
             char_tfidf_cosine_to_poetry_centroid(t, vocab, cent, idf)]
            for t in train_texts_v6b
        ], dtype=np.float64)
    ], axis=1)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_combined)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=42)
    clf.fit(X_train_s, y)
    v6b_preds = clf.predict(X_eval)
    v6b_probs = clf.predict_proba(X_eval)[:, 1]

    # Now compare v2 vs v6b on expert set
    print(f"\n=== v2 vs v6b on expert set (n=100) ===")
    print(f"v2 acc: {(v2_preds == labels).mean():.4f}  kappa: "
          f"{2*(v2_preds*labels).sum() / (v2_preds**2 + labels**2).sum() - 0.5:.4f}")  # rough
    from sklearn.metrics import cohen_kappa_score
    print(f"v2: kappa={cohen_kappa_score(labels, v2_preds, weights='quadratic'):.4f}")
    print(f"v6b: kappa={cohen_kappa_score(labels, v6b_preds, weights='quadratic'):.4f}")

    # Find samples where v2 and v6b disagree
    disagree_idx = np.where(v2_preds != v6b_preds)[0]
    print(f"\nv2 vs v6b disagree on {len(disagree_idx)} samples:")
    for i in disagree_idx:
        it = items[i]
        print(f"  [{i}] label={labels[i]} v2={v2_preds[i]} v6b={v6b_preds[i]} v6b_prob={v6b_probs[i]:.3f}  "
              f"title={it['title']!r} author={it['author']!r}")
        if it['title']: print(f"      text preview: {it['text'][:80].replace(chr(10),' ')}...")

    # per-strat
    print(f"\n=== per-author breakdown (expert set) ===")
    from collections import defaultdict
    by_author = defaultdict(list)
    for i, it in enumerate(items):
        by_author[it['author']].append((labels[i], v2_preds[i], v6b_preds[i]))
    for author, votes in by_author.items():
        n = len(votes)
        v2_acc = sum(1 for l, v2, v6b in votes if v2 == l) / n
        v6b_acc = sum(1 for l, v2, v6b in votes if v6b == l) / n
        print(f"  {author:10s}  n={n}  v2_acc={v2_acc:.3f}  v6b_acc={v6b_acc:.3f}")


if __name__ == "__main__":
    sys.exit(main())
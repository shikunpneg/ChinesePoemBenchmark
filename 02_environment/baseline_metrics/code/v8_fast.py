"""v8 fast: only v8b (+AI neg), write results to file (no pipe buffering)."""
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
RUN_LOG = ARTIFACTS / "run.log"

logf = open(RUN_LOG, "w", encoding="utf-8")
def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


def cc(texts, n):
    df, total = Counter(), 0
    for t in texts:
        han = "".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams); total += len(grams)
    return df, total


def main():
    t0 = time.time()
    log(f"[init] FEATURE_NAMES={len(FEATURE_NAMES)}")
    samples = load_v2()
    _, val = train_val_split(samples, val_ratio=0.2, seed=42)
    val_texts = [s.text for s in val]
    y_val = np.asarray([s.label for s in val], dtype=np.int64)

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
    log(f"[load] ai_neg={len(ai_neg)}")

    # v8b: train with AI negs
    train, _ = train_val_split(samples, val_ratio=0.2, seed=42)
    train_texts = [s.text for s in train] + ai_neg
    labels = [s.label for s in train] + [0] * len(ai_neg)
    log("[feat] extracting train features ...")
    tf = time.time()
    X_own = extract_batch(train_texts, use_semantic=True)
    log(f"  own features {X_own.shape} in {time.time()-tf:.0f}s")
    y = np.asarray(labels, dtype=np.int64)
    poems = [t for t, l in zip(train_texts, labels) if l == 1]
    nonpoems = [t for t, l in zip(train_texts, labels) if l == 0]
    cp, tp = cc(poems, 1)
    cn, tn = cc(nonpoems, 1)
    cp2, _ = cc(poems, 2)
    cn2, _ = cc(nonpoems, 2)
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
                             random_state=42)
    clf.fit(X_s, y)
    log(f"[train] LR done, {X.shape[1]} dims, {time.time()-t0:.0f}s")

    def preds(texts):
        out = []
        for i, t in enumerate(texts):
            if i % 100 == 0:
                log(f"  predict {i}/{len(texts)}")
            own = extract_batch([t], use_semantic=True)[0]
            b = np.asarray([
                bleu_self(t, cp, tp), bleu_self(t, cn, tn),
                bigram_jaccard_self(t, cp2), bigram_jaccard_self(t, cn2),
                char_tfidf_cosine_to_poetry_centroid(t, vocab, cent, idf),
            ], dtype=np.float64)
            X1 = np.concatenate([own, b])[None, :]
            p = clf.predict(scaler.transform(X1))[0]
            out.append(int(p))
        return np.asarray(out, dtype=np.int64)

    log("=== val ===")
    pv = preds(val_texts)
    log(f"  val acc={accuracy_score(y_val, pv):.4f} kappa={cohen_kappa_score(y_val, pv, weights='quadratic'):.4f}")

    log("=== expert ===")
    import re
    with EXPERT_JS.open("r", encoding="utf-8") as f:
        raw = f.read()
    expert = []
    for m in re.finditer(r"\{ title: \"([^\"]+)\".*?author: \"([^\"]*)\".*?text: \"((?:[^\"\\]|\\.)*)\".*?genre: \"(poem|nonpoem)\"", raw, re.DOTALL):
        title, author, text, genre = m.groups()
        text = text.replace("\\n", "\n").replace('\\"', '"')
        expert.append((text, 1 if genre == "poem" else 0))
    e_texts = [e[0] for e in expert]
    e_labels = np.asarray([e[1] for e in expert])
    pe = preds(e_texts)
    log(f"  expert acc={accuracy_score(e_labels, pe):.4f} kappa={cohen_kappa_score(e_labels, pe, weights='quadratic'):.4f}")

    log("=== AI set ===")
    y_ai = np.asarray([1 if m["is_poetry"] else 0 for m in matched])
    ai_texts = [m["generated"] for m in matched]
    pa = preds(ai_texts)
    acc = accuracy_score(y_ai, pa)
    kap = cohen_kappa_score(y_ai, pa, weights="quadratic")
    fp = int(((y_ai == 0) & (pa == 1)).sum())
    fn = int(((y_ai == 1) & (pa == 0)).sum())
    log(f"  AI acc={acc:.4f} kappa={kap:.4f} fp={fp} fn={fn} yes={int(pa.sum())}/{len(pa)}")

    LOG_PATH.write_text(json.dumps({
        "stage": "stage2", "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_features": len(FEATURE_NAMES) + 5,
        "val": {"acc": float(accuracy_score(y_val, pv)),
                "kappa": float(cohen_kappa_score(y_val, pv, weights="quadratic"))},
        "expert": {"acc": float(accuracy_score(e_labels, pe)),
                   "kappa": float(cohen_kappa_score(e_labels, pe, weights="quadratic"))},
        "ai": {"acc": float(acc), "kappa": float(kap), "fp": int(fp), "fn": int(fn)},
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[done] elapsed {time.time()-t0:.0f}s")
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
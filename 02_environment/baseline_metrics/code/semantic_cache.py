"""Pre-compute and cache semantic features for all datasets.

Semantic encoding with bge-small-zh on CPU is slow (~10s per 100 lines).
To avoid recomputing every training run, we cache the per-text semantic
feature vector (6 floats) to a JSON/npz file keyed by text hash.

This script:
  1. Loads v2 corpus (1855), expert set (100), AI poem set (209)
  2. Computes semantic features for ALL unique texts once
  3. Saves to 07_reproducibility/semantic_cache.npz
"""
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code.data_loader_v2 import load_v2  # noqa: E402
from code.semantic import semantic_features_batch, _lines  # noqa: E402

EXPORT6 = Path(r"E:\生成诗歌\annotations_export6.csv")
HARD_FILES = [
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl"),
]
EXPERT_JS = Path(r"E:\生成诗歌\eval-annotation\data\samples.js")
CACHE = ROOT / "07_reproducibility" / "semantic_cache.npz"

SEM_KEYS = ("sem_adj_line_sim_mean", "sem_adj_line_sim_cv",
            "sem_first_last_sim", "sem_bridge_rate",
            "sem_wholeness", "sem_dispersion")


def text_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def collect_texts():
    texts = []
    seen = set()
    def add(t):
        if t and t.strip() and t not in seen:
            seen.add(t)
            texts.append(t)
    for s in load_v2():
        add(s.text)
    # expert
    with EXPERT_JS.open("r", encoding="utf-8") as f:
        raw = f.read()
    for m in re.finditer(r"text: \"((?:[^\"\\]|\\.)*)\"", raw, re.DOTALL):
        add(m.group(1).replace("\\n", "\n").replace('\\"', '"'))
    # AI poems
    for path in HARD_FILES:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                add(obj.get("generated", ""))
    return texts


def main():
    t0 = time.time()
    texts = collect_texts()
    print(f"[collect] {len(texts)} unique texts", flush=True)
    hashes = [text_hash(t) for t in texts]

    # load existing cache if any
    cache = {}
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        cache = dict(zip(data["hashes"], data["feats"]))
        print(f"[cache] loaded {len(cache)} cached entries", flush=True)

    to_compute = [(h, t) for h, t in zip(hashes, texts) if h not in cache]
    print(f"[compute] need {len(to_compute)} new", flush=True)
    if to_compute:
        batch_texts = [t for _, t in to_compute]
        # encode all lines of all new texts in one giant batch
        all_lines = []
        line_owner = []  # index into batch_texts
        for bi, t in enumerate(batch_texts):
            for ln in _lines(t):
                all_lines.append(ln)
                line_owner.append(bi)
        print(f"[encode] {len(all_lines)} total lines ...", flush=True)
        from code.semantic import encode_texts
        vecs = encode_texts(all_lines, batch_size=256)
        # group back
        from code.semantic import _cos
        per_text = {}
        for (bi, v) in zip(line_owner, vecs):
            per_text.setdefault(bi, []).append(v)
        feats_list = []
        for bi in range(len(batch_texts)):
            arr = np.stack(per_text.get(bi, [])) if per_text.get(bi) else None
            if arr is None or len(arr) < 2:
                feats_list.append(np.zeros(6, dtype=np.float64))
                continue
            n = len(arr)
            sims = [_cos(arr[i], arr[i + 1]) for i in range(n - 1)]
            mean_sim = float(np.mean(sims)) if sims else 0.0
            std_sim = float(np.std(sims)) if sims else 0.0
            cv_sim = std_sim / mean_sim if mean_sim > 0 else 0.0
            first_last = _cos(arr[0], arr[-1])
            centroid = arr.mean(axis=0)
            if np.linalg.norm(centroid) == 0:
                bridge = 0.0
            else:
                centroid = centroid / np.linalg.norm(centroid)
                rupture = [i for i in range(n - 1) if sims[i] < 0.3]
                if rupture:
                    deep = sum(1 for i in rupture
                               if _cos(arr[i], centroid) > 0.15
                               or _cos(arr[i + 1], centroid) > 0.15)
                    bridge = deep / len(rupture)
                else:
                    bridge = 1.0
            wholeness = float(np.clip(0.4 * mean_sim + 0.3 * bridge
                                      + 0.3 * first_last, 0, 1))
            feats_list.append(np.asarray(
                [mean_sim, cv_sim, first_last, bridge, wholeness,
                 1.0 - wholeness], dtype=np.float64))
        for (h, _), f in zip(to_compute, feats_list):
            cache[h] = f
        print(f"[compute] done in {time.time()-t0:.1f}s", flush=True)

    # save
    hs = np.array(list(cache.keys()))
    fs = np.stack(list(cache.values()))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, hashes=hs, feats=fs)
    print(f"[save] {len(cache)} entries -> {CACHE}", flush=True)
    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
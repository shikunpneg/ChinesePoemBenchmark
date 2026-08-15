"""Pre-compute and cache bge embeddings for all entity words.

Same pattern as `semantic_cache.py`. The imagery_ner v8 module
looks up entity pairs in this cache (process-in-memory dict).

We collect all entity words from v2 corpus + expert + AI poem set.
Total unique entities: a few hundred. bge-small-zh encodes ~1000 words
in <2 seconds on CPU.
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code.data_loader_v2 import load_v2
from code.imagery_ner import extract_entities

CACHE = ROOT / "07_reproducibility" / "entity_vec_cache.npz"


def collect_entities():
    """Collect all unique entity words from the dataset."""
    entities = set()
    for s in load_v2():
        for e in extract_entities(s.text).entities:
            entities.add(e)
    return sorted(entities)


def main():
    t0 = time.time()
    ents = collect_entities()
    print(f"[collect] {len(ents)} unique entities", flush=True)

    # load existing cache
    cache = {}
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        cache = dict(zip(data["words"], data["vecs"]))
        print(f"[cache] loaded {len(cache)} entries", flush=True)

    to_compute = [w for w in ents if w not in cache]
    print(f"[compute] need {len(to_compute)} new", flush=True)
    if to_compute:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        vecs = model.encode(to_compute, normalize_embeddings=True,
                            batch_size=128, show_progress_bar=False,
                            convert_to_numpy=True)
        for w, v in zip(to_compute, vecs):
            cache[w] = v
        print(f"[encode] done in {time.time()-t0:.1f}s", flush=True)

    # save
    ws = list(cache.keys())
    vs = np.stack(list(cache.values()))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, words=np.array(ws), vecs=vs)
    print(f"[save] {len(cache)} entries -> {CACHE}", flush=True)
    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
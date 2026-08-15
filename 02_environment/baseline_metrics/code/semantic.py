"""Semantic embedding features using sentence-transformers (bge-small-zh).

This is the CORE upgrade addressing the user's feedback:
  - "没有任何语义向量，我们有gpu" — we now use real semantic embeddings.
  - Provides line/paragraph-level semantic similarity that the shallow
    char-overlap features cannot capture (e.g., 明月 vs 霜, 舟 vs 帆).

Model: BAAI/bge-small-zh-v1.5 (~24M params, CPU-friendly, GPU-optional).

We implement:
  1. `SemanticModel` — lazy-loaded singleton; `encode(texts)` returns vectors.
  2. `semantic_features(text)` — the 5+1 "断裂-引力" / wholeness features:
       - sem_adj_line_sim_mean:  mean cosine between adjacent lines
       - sem_adj_line_sim_cv:    CV of adjacent-line similarity
       - sem_first_last_sim:     first vs last line (首尾呼应)
       - sem_bridge_rate:        fraction of low-sim (rupture) adjacent
                                 pairs that share a TOPIC cluster
       - sem_wholeness:          composite: how "one poem" it is
       - sem_dispersion:         1 - wholeness (how scattered)

  The bridge/wholeness logic follows the original research intuition:
    random text:  adj_sim low, CV high, wholeness low
    prose:        adj_sim mid, CV low, wholeness mid
    poetry:       adj_sim mid-low but BRIDGED, CV mid, wholeness HIGH
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# lazy import so the rest of the project can run without sentence-transformers
_sent_tf = None
_model = None
_model_name = "BAAI/bge-small-zh-v1.5"

# disk cache of per-text semantic features (computed once, reused)
# semantic.py -> code/ -> baseline_metrics/ -> 02_environment/ -> project root
_CACHE_PATH = Path(__file__).resolve().parents[3] / "07_reproducibility" / "semantic_cache.npz"
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if _CACHE_PATH.exists():
            try:
                data = np.load(_CACHE_PATH, allow_pickle=True)
                hashes = data["hashes"]
                feats = data["feats"]
                for h, f in zip(hashes, feats):
                    _cache[str(h)] = f
            except Exception:
                _cache = {}
    return _cache


def _text_hash(t: str) -> str:
    import hashlib
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _get_model():
    global _sent_tf, _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _sent_tf = SentenceTransformer
        _model = SentenceTransformer(_model_name)
    return _model


def encode_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode list of texts into a (n, dim) numpy array."""
    model = _get_model()
    vecs = model.encode(texts, batch_size=batch_size,
                        show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def _lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compute_sem_vecs(lines: list[str]) -> np.ndarray | None:
    """Encode lines, returning None on failure."""
    try:
        return encode_texts(lines)
    except Exception:
        return None


def semantic_features(text: str, verbose: bool = False) -> dict[str, float]:
    """Compute semantic structure features for one text.

    Falls back to all-zeros if the model cannot be loaded (CPU-only,
    no download, etc.) so the rest of the pipeline never breaks.
    """
    lines = _lines(text)
    if len(lines) < 2:
        return {"sem_adj_line_sim_mean": 0.0, "sem_adj_line_sim_cv": 0.0,
                "sem_first_last_sim": 0.0, "sem_bridge_rate": 0.0,
                "sem_wholeness": 0.0, "sem_dispersion": 0.0,
                "sem_available": 0.0}
    vecs = _compute_sem_vecs(lines)
    if vecs is None:
        return {"sem_adj_line_sim_mean": 0.0, "sem_adj_line_sim_cv": 0.0,
                "sem_first_last_sim": 0.0, "sem_bridge_rate": 0.0,
                "sem_wholeness": 0.0, "sem_dispersion": 0.0,
                "sem_available": 0.0}

    n = len(vecs)
    sims = []
    for i in range(n - 1):
        sims.append(_cos(vecs[i], vecs[i + 1]))
    mean_sim = float(np.mean(sims)) if sims else 0.0
    std_sim = float(np.std(sims)) if sims else 0.0
    cv_sim = std_sim / mean_sim if mean_sim > 0 else 0.0

    first_last = _cos(vecs[0], vecs[-1])

    # bridge: among low-sim (rupture) adjacent pairs, how many still
    # correlate above 0.15 with the poem's centroid (deep linkage)?
    centroid = vecs.mean(axis=0)
    if np.linalg.norm(centroid) == 0:
        bridge = 0.0
    else:
        centroid = centroid / np.linalg.norm(centroid)
        rupture_pairs = [i for i in range(n - 1) if sims[i] < 0.3]
        if rupture_pairs:
            deep = sum(1 for i in rupture_pairs
                       if _cos(vecs[i], centroid) > 0.15
                       or _cos(vecs[i + 1], centroid) > 0.15)
            bridge = deep / len(rupture_pairs)
        else:
            bridge = 1.0  # no ruptures -> perfectly bridged

    # wholeness composite
    wholeness = 0.4 * mean_sim + 0.3 * bridge + 0.3 * first_last
    wholeness = float(np.clip(wholeness, 0, 1))

    return {
        "sem_adj_line_sim_mean": float(mean_sim),
        "sem_adj_line_sim_cv": float(cv_sim),
        "sem_first_last_sim": float(first_last),
        "sem_bridge_rate": float(bridge),
        "sem_wholeness": float(wholeness),
        "sem_dispersion": float(1.0 - wholeness),
        "sem_available": 1.0,
    }


def semantic_features_batch(texts: list[str],
                            batch_size: int = 64) -> list[list[float]]:
    """Compute semantic features for many texts efficiently.

    Checks the disk cache first; encodes only texts not cached.
    Returns list of 6 floats per text.
    """
    if not texts:
        return []
    cache = _load_cache()
    results = []
    to_compute_idx = []
    to_compute_texts = []
    for ti, t in enumerate(texts):
        h = _text_hash(t)
        if h in cache:
            results.append([float(x) for x in cache[h][:6]])
        else:
            results.append(None)
            to_compute_idx.append(ti)
            to_compute_texts.append(t)
    if to_compute_texts:
        new_feats = _compute_batch_uncached(to_compute_texts, batch_size)
        for idx, feats in zip(to_compute_idx, new_feats):
            results[idx] = feats
            cache[_text_hash(texts[idx])] = np.asarray(feats, dtype=np.float64)
    return results


def semantic_available() -> bool:
    """True if the model is loadable (or cache covers everything)."""
    return len(_load_cache()) > 0 or _get_model() is not None


def _compute_batch_uncached(texts: list[str],
                            batch_size: int = 64) -> list[list[float]]:
    """Raw compute (no cache) for a batch of texts."""
    if not texts:
        return []
    # short-circuit: if only 1 line per text, semantic features are zero
    if all(len(_lines(t)) < 2 for t in texts):
        return [[0.0] * 6 for _ in texts]
    # collect (text_idx, line_idx, line_text)
    groups: list[list[str]] = []
    line_to_group: list[tuple[int, int]] = []
    all_lines: list[str] = []
    for ti, t in enumerate(texts):
        lines = _lines(t)
        groups.append(lines)
        for li, ln in enumerate(lines):
            line_to_group.append((ti, li))
            all_lines.append(ln)
    if not all_lines:
        return [[0.0] * 6 for _ in texts]

    vecs = _compute_sem_vecs(all_lines)
    if vecs is None:
        return [[0.0] * 6 for _ in texts]

    # group back per text
    per_text_vecs: dict[int, list[np.ndarray]] = {}
    for (ti, li), v in zip(line_to_group, vecs):
        per_text_vecs.setdefault(ti, []).append(v)

    out = []
    for ti in range(len(texts)):
        vlist = per_text_vecs.get(ti, [])
        if len(vlist) < 2:
            out.append([0.0] * 6)
            continue
        arr = np.stack(vlist)
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
        out.append([mean_sim, cv_sim, first_last, bridge,
                    wholeness, 1.0 - wholeness])
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("loading bge-small-zh (first call downloads model)...")
    poem = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
    r = semantic_features(poem, verbose=True)
    print(f"poem (静夜思): {r}  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    news = ("央行宣布降准0.5个百分点，释放长期资金约1万亿元\n"
            "市场分析认为，此举有助于降低实体经济融资成本\n"
            "多家上市银行公布三季度财报，业绩普遍超预期")
    r2 = semantic_features(news, verbose=True)
    print(f"news: {r2}  [{time.time()-t0:.1f}s]")

    t0 = time.time()
    modern = "小巷\n又弯又长\n没有门\n没有窗\n我拿把旧钥匙\n敲着厚厚的墙"
    r3 = semantic_features(modern, verbose=True)
    print(f"顾城: {r3}  [{time.time()-t0:.1f}s]")
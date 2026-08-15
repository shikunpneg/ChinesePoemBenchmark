"""Trivial baselines (方案 §3.2) — implemented with pure stdlib + numpy.

These intentionally use **no** learned models, so they are reproducible
without any GPU / API / dependency beyond jieba + pypinyin.

Implemented:
    - bleu_self       : 1-gram BLEU of one text against a reference corpus
                        (here: vs the user's own class centroid)
    - rouge_l_self    : ROUGE-L F1 of one text vs the class centroid
    - char_tfidf_cosine_to_poetry_centroid : char-level TF-IDF cosine to
                        the centroid of all poetry training texts.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


# --- shared helpers ------------------------------------------------------

def _char_ngrams(text: str, n: int) -> Counter:
    """Return char n-gram counts (Han-only)."""
    han = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    if len(han) < n:
        return Counter()
    return Counter(han[i:i + n] for i in range(len(han) - n + 1))


def _char_jaccard(a: Counter, b: Counter) -> float:
    """Jaccard similarity of two char-n-gram Counters. O(|union|)."""
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())  # Counter intersection: min per key
    union = sum((a | b).values())
    return inter / union if union else 0.0


def _lcs_len(a: list, b: list) -> int:
    """Standard LCS length, O(len(a)*len(b)) DP.

    NOTE: caller is responsible for keeping inputs small (<= 10000 chars).
    Provided for completeness; in round 1 we replace ROUGE-L with
    char-bigram Jaccard (faster, similar signal).
    """
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    if n > 12000 or m > 12000:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[m]


# --- BLEU-1 (corpus-relative) -------------------------------------------

def bleu_self(text: str, ref_counts: Counter, ref_total: int) -> float:
    """1-gram BLEU (no brevity penalty) vs a precomputed reference corpus.

    Args:
        text: candidate text.
        ref_counts: aggregated n-gram counts of the reference corpus.
        ref_total: total n-gram count of the reference corpus.
    """
    cand = _char_ngrams(text, 1)
    if not cand or ref_total == 0:
        return 0.0
    overlap = 0
    for ng, c in cand.items():
        overlap += min(c, ref_counts.get(ng, 0))
    cand_total = sum(cand.values())
    return overlap / cand_total if cand_total else 0.0


# --- ROUGE-L (corpus-relative) ------------------------------------------

def rouge_l_self(text: str, ref_tokens: list[str]) -> float:
    """ROUGE-L F1 vs a single token sequence (here: class centroid char string).

    SLOW on long inputs (O(n*m) DP). For round 1 we recommend using
    `bigram_jaccard_self` instead.
    """
    cand_tokens = list("".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff"))
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(cand_tokens)
    r = lcs / len(ref_tokens)
    return 2 * p * r / (p + r)


# --- bigram Jaccard (faster ROUGE-L proxy) ------------------------------

def bigram_jaccard_self(text: str, ref_counts: Counter) -> float:
    """Char-bigram Jaccard similarity vs the reference corpus counts.

    Faster than ROUGE-L on long texts (no DP); information-wise similar
    for character-level overlap.
    """
    cand = _char_ngrams(text, 2)
    if not cand or not ref_counts:
        return 0.0
    inter = 0
    cand_total = sum(cand.values())
    ref_total = sum(ref_counts.values())
    for ng, c in cand.items():
        # cap by ref's count (same as BLEU clipping)
        inter += min(c, ref_counts.get(ng, 0))
    union = cand_total + ref_total - inter
    return inter / union if union else 0.0


# --- char-TFIDF centroid -------------------------------------------------

def build_char_vocab(train_texts: Iterable[str], min_df: int = 2):
    """Build a {char -> index} mapping restricted to chars seen >= min_df times."""
    df: Counter = Counter()
    for t in train_texts:
        chars = set("".join(ch for ch in t if "\u4e00" <= ch <= "\u9fff"))
        df.update(chars)
    vocab = {ch: i for i, ch in enumerate(sorted(c for c, d in df.items() if d >= min_df))}
    return vocab


def char_counts(text: str, vocab: dict[str, int]) -> np.ndarray:
    """Return a sparse count vector over `vocab`."""
    v = np.zeros(len(vocab), dtype=np.float64)
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            idx = vocab.get(ch)
            if idx is not None:
                v[idx] += 1.0
    return v


def build_tfidf_centroid(
    texts: Iterable[str],
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TF-IDF centroid of all `texts` over the given vocab.

    Returns (centroid_vector, idf_array).
    """
    texts = list(texts)
    if not texts:
        z = np.zeros(len(vocab))
        return z, z.copy()
    df: Counter = Counter()
    counts = []
    for t in texts:
        c = char_counts(t, vocab)
        counts.append(c)
        # each text contributes 1 to df for each present char
        present = np.flatnonzero(c)
        for i in present:
            df[i] += 1
    n_docs = len(texts)
    idf_arr = np.zeros(len(vocab), dtype=np.float64)
    for i, d in df.items():
        idf_arr[i] = np.log((1 + n_docs) / (1 + d)) + 1.0
    # convert counts to tfidf
    tfidf = []
    for c in counts:
        # simple tf = 1 + log(count) if count > 0 else 0
        tf = np.where(c > 0, 1.0 + np.log1p(c), 0.0)
        tfidf.append(tf * idf_arr)
    centroid = np.mean(np.stack(tfidf, axis=0), axis=0)
    # L2-normalize
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid, idf_arr


def char_tfidf_cosine_to_poetry_centroid(
    text: str,
    vocab: dict[str, int],
    centroid: np.ndarray,
    idf_arr: np.ndarray,
) -> float:
    """Cosine similarity of `text`'s char-TFIDF to the poetry centroid."""
    c = char_counts(text, vocab)
    tf = np.where(c > 0, 1.0 + np.log1p(c), 0.0)
    v = tf * idf_arr
    norm = np.linalg.norm(v)
    if norm == 0 or np.linalg.norm(centroid) == 0:
        return 0.0
    v = v / norm
    return float(np.dot(v, centroid))
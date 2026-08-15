"""Structure features v2: paragraph statistics + paragraph-theme NLP.

Addresses user feedback on `struct` family:
  - current struct only counts lines/punct/short-line ratio (too shallow)
  - need: (1) paragraph segmentation statistics
         (2) NLP-based per-paragraph THEME summarization to capture the
             underlying logic of the structure

Theme representation without external embeddings:
  - per-paragraph keyword vector (jieba content-word TF)
  - paragraph-theme change = cosine distance between adjacent paragraphs'
    keyword vectors (high = theme jumps; prose: low; poetry: mid-high but
    coherent at the imagery level — captured by bridge features elsewhere)
  - theme concentration = how few distinct theme clusters the poem has
"""

from __future__ import annotations

from collections import Counter

import jieba
import numpy as np

jieba.setLogLevel(20)

# stop words for keyword extraction (content-word filter)
_STOPWORDS = set(
    "的了着和与及或等之性化因为所以但是然而并且而且虽然即使如果则因此于是"
    "然后接着此外另外同时同样其实不过可是尽管无论不管只要既然以及没有不是"
    "这个那个这些那些什么怎么怎么样我们你们他们它们我的你的他的她的它的"
    "一一个一次有的在又都也就要就能会可以应该可能还是或是就是而是但是"
    "很非常太十分多么真挺最更再还也哦嗯啊吧呢吗呀嘛呵呵哈哈"
)


def _content_words(text: str) -> list[str]:
    """jieba tokens that are content words (len>=2, not stopword)."""
    words = [w for w in jieba.lcut(text) if w.strip()]
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS]


def _kw_vector(words: list[str], vocab: dict[str, int] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """TF vector over words; returns (vec, vocab_used)."""
    if vocab is None:
        vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    vec = np.zeros(len(vocab), dtype=np.float64)
    for w in words:
        i = vocab.get(w)
        if i is not None:
            vec[i] += 1.0
    return vec, vocab


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def paragraphs(text: str) -> list[str]:
    """Split by blank lines first; if single block, fall back to lines."""
    blocks = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(blocks) >= 2:
        return blocks
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines if len(lines) >= 2 else ([text] if text.strip() else [])


def paragraph_stats(text: str) -> dict[str, float]:
    """Paragraph-level statistics (v2 struct addition)."""
    paras = paragraphs(text)
    n_paras = len(paras)
    if n_paras == 0:
        return {"para_count": 0.0, "para_len_mean": 0.0, "para_len_cv": 0.0,
                "para_var": 0.0}
    lens = [len(p) for p in paras]
    mean = sum(lens) / len(lens)
    std = (sum((l - mean) ** 2 for l in lens) / len(lens)) ** 0.5
    cv = std / mean if mean else 0.0
    # poems: few paragraphs, short, low cv; prose: many paragraphs, long, high cv
    return {
        "para_count": float(n_paras),
        "para_len_mean": float(min(mean / 60.0, 1.0)),   # normalize: poem ~30, prose ~100+
        "para_len_cv": float(cv),
        "para_var": float(std),
    }


def theme_analysis(text: str) -> dict[str, float]:
    """Theme-level structure features.

    Returns:
      - theme_jump_mean:   mean cosine DISTANCE between adjacent paragraph
                           keyword vectors (0-1; higher = theme jumps more)
      - theme_jump_cv:     CV of adjacent-theme distances (poetry: moderate
                           variation; prose: low; random: high)
      - theme_coherence:   1 - normalized theme entropy (few distinct themes
                           => high coherence = poem-like wholeness)
      - theme_cluster_ratio: fraction of paragraphs that share >=1 content
                           word with the FIRST paragraph (theme persistence)
      - opening_closure:   cosine between first & last paragraph keyword
                           vectors (首尾呼应)
    """
    paras = paragraphs(text)
    if len(paras) < 2:
        return {"theme_jump_mean": 0.0, "theme_jump_cv": 0.0,
                "theme_coherence": 1.0, "theme_cluster_ratio": 1.0,
                "opening_closure": 0.0}

    # global vocab from all paras
    all_words = [w for p in paras for w in _content_words(p)]
    if not all_words:
        return {"theme_jump_mean": 0.0, "theme_jump_cv": 0.0,
                "theme_coherence": 1.0, "theme_cluster_ratio": 1.0,
                "opening_closure": 0.0}
    vocab = {w: i for i, w in enumerate(sorted(set(all_words)))}

    vecs = []
    for p in paras:
        w = _content_words(p)
        v, _ = _kw_vector(w, vocab)
        vecs.append(v)

    # adjacent theme distances
    dists = []
    for i in range(len(vecs) - 1):
        d = 1.0 - _cosine(vecs[i], vecs[i + 1])
        dists.append(d)
    mean_d = float(np.mean(dists)) if dists else 0.0
    cv_d = (float(np.std(dists)) / mean_d) if mean_d > 0 else 0.0

    # theme entropy over global word frequency
    total = len(all_words)
    freq = Counter(all_words)
    ent = -sum((c / total) * np.log2(c / total) for c in freq.values())
    max_ent = np.log2(len(freq)) if len(freq) > 1 else 1.0
    norm_ent = ent / max_ent if max_ent else 0.0
    coherence = 1.0 - min(norm_ent, 1.0)

    # cluster ratio: share >=1 content word with first para
    first_words = set(_content_words(paras[0]))
    if first_words:
        share = sum(1 for p in paras[1:] if set(_content_words(p)) & first_words)
        cluster = share / (len(paras) - 1)
    else:
        cluster = 0.0

    # opening-close closure
    opening_closure = _cosine(vecs[0], vecs[-1])

    return {
        "theme_jump_mean": float(mean_d),
        "theme_jump_cv": float(cv_d),
        "theme_coherence": float(coherence),
        "theme_cluster_ratio": float(cluster),
        "opening_closure": float(opening_closure),
    }


def structure_v2_features(text: str) -> dict[str, float]:
    """Full v2 struct features (paragraph stats + theme analysis)."""
    ps = paragraph_stats(text)
    th = theme_analysis(text)
    return {**{f"para_{k}": v for k, v in ps.items()},
            **{f"theme_{k}": v for k, v in th.items()}}


if __name__ == "__main__":
    print("=== paragraph + theme analysis demo ===")
    poem = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
    print("poem:", structure_v2_features(poem))
    news = ("本报记者报道，昨日央行宣布降准0.5个百分点，释放长期资金约1万亿元。\n"
            "市场分析认为，此举将有助于降低实体经济融资成本。\n"
            "与此同时，多家上市银行公布了三季度财报，业绩普遍超预期。")
    print("news:", structure_v2_features(news))
    # 顾城《小巷》式
    gu = "小巷\n又弯又长\n没有门\n没有窗\n我拿把旧钥匙\n敲着厚厚的墙"
    print("modern poem:", structure_v2_features(gu))
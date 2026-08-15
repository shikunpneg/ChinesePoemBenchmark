"""NER-based imagery extraction + sequential logic analysis (jump v2/v8).

v2: uses char-bigram overlap as entity similarity proxy.
v8: uses bge-small-zh embeddings to compute entity semantic similarity —
    this fixes the "明月 vs 霜" problem (no common chars but semantically close).

NER approach (no external model for extraction):
  - jieba.posseg for POS tags (n=名词, nr=人名, ns=地名, nt=组织, v=动词)
  - imagery lexicon from `FIELD_WORDS` (7 imagery fields)

Sequential analysis:
  - extract entity sequence in text order
  - compute adjacent-entity semantic similarity:
      * v7: char-bigram Jaccard (fast, shallow)
      * v8: bge-small-zh cosine (semantic, slower but correct)
  - rupture = high distance with NO shared imagery field
  - bridge = adjacent entities share a semantic field (imagery family)
  - logic-jump score = mean rupture with bridge compensation

Imagery fields (意象场):
  - 天象: 日月星辰风云雨雪雷雾霜露霞虹
  - 山水: 山江河水湖海溪泉林树花草
  - 动物: 鸟鹰雁燕蝶蜂鱼龙马羊鹿
  - 季节: 春夏秋冬晨夜黄昏黎明
  - 情感: 愁思忆梦情心魂泪悲欢爱恨孤独寂寞
  - 感官: 红青白黄紫碧清寒暖暗
  - 现代: 太阳月亮星星天空大地海洋火焰花朵村庄城市道路镜子影子
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

import jieba
import jieba.posseg as pseg

jieba.setLogLevel(20)
jieba.initialize()

# imagery field lexicon
FIELD_WORDS: dict[str, set[str]] = {
    "天象": {"日", "月", "星", "风", "云", "雨", "雪", "雷", "雾", "霜", "露", "霞",
             "虹", "太阳", "月亮", "星星", "夜空", "黄昏", "夕阳", "朝阳", "明月",
             "清风", "白云", "烟", "雾"},
    "山水": {"山", "水", "江", "河", "湖", "海", "溪", "泉", "林", "树", "花", "草",
             "松", "竹", "梅", "兰", "荷", "柳", "桃", "杏", "梨", "村庄", "草原",
             "荒原", "大地", "海洋", "绿水", "青山"},
    "动物": {"鸟", "鹰", "雁", "燕", "蝶", "蜂", "鱼", "龙", "马", "羊", "鹿",
             "鹤", "鸦", "鸥", "猿", "蝉", "蜻蜓", "鸳鸯", "白鹭"},
    "季节": {"春", "夏", "秋", "冬", "晨", "夜", "黄昏", "黎明", "拂晓"},
    "情感": {"愁", "思", "忆", "梦", "情", "心", "魂", "泪", "悲", "欢", "爱",
             "恨", "孤独", "寂寞", "温柔", "荒凉", "寂静", "思念"},
    "感官": {"红", "青", "白", "黄", "紫", "碧", "清", "寒", "暖", "暗", "冷",
             "香", "芬芳", "甜蜜"},
    "现代": {"太阳", "月亮", "星星", "天空", "大地", "海洋", "火焰", "花朵",
             "孩子", "老人", "城市", "道路", "镜子", "影子", "钟声", "嘴唇",
             "乳房", "头颅", "骨头", "血液", "泥土", "沙子", "石头", "草原",
             "村庄", "夜"},
}

# reverse: word -> field (first match wins)
_WORD_FIELD: dict[str, str] = {}
for field, words in FIELD_WORDS.items():
    for w in words:
        _WORD_FIELD.setdefault(w, field)

# POS tags we treat as entities
_ENTITY_POS = {"n", "nr", "ns", "nt", "nz", "v", "vn", "an", "i"}


@dataclass
class ImageryResult:
    entities: list[str]           # entity tokens in text order
    entity_pos: list[str]         # their POS tags
    fields: list[str | None]      # imagery field or None
    n_entities: int
    n_fields: int
    field_sequence: list[str]     # fields in text order (non-None)


_POSSEG_CACHE: dict[str, list[tuple[str, str]]] = {}


def _posseg_cached(text: str) -> list[tuple[str, str]]:
    """jieba.posseg.cut with cache (avoid re-segmentation)."""
    if text not in _POSSEG_CACHE:
        _POSSEG_CACHE[text] = list(pseg.cut(text))
    return _POSSEG_CACHE[text]


def extract_entities(text: str) -> ImageryResult:
    """Extract named entities / imagery in text order."""
    entities, pos_list, fields = [], [], []
    for w, flag in _posseg_cached(text):
        if not w.strip() or len(w) < 1:
            continue
        f0 = flag[0] if flag else "x"
        is_imagery = w in _WORD_FIELD
        is_entity = (f0 == "n" or flag in _ENTITY_POS)
        if is_imagery or is_entity:
            entities.append(w)
            pos_list.append(flag)
            fields.append(_WORD_FIELD.get(w))
    return ImageryResult(entities=entities, entity_pos=pos_list,
                         fields=fields, n_entities=len(entities),
                         n_fields=len(set(f for f in fields if f)),
                         field_sequence=[f for f in fields if f])


def _char_overlap(a: str, b: str) -> float:
    """Char-level overlap similarity (proxy for semantic closeness)."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    return len(inter) / max(len(sa | sb), 1)


# --- v8: bge-small-zh semantic similarity cache for entity pairs -------

_ENTITY_VEC_CACHE: dict[str, np.ndarray | None] = {}
_bge_model = None
_BGE_OK = None
_ENTITY_CACHE_PATH = None


def _load_entity_cache():
    """Load entity embeddings from disk cache (built by entity_cache.py)."""
    global _ENTITY_CACHE_PATH
    if _ENTITY_CACHE_PATH is None:
        from pathlib import Path
        _ENTITY_CACHE_PATH = (Path(__file__).resolve().parents[3]
                             / "07_reproducibility" / "entity_vec_cache.npz")
    if not _ENTITY_CACHE_PATH.exists():
        return
    try:
        data = np.load(_ENTITY_CACHE_PATH, allow_pickle=True)
        ws = data["words"]
        vs = data["vecs"]
        for w, v in zip(ws, vs):
            _ENTITY_VEC_CACHE[str(w)] = v
    except Exception as e:
        import sys
        print(f"[imagery_ner] entity cache load failed: {e}", file=sys.stderr)


def _get_bge():
    """Lazy load bge-small-zh. Returns None on failure."""
    global _bge_model, _BGE_OK
    if _BGE_OK is False:
        return None
    if _bge_model is None:
        # Try disk cache first (fast path)
        _load_entity_cache()
        if len(_ENTITY_VEC_CACHE) > 100:
            _BGE_OK = "cache_only"
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _bge_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            _BGE_OK = True
        except Exception:
            _BGE_OK = False
            return None
    return _bge_model


def _entity_vec(word: str) -> np.ndarray | None:
    """Get bge embedding for an entity word (cached)."""
    if word in _ENTITY_VEC_CACHE:
        return _ENTITY_VEC_CACHE[word]
    model = _get_bge()
    if model is None:
        return None
    vec = model.encode([word], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)[0]
    _ENTITY_VEC_CACHE[word] = vec
    return vec


def _entity_sim_bge(a: str, b: str) -> float | None:
    """Semantic similarity between two entity words (None if no model)."""
    va, vb = _entity_vec(a), _entity_vec(b)
    if va is None or vb is None:
        return None
    return float(np.dot(va, vb))


def imagery_logic_jump(text: str, use_semantic: bool = True) -> dict[str, float]:
    """Sequential imagery logic analysis.

    Returns:
      - ent_adj_sim_mean:  mean adjacent-entity similarity
      - ent_adj_sim_cv:    CV of those similarities
      - field_switch_rate: fraction of adjacent entity pairs that CHANGE
                           imagery field (high = 跳跃 between fields)
      - field_return:      fraction of pairs returning to a PREVIOUS field
                           (诗的意象回环)
      - rupture_bridge:    fraction of adjacent pairs with low sim
                           (rupture) that STILL share a field (bridge)
      - logic_jump_score:  composite: mean rupture weighted by bridge
    """
    r = extract_entities(text)
    ents = r.entities
    if len(ents) < 2:
        return {"ent_adj_sim_mean": 0.0, "ent_adj_sim_cv": 0.0,
                "field_switch_rate": 0.0, "field_return": 0.0,
                "rupture_bridge": 0.0, "logic_jump_score": 0.0}

    sims = []
    switches = 0
    returns = 0
    ruptures = 0
    bridge_ok = 0
    seen_fields = set()
    prev_field = None

    # rupture threshold: 0.3 for char-overlap, 0.5 for semantic cos
    rupture_thr = 0.5 if use_semantic else 0.3

    for i in range(len(ents) - 1):
        a, b = ents[i], ents[i + 1]
        # semantic sim if requested & available, else char fallback
        sim = None
        if use_semantic:
            sim = _entity_sim_bge(a, b)
        if sim is None:
            sim = _char_overlap(a, b)
        sims.append(sim)
        fa, fb = r.fields[i], r.fields[i + 1]
        if fa and fb and fa != fb:
            switches += 1
        if fb and fb in seen_fields and prev_field != fb:
            returns += 1
        if fa:
            seen_fields.add(fa)
        prev_field = fb
        if sim < rupture_thr:
            ruptures += 1
            if fa and fb and fa == fb:
                bridge_ok += 1

    n_pairs = len(ents) - 1
    mean_sim = sum(sims) / n_pairs
    std_sim = (sum((s - mean_sim) ** 2 for s in sims) / n_pairs) ** 0.5
    cv_sim = std_sim / mean_sim if mean_sim > 0 else 0.0

    rupture_rate = ruptures / n_pairs
    bridge_rate = (bridge_ok / ruptures) if ruptures else 0.0
    logic_jump = rupture_rate * (0.5 + 0.5 * bridge_rate)

    return {
        "ent_adj_sim_mean": float(mean_sim),
        "ent_adj_sim_cv": float(cv_sim),
        "field_switch_rate": float(switches / n_pairs),
        "field_return": float(returns / n_pairs),
        "rupture_bridge": float(bridge_rate),
        "logic_jump_score": float(logic_jump),
    }


def imagery_features(text: str, use_semantic: bool = True) -> dict[str, float]:
    """Composite imagery+NER features.

    v7 (use_semantic=False): char-overlap similarity between entities.
    v8 (use_semantic=True):  bge-small-zh cosine between entities.

    Note: use_semantic=True is slow at scale; use cache to speed up.
    """
    r = extract_entities(text)
    j = imagery_logic_jump(text, use_semantic=use_semantic)
    return {
        "ner_entity_density": float(min(r.n_entities / max(len(text), 1) * 20.0, 1.0)),
        "ner_field_diversity": float(min(r.n_fields / 7.0, 1.0)),
        "ner_field_sequence_len": float(len(r.field_sequence)),
        **{f"img_{k}": v for k, v in j.items()},
    }


if __name__ == "__main__":
    print("=== NER imagery demo (v8 with bge semantics) ===")
    poem = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
    r = extract_entities(poem)
    print("poem entities:", r.entities)
    print("poem fields:", r.fields)
    print("poem jump (v7 char-overlap):", imagery_logic_jump(poem, use_semantic=False))
    print("poem jump (v8 bge-semantic): ", imagery_logic_jump(poem, use_semantic=True))
    print()
    news = "央行宣布降准0.5个百分点，银行股大涨，市场分析师表示流动性充裕"
    r2 = extract_entities(news)
    print("news entities:", r2.entities)
    print("news jump (v7):", imagery_logic_jump(news, use_semantic=False))
    print("news jump (v8):", imagery_logic_jump(news, use_semantic=True))
    print()
    gu = "小巷\n又弯又长\n没有门\n没有窗\n我拿把旧钥匙\n敲着厚厚的墙"
    r3 = extract_entities(gu)
    print("顾城 entities:", r3.entities)
    print("顾城 jump (v7):", imagery_logic_jump(gu, use_semantic=False))
    print("顾城 jump (v8):", imagery_logic_jump(gu, use_semantic=True))
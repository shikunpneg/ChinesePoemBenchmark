"""NER-based imagery extraction + sequential logic analysis (jump v2).

Addresses user feedback on `jump` family:
  - current jump only counts connectors + char density (too shallow)
  - need: use NER to extract named entities / imagery (意象), then analyze
    them IN TEXT ORDER to measure the LOGICAL JUMPINESS.

NER approach (no external model, works offline):
  - jieba.posseg for POS tags (n=名词, nr=人名, ns=地名, nt=组织, v=动词)
  - imagery lexicon from `features.IMAGERY_WORDS` (自然/感官/情感词)
  - entities = content words that are either (a) in imagery lexicon
    or (b) tagged as n/nr/ns/nt/v by jieba posseg

Sequential analysis:
  - extract entity sequence in text order
  - compute adjacent-entity semantic distance using char-bigram overlap
    (shallow proxy; upgraded to bge embeddings in the semantic phase)
  - rupture = high distance with NO shared imagery field
  - bridge = adjacent entities share a semantic field (imagery family)
  - logic-jump score = mean rupture with bridge compensation

Imagery fields (意象场): classify each imagery word into a semantic field
to measure "jump between fields" vs "stay in one field":
  - 自然天象: 日月星辰风云雨雪雷雾霜露霞虹
  - 山水: 山江河水湖海溪泉林树花草
  - 动物: 鸟鹰雁燕蝶蜂鱼龙马羊鹿
  - 季节: 春夏秋冬晨夜黄昏黎明
  - 情感: 愁思忆梦情心魂泪悲欢爱恨孤独寂寞
  - 感官色彩: 红青白黄紫碧清寒暖暗
  - 现代意象: 太阳月亮星星天空大地海洋火焰花朵村庄城市道路镜子影子
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

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


def imagery_logic_jump(text: str) -> dict[str, float]:
    """Sequential imagery logic analysis.

    Returns:
      - ent_adj_sim_mean:  mean adjacent-entity char-overlap similarity
      - ent_adj_sim_cv:    CV of those similarities
      - field_switch_rate: fraction of adjacent entity pairs that CHANGE
                           imagery field (high = 跳跃 between fields)
      - field_return:      fraction of pairs returning to a PREVIOUS field
                           (诗的意象回环)
      - rupture_bridge:    fraction of adjacent pairs with low overlap
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
    for i in range(len(ents) - 1):
        a, b = ents[i], ents[i + 1]
        sim = _char_overlap(a, b)
        sims.append(sim)
        fa, fb = r.fields[i], r.fields[i + 1]
        # field switch
        if fa and fb and fa != fb:
            switches += 1
        # field return
        if fb and fb in seen_fields and prev_field != fb:
            returns += 1
        if fa:
            seen_fields.add(fa)
        prev_field = fb
        # rupture (low char overlap) + bridge (shared field)
        if sim < 0.3:
            ruptures += 1
            if fa and fb and fa == fb:
                bridge_ok += 1

    n_pairs = len(ents) - 1
    mean_sim = sum(sims) / n_pairs
    std_sim = (sum((s - mean_sim) ** 2 for s in sims) / n_pairs) ** 0.5
    cv_sim = std_sim / mean_sim if mean_sim > 0 else 0.0

    # logic-jump: rupture mean with bridge compensation
    rupture_rate = ruptures / n_pairs
    bridge_rate = (bridge_ok / ruptures) if ruptures else 0.0
    # poem: rupture high but bridge high => "断裂但有引力"
    logic_jump = rupture_rate * (0.5 + 0.5 * bridge_rate)

    return {
        "ent_adj_sim_mean": float(mean_sim),
        "ent_adj_sim_cv": float(cv_sim),
        "field_switch_rate": float(switches / n_pairs),
        "field_return": float(returns / n_pairs),
        "rupture_bridge": float(bridge_rate),
        "logic_jump_score": float(logic_jump),
    }


def imagery_features(text: str) -> dict[str, float]:
    """Composite imagery+Ner features."""
    r = extract_entities(text)
    j = imagery_logic_jump(text)
    return {
        "ner_entity_density": float(min(r.n_entities / max(len(text), 1) * 20.0, 1.0)),
        "ner_field_diversity": float(min(r.n_fields / 7.0, 1.0)),
        "ner_field_sequence_len": float(len(r.field_sequence)),
        **{f"img_{k}": v for k, v in j.items()},
    }


if __name__ == "__main__":
    print("=== NER imagery demo ===")
    poem = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
    r = extract_entities(poem)
    print("poem entities:", r.entities)
    print("poem fields:", r.fields)
    print("poem jump:", imagery_logic_jump(poem))
    print()
    news = "央行宣布降准0.5个百分点，银行股大涨，市场分析师表示流动性充裕"
    r2 = extract_entities(news)
    print("news entities:", r2.entities)
    print("news jump:", imagery_logic_jump(news))
    print()
    # 顾城
    gu = "小巷\n又弯又长\n没有门\n没有窗\n我拿把旧钥匙\n敲着厚厚的墙"
    r3 = extract_entities(gu)
    print("顾城 entities:", r3.entities)
    print("顾城 fields:", r3.fields)
    print("顾城 jump:", imagery_logic_jump(gu))
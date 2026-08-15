"""P0 + simplified P1 feature extractors.

Design follows `02_environment/baseline_metrics/feature_catalog.md`:

P0  (must-have, low implementation cost):
    F-form    - regularity of poetic form (line count, char regularity)
    F-struct  - structural markers of "looks-like-poetry"
    F-jump    - logical-jump proxy (connector density + char density)
    F-lang    - language features (imagery density, classical markers)
                -- added in round 2 to fix modern-poetry bias

P1  (simplified, one round delayed):
    F-music   - music features based on Mandarin 5-tone (pypinyin)

All features are normalized to [0, 1] so the downstream classifier can
treat them as comparable. Higher value => more poem-like.

These are **single-feature scorers**. The "indicator combination" is the
logistic-regression classifier trained on these features downstream.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import jieba
import numpy as np
from pypinyin import Style, lazy_pinyin


# --- normalization helpers ------------------------------------------------

_PUNCT = "，。！？；：、""''《》（）()【】[]…—\-·,.\!?;:\"'()<>[]"
_PUNCT_RE = re.compile(f"[{re.escape(_PUNCT)}]")
_WS_RE = re.compile(r"\s+")

_CONNECTORS = {
    "因为", "所以", "但是", "然而", "并且", "而且", "虽然", "即使",
    "如果", "则", "因此", "于是", "于是乎", "而后", "然后", "接着",
    "此外", "另外", "同时", "同样", "事实上", "其实", "不过", "可是",
    "尽管", "无论", "不管", "只要", "既然", "既而", "乃至", "乃至说",
    "换句话说", "也就是说", "进一步", "更进一步", "更重要的是",
    "首先", "其次", "再次", "最后", "总之", "综上", "综上所述",
    "故", "是以", "盖", "夫", "且夫", "若夫", "至若",
}


def _strip_punct(text: str) -> str:
    return _PUNCT_RE.sub(" ", text)


def _lines(text: str) -> list[str]:
    """Split text into non-empty lines, ignoring blank lines."""
    return [ln for ln in (_WS_RE.sub("", ln) for ln in text.splitlines()) if ln]


def _han_chars(text: str) -> str:
    """Keep only Han characters."""
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")


# --- P0 · F-form (form regularity) ---------------------------------------

@dataclass
class FormFeatures:
    line_count: int
    line_char_var: float       # coefficient of variation across line lengths
    classical_match: float     # 0/1 match to 5- or 7-char classical patterns
    n_lines_score: float       # normalized score for "fewer/more lines than typical text"


def _form_line_count_score(n_lines: int) -> float:
    """Poetry typically has 2-16 lines (most 4 or 8). Long prose has 1 'line'."""
    if n_lines == 1:
        return 0.05
    if 2 <= n_lines <= 4:
        return 1.0
    if 5 <= n_lines <= 16:
        return 0.85
    if 17 <= n_lines <= 40:
        return 0.4
    return 0.1


def _form_classical_match(lines: list[str]) -> float:
    """Fraction of lines whose Han-char count is uniformly 5 or 7."""
    if not lines:
        return 0.0
    counts = [len(_han_chars(ln)) for ln in lines]
    counts = [c for c in counts if c > 0]
    if not counts:
        return 0.0
    uniform = all(c == counts[0] for c in counts)
    classical_len = counts[0] in (5, 7)
    if uniform and classical_len:
        return 1.0
    if uniform:
        return 0.4
    # partial: most lines share the modal length
    from collections import Counter
    mode_count, mode_freq = Counter(counts).most_common(1)[0]
    if mode_count in (5, 7) and mode_freq / len(counts) >= 0.6:
        return 0.7
    return 0.2


def feat_form(text: str) -> dict[str, float]:
    """Return a dict of named sub-features; values are NOT yet combined."""
    ls = _lines(text)
    n = len(ls)
    han_counts = [len(_han_chars(ln)) for ln in ls]
    han_counts = [c for c in han_counts if c > 0]
    if not han_counts:
        return {"line_count": 0.0, "line_char_var": 0.5, "classical_match": 0.0,
                "n_lines_score": 0.0}
    mean = sum(han_counts) / len(han_counts)
    var = sum((c - mean) ** 2 for c in han_counts) / len(han_counts)
    std = var ** 0.5
    cv = std / mean if mean > 0 else 0.0
    # low cv (uniform lines) => more poem-like
    line_char_var = max(0.0, 1.0 - cv)   # in [0, 1]
    return {
        "line_count": float(n),
        "line_char_var": float(line_char_var),
        "classical_match": float(_form_classical_match(ls)),
        "n_lines_score": float(_form_line_count_score(n)),
    }


# --- P0 · F-struct (structural markers) ---------------------------------

def feat_structure(text: str) -> dict[str, float]:
    """Structural markers: how much does it 'look like a poem'?"""
    ls = _lines(text)
    total = text.strip()
    if not total:
        return {"n_lines": 0.0, "line_ending_punct": 0.0, "short_line_ratio": 0.0}

    n_lines = len(ls)
    # 1) line count (already used in form; we keep an extra signal)
    n_lines_score = _form_line_count_score(n_lines)

    # 2) end-of-line punctuation: poems usually have NO punctuation at line ends
    ends_with_punct = sum(1 for ln in ls if ln and ln[-1] in "。，！？；：")
    line_ending_punct = (ends_with_punct / n_lines) if n_lines else 0.0
    line_ending_punct_score = 1.0 - line_ending_punct  # high => poem-like

    # 3) ratio of "short lines" (≤ 14 Han chars): poems are dominated by short lines
    han_counts = [len(_han_chars(ln)) for ln in ls]
    short = sum(1 for c in han_counts if 0 < c <= 14)
    short_line_ratio = (short / n_lines) if n_lines else 0.0

    return {
        "n_lines": float(n_lines_score),
        "line_ending_punct": float(line_ending_punct_score),
        "short_line_ratio": float(short_line_ratio),
    }


# --- P0 · F-jump (logic-jump proxy) -------------------------------------

# Use jieba to tokenize the full text
jieba.setLogLevel(20)  # quiet

def feat_logic_jump(text: str) -> dict[str, float]:
    """Logic-jump proxy (P0 simple version).

    Intuition (from old research):
        - poetry has fewer logical connectors per character (more "jumps")
        - poetry has lower char density per line on average (more whitespace/breaks)

    These are CRUDE proxies. The full "rupture-gravity distribution"
    is a later-stage refinement.
    """
    han = _han_chars(text)
    n_chars = len(han)
    if n_chars == 0:
        return {"connector_density": 0.0, "char_per_line": 0.0,
                "line_density_var": 0.0}

    tokens = [t for t in jieba.lcut(han) if t.strip()]
    n_tok = len(tokens)
    connector_hits = sum(1 for t in tokens if t in _CONNECTORS)
    connector_density = connector_hits / n_tok if n_tok else 0.0
    # poem-like: lower connector density
    connector_score = max(0.0, 1.0 - min(connector_density * 50.0, 1.0))

    # char per line: low => poem-like
    lines = _lines(text)
    han_counts = [len(_han_chars(ln)) for ln in lines if _han_chars(ln)]
    if not han_counts:
        char_per_line_score = 0.0
        line_density_var = 0.0
    else:
        avg = sum(han_counts) / len(han_counts)
        # 5-12 chars/line is typical poetry
        if avg <= 12:
            char_per_line_score = 1.0
        elif avg <= 20:
            char_per_line_score = 0.7
        elif avg <= 40:
            char_per_line_score = 0.3
        else:
            char_per_line_score = 0.05
        var = sum((c - avg) ** 2 for c in han_counts) / len(han_counts)
        std = var ** 0.5
        cv = std / avg if avg > 0 else 0.0
        # moderate cv (some variation, not chaotic)
        # 0.2-0.6 is "poetic"; pure uniform or pure chaotic both score lower
        if 0.2 <= cv <= 0.6:
            line_density_var = 1.0
        elif cv < 0.2:
            line_density_var = 0.5
        else:
            line_density_var = max(0.0, 1.0 - (cv - 0.6))

    return {
        "connector_density": float(connector_score),
        "char_per_line": float(char_per_line_score),
        "line_density_var": float(line_density_var),
    }


# --- P1 · F-music (simplified music feature) ----------------------------

# Mandarin tone values (5-degree scale, simplified):
#   1st tone = 55  (high level)     -> "ping" (level)
#   2nd tone = 35  (rising)         -> "ze"  (rising)
#   3rd tone = 214 (dipping)        -> "ze"  (dipping)
#   4th tone = 51  (falling)        -> "ze"  (falling)
# In classical metrics, 1st & 2nd tones are "ping"; 3rd & 4th are "ze".
# Light/neutral tones (5th) don't count toward ping/ze.

_TONE_NORMAL = {1: "ping", 2: "ze", 3: "ze", 4: "ze", 5: None}


# --- P0 · F-purity (text purity, added round 3 / v3) --------------------

import re as _re

_ENGLISH_RE = _re.compile(r"[A-Za-z]")
_DIGIT_RE = _re.compile(r"[0-9]")
_WHITESPACE_RE = _re.compile(r"\s")


def feat_purity(text: str) -> dict[str, float]:
    """Text-purity features (v3 addition).

    Rationale (from Stage 2 R9): AI-generated poems that mix in English
    letters / digits / code snippets are judged 'not poetry' by humans,
    but the Stage-1 metric (which looks at Han-only structure) classifies
    them as poems. Adding purity features lets the metric detect this.

    Signals (all higher = MORE poem-like):
      - han_ratio:     Han chars / all chars (high = clean Chinese)
      - no_english:    1 - english_chars / all chars (high = no English)
      - no_digit:      1 - digit_chars / all chars
      - line_cleanliness: fraction of lines that contain no English/digit
    """
    total = len(text)
    if total == 0:
        return {"han_ratio": 0.0, "no_english": 1.0, "no_digit": 1.0,
                "line_cleanliness": 1.0}
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    eng = len(_ENGLISH_RE.findall(text))
    dig = len(_DIGIT_RE.findall(text))
    han_ratio = han / total

    no_english = 1.0 - min(eng / total, 1.0)
    no_digit = 1.0 - min(dig / total, 1.0)

    lines = _lines(text)
    if not lines:
        line_cleanliness = 1.0
    else:
        clean = 0
        for ln in lines:
            if not _ENGLISH_RE.search(ln) and not _DIGIT_RE.search(ln):
                clean += 1
        line_cleanliness = clean / len(lines)

    return {
        "han_ratio": float(han_ratio),
        "no_english": float(no_english),
        "no_digit": float(no_digit),
        "line_cleanliness": float(line_cleanliness),
    }


# --- P0 · F-style (style / register features, added round 3 / v5) -----

# Hand-curated lexicons for "register" detection. Higher density => more
# news/prose-like; lower density => more poetry-like.
# Sources: common news/finance/sports/social-media words (high frequency in
# the 1200 social+news samples we know are not poems).

# Single-word news register markers
_NEWS_WORDS = {
    # 财经
    "基金", "股票", "债券", "涨幅", "跌幅", "净值", "收益", "投资", "市场",
    "央行", "经济", "增长", "下降", "财经", "证券", "上市", "公司", "企业",
    "融资", "收购", "并购", "上市", "股价", "市值", "盈利", "亏损", "财报",
    # 体育
    "比赛", "球队", "球员", "教练", "赛季", "联赛", "决赛", "冠军", "球迷",
    "进球", "助攻", "出场", "首发", "替补", "半场", "全场", "加时", "点球",
    "篮球", "足球", "网球", "排球", "乒乓球", "羽毛球", "游泳", "跑步",
    # 新闻常见
    "报道", "消息", "据", "获悉", "据悉", "近日", "日前", "昨日", "今日",
    "记者", "编辑", "发布", "公布", "宣布", "声明", "表示", "指出", "强调",
    "举行", "召开", "出席", "参加", "表示", "认为", "指出", "强调",
    "新华社", "央视", "人民网", "中新社", "报道", "综合", "电",
    # 娱乐/时尚
    "明星", "粉丝", "微博", "热搜", "绯闻", "恋情", "结婚", "离婚", "出轨",
    "主演", "导演", "票房", "收视", "综艺", "真人秀", "选秀", "节目",
    "时装", "秀场", "红毯", "礼服", "造型", "街拍", "潮流", "穿搭", "美妆",
    # 科技/产品
    "手机", "电脑", "屏幕", "电池", "充电", "像素", "处理器", "内存", "存储",
    "配置", "售价", "上市", "发布", "搭载", "支持", "功能", "性能", "测试",
    # 论坛/社交
    "楼主", "回复", "评论", "点赞", "转发", "关注", "粉丝", "关注", "私信",
    "求", "问", "请教", "求教", "求助", "在线等", "急", "顶", "路过",
    "哈哈哈", "哈哈", "呵呵", "嗯", "哦", "啊", "咦", "唉", "额", "妈呀",
}

# Phrase-level markers (multi-char)
_NEWS_PHRASES = {
    "本报记者", "综合报道", "消息人士", "据报道", "据介绍", "据了解", "据悉",
    "日前召开", "近日举行", "昨日下午", "今日上午", "本周一", "本月初",
    "同比增长", "环比下降", "市值蒸发", "盘面上", "板块方面", "个股方面",
    "赛季", "首发阵容", "替补登场", "加时赛", "点球大战", "比分定格",
    "红毯", "造型师", "穿搭", "街拍", "种草", "拔草", "种草姬", "颜值",
    "快来", "一起", "我们一起", "有没有", "小伙伴", "宝子", "家人们",
    "链接：", "扫码", "点击", "关注", "私信", "加微信",
}

# 论坛/社交的语气词（强噪声信号）
_FORUM_FILLERS = {
    "哈哈哈", "呵呵呵", "哈哈", "呵呵", "哦哦", "啊啊", "哇哇",
    "肿么", "酱紫", "表", "orz", "23333", "666", "88", "99",
    "俺", "咱", "乃", "吼吼", "桑心", "蓝瘦", "香菇",
}


def feat_style(text: str) -> dict[str, float]:
    """Style / register features (v5 addition).

    Rationale (from Stage 2 R3 / R7): the v2 metric over-predicts "poem" on
    multi-paragraph social/news/forum text because line breaks make it look
    structurally poetic. These features add negative register signals.

    Signals (higher = more prose/news-like, LOWER = more poem-like):
      - news_word_density:    news single-word density
      - news_phrase_density:  news phrase density
      - forum_filler_density: forum/slang density
      - avg_para_len:         average chars per paragraph (prose ~long)
    """
    total_chars = len(text)
    if total_chars == 0:
        return {"news_word_density": 0.0, "news_phrase_density": 0.0,
                "forum_filler_density": 0.0, "avg_para_len": 0.0}

    # 1) single-word news density
    n_news = 0
    for w in _NEWS_WORDS:
        n_news += text.count(w)
    news_word_density = n_news / max(total_chars / 100.0, 1.0)  # per 100 chars

    # 2) news phrase density
    n_phrase = 0
    for p in _NEWS_PHRASES:
        n_phrase += text.count(p)
    news_phrase_density = n_phrase / max(total_chars / 100.0, 1.0)

    # 3) forum fillers
    n_filler = 0
    for f in _FORUM_FILLERS:
        n_filler += text.count(f)
    forum_filler_density = n_filler / max(total_chars / 100.0, 1.0)

    # 4) average paragraph length (paragraphs = non-empty lines / blocks)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    avg_para_len = sum(len(p) for p in paragraphs) / len(paragraphs)
    # Normalize: classical poem avg ~ 25-40 chars/para; prose ~100+
    # Map to [0,1] where high = prose-like
    avg_para_len_norm = min(avg_para_len / 150.0, 1.0)

    return {
        "news_word_density": float(news_word_density),
        "news_phrase_density": float(news_phrase_density),
        "forum_filler_density": float(forum_filler_density),
        "avg_para_len": float(avg_para_len_norm),
    }


# --- P0 · F-lang (language features, added round 2) ---------------------

# A small lexicon of common Chinese poetic imagery / natural / emotional words.
# Compiled manually from 古诗常见意象 + 海子/顾城/张枣语料抽样.
IMAGERY_WORDS = {
    # 自然
    "山", "水", "月", "风", "云", "雪", "雨", "花", "草", "树", "林",
    "海", "江", "河", "湖", "日", "星", "天", "地", "春", "夏", "秋",
    "冬", "雾", "霜", "露", "霞", "虹", "雷", "火", "灯", "夜", "晨",
    "黄昏", "夕阳", "朝阳", "明月", "清风", "白云", "青山", "绿水",
    # 动物
    "鸟", "鹰", "雁", "燕", "蝶", "蜂", "鱼", "龙", "马", "羊", "鹿",
    # 情感
    "愁", "思", "忆", "梦", "情", "心", "魂", "泪", "悲", "欢", "爱",
    "恨", "孤独", "寂寞", "温柔", "荒凉", "寂静",
    # 感官色
    "红", "青", "白", "黄", "紫", "碧", "清", "寒", "暖", "暗",
    # 现代意象 (顾城/海子/张枣 常用)
    "太阳", "月亮", "星星", "天空", "大地", "海洋", "火焰", "花朵",
    "孩子", "老人", "村庄", "城市", "道路", "镜子", "影子", "钟声",
    "嘴唇", "乳房", "头颅", "骨头", "血液", "泥土", "沙子", "石头",
    "村庄", "草原", "荒原", "夜空",
}

# Classical Chinese markers (文言虚词 / 之乎者也).
# Classical poems are dense in these; modern prose / news almost never uses them.
CLASSICAL_MARKERS = set("之乎者也其而且以于焉哉兮矣耳然则盖夫")

# Common news / prose particles (反向标记，密度高 -> 不像诗)
PROSE_PARTICLES = set("的着了和与及或等之性化")


def feat_language(text: str) -> dict[str, float]:
    """Language features capturing 'poetic diction' regardless of structure.

    Three signals:
      - imagery_density:  characters / Han chars in known poetic imagery vocab
                          (high => poem-like)
      - classical_marker_density: classical Chinese particles / Han chars
                          (high => classical-poem-like)
      - prose_particle_density: news/prose particles / Han chars
                          (high => NOT poem-like)
      - line_break_existence: 1 if multi-line, 0 otherwise
                          (just existence; not weighted by line length)
    """
    han = _han_chars(text)
    n = len(han)
    if n == 0:
        return {
            "imagery_density": 0.0,
            "classical_marker_density": 0.0,
            "prose_particle_density": 0.5,
            "line_break_existence": 0.0,
        }

    # imagery density: count chars in IMAGERY_WORDS divided by total Han chars
    img_hits = sum(1 for ch in han if ch in IMAGERY_WORDS)
    # also try 2-character matches for multi-char imagery words
    img_2chars = sum(
        1 for i in range(len(han) - 1) if han[i:i + 2] in IMAGERY_WORDS)
    # total weighted hits (2-char counts as 2 to give them weight)
    img_score = (img_hits + 2 * img_2chars) / n
    # saturating: 0.0 -> 0, 0.10 -> 1.0 (so dense imagery => 1.0)
    imagery_density = min(1.0, img_score / 0.10)

    cls_hits = sum(1 for ch in han if ch in CLASSICAL_MARKERS)
    classical_marker_density = min(1.0, cls_hits / n * 50.0)

    pro_hits = sum(1 for ch in han if ch in PROSE_PARTICLES)
    prose_particle_density = min(1.0, pro_hits / n * 5.0)

    line_break_existence = 1.0 if len(_lines(text)) > 1 else 0.0

    return {
        "imagery_density": float(imagery_density),
        "classical_marker_density": float(classical_marker_density),
        "prose_particle_density": float(prose_particle_density),
        "line_break_existence": float(line_break_existence),
    }


def _line_tone_pattern(line: str) -> list[str]:
    """Return ping/ze pattern for a Han-only line. Light tones are skipped."""
    pys = lazy_pinyin(line, style=Style.TONE3, errors=lambda x: ["5"])
    out = []
    for py in pys:
        if not py or py == "5":
            continue
        # py looks like "ma3" or "shi4"
        m = re.match(r"^([a-zü]+)([1-5])$", py)
        if not m:
            continue
        tone = int(m.group(2))
        tag = _TONE_NORMAL.get(tone)
        if tag is not None:
            out.append(tag)
    return out


def feat_music_simple(text: str) -> dict[str, float]:
    """Simplified music feature (P1 round-1 version).

    For each line, compute ping/ze pattern. Then:
      - pattern_regularity: how often adjacent lines share the same pattern
      - final_char_match: how often line-final chars fall on "ping" (preferred
        for classical 律诗: line-final should usually be ping for even lines
        — we don't enforce, we just measure)
    """
    lines = _lines(text)
    han_lines = [_han_chars(ln) for ln in lines]
    han_lines = [ln for ln in han_lines if ln]
    if not han_lines:
        return {"pattern_regularity": 0.0, "ping_ze_balance": 0.0,
                "final_char_ping_ratio": 0.0}

    patterns = [_line_tone_pattern(ln) for ln in han_lines]
    patterns = [p for p in patterns if p]

    # 1) pattern regularity: adjacent-line Jaccard similarity, averaged
    if len(patterns) >= 2:
        sims = []
        for i in range(len(patterns) - 1):
            a, b = patterns[i], patterns[i + 1]
            if not a or not b:
                continue
            # align by min length
            L = min(len(a), len(b))
            same = sum(1 for x, y in zip(a[:L], b[:L]) if x == y)
            sims.append(same / L)
        pattern_regularity = sum(sims) / len(sims) if sims else 0.0
    else:
        pattern_regularity = 0.0

    # 2) ping/ze balance: too imbalanced => less musical
    flat = [t for p in patterns for t in p]
    if not flat:
        ping_ze_balance = 0.0
    else:
        ping = sum(1 for t in flat if t == "ping")
        ze = sum(1 for t in flat if t == "ze")
        if ping == 0 or ze == 0:
            ping_ze_balance = 0.0
        else:
            ratio = ping / (ping + ze)
            # ideal is around 0.4-0.6; map via inverted triangle
            ping_ze_balance = max(0.0, 1.0 - abs(ratio - 0.5) * 2)

    # 3) final-char ping ratio (lightly weighted signal)
    finals = []
    for pat in patterns:
        if pat:
            finals.append(pat[-1])
    final_ping = sum(1 for t in finals if t == "ping")
    final_ping_ratio = final_ping / len(finals) if finals else 0.0

    return {
        "pattern_regularity": float(pattern_regularity),
        "ping_ze_balance": float(ping_ze_balance),
        "final_char_ping_ratio": float(final_ping_ratio),
    }


# --- Aggregator ----------------------------------------------------------

#: All feature names produced by `extract_all_features` (in fixed order).
FEATURE_NAMES: list[str] = [
    # form (classical meter)
    "meter_form_score", "meter_line_pattern_ok", "meter_dui_ok",
    "meter_nian_ok", "meter_rhyme_agreement", "meter_parallelism",
    "meter_is_metrical",
    # structure (v2: paragraphs + theme)
    "para_para_count", "para_para_len_mean", "para_para_len_cv", "para_para_var",
    "theme_theme_jump_mean", "theme_theme_jump_cv", "theme_theme_coherence",
    "theme_theme_cluster_ratio", "theme_opening_closure",
    # structure v8: theme on semantic units (merged short lines)
    "theme8_theme_jump_mean", "theme8_theme_jump_cv", "theme8_theme_coherence",
    "theme8_theme_cluster_ratio", "theme8_opening_closure",
    "theme8_unit_count_norm",
    # structure (original 3)
    "struct_n_lines", "struct_line_ending_punct", "struct_short_line_ratio",
    # logic-jump (v2: NER imagery)
    "ner_entity_density", "ner_field_diversity", "ner_field_sequence_len",
    "img_ent_adj_sim_mean", "img_ent_adj_sim_cv", "img_field_switch_rate",
    "img_field_return", "img_rupture_bridge", "img_logic_jump_score",
    # semantic (v7: bge-small-zh embeddings)
    "sem_adj_line_sim_mean", "sem_adj_line_sim_cv", "sem_first_last_sim",
    "sem_bridge_rate", "sem_wholeness", "sem_dispersion",
    # logic-jump (original 3, kept for compat)
    "jump_connector_density", "jump_char_per_line", "jump_line_density_var",
    # language (round 2)
    "lang_imagery_density", "lang_classical_marker_density",
    "lang_prose_particle_density", "lang_line_break_existence",
    # purity (round 3 / v3)
    "purity_han_ratio", "purity_no_english", "purity_no_digit",
    "purity_line_cleanliness",
    # style (round 3 / v5)
    "style_news_word_density", "style_news_phrase_density",
    "style_forum_filler_density", "style_avg_para_len",
    # music (simplified)
    "music_pattern_regularity", "music_ping_ze_balance", "music_final_char_ping_ratio",
    # phonetics (real sound, v7)
    "phon_tone_smoothness", "phon_tone_cv", "phon_resonance_var",
    "phon_rhyme_distance", "phon_rhyme_repeat", "phon_tone_balance",
]


def extract_all_features(text: str, _skip_semantic: bool = False) -> dict[str, float]:
    """Run all feature extractors and return a flat dict.

    The keys are the canonical FEATURE_NAMES. New feature families
    (meter / paragraph-theme / NER-imagery / semantic) are included.

    NOTE: semantic features lazily load the bge-small-zh model on first use;
    if unavailable they return 0.0 without breaking the pipeline.
    When `_skip_semantic=True`, semantic features are set to 0.0 (batch
    mode fills them in afterwards).
    """
    f = {}
    # form: classical meter (from meter.py)
    from .meter import meter_to_features
    f.update(meter_to_features(text))
    # structure: paragraph + theme (from structure.py)
    from .structure import structure_v2_features
    f.update(structure_v2_features(text))
    # structure: original
    f.update({f"struct_{k}": v for k, v in feat_structure(text).items()})
    # NER imagery + logic jump (from imagery_ner.py)
    from .imagery_ner import imagery_features
    f.update(imagery_features(text))
    # semantic (from semantic.py; lazy, safe)
    if _skip_semantic:
        for k in ("sem_adj_line_sim_mean", "sem_adj_line_sim_cv",
                  "sem_first_last_sim", "sem_bridge_rate",
                  "sem_wholeness", "sem_dispersion"):
            f[k] = 0.0
    else:
        try:
            from .semantic import semantic_features
            f.update(semantic_features(text))
        except Exception:
            for k in ("sem_adj_line_sim_mean", "sem_adj_line_sim_cv",
                      "sem_first_last_sim", "sem_bridge_rate",
                      "sem_wholeness", "sem_dispersion"):
                f[k] = 0.0
    # original jump / lang / purity / style / music
    f.update({f"jump_{k}": v for k, v in feat_logic_jump(text).items()})
    f.update({f"lang_{k}": v for k, v in feat_language(text).items()})
    f.update({f"purity_{k}": v for k, v in feat_purity(text).items()})
    f.update({f"style_{k}": v for k, v in feat_style(text).items()})
    f.update({f"music_{k}": v for k, v in feat_music_simple(text).items()})
    # phonetics (real sound, v7)
    from .phonetics import phonetic_features
    f.update({f"phon_{k}": v for k, v in phonetic_features(text).items()})
    return f


def extract_batch(texts: Iterable[str], use_semantic: bool = True) -> np.ndarray:
    """Compute features for an iterable of texts, return shape (N, F) array.

    Semantic features are computed in batch when `use_semantic=True` (much
    faster than per-text). Falls back to non-semantic if model unavailable.
    """
    texts = list(texts)
    rows = []
    # precompute semantic features in batch (once for all texts)
    sem_rows = None
    if use_semantic:
        try:
            from .semantic import semantic_features_batch
            sem_rows = semantic_features_batch(texts)
        except Exception:
            sem_rows = None
    sem_keys = ("sem_adj_line_sim_mean", "sem_adj_line_sim_cv",
                "sem_first_last_sim", "sem_bridge_rate",
                "sem_wholeness", "sem_dispersion")
    for i, t in enumerate(texts):
        d = extract_all_features(t, _skip_semantic=True)
        if sem_rows is not None and i < len(sem_rows):
            for k, v in zip(sem_keys, sem_rows[i]):
                d[k] = float(v)
        rows.append([float(d.get(k, 0.0)) for k in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float64)


# --- text-level gating (round 3, short-text handling) -----------------

# Texts shorter than this (in Han chars) are too short for reliable judgment.
# Below this threshold the metric should abstain or use a weak default.
MIN_HAN_CHARS_FOR_RELIABLE_JUDGMENT = 30


def text_reliability(text: str) -> dict[str, float]:
    """Return per-text reliability signals.

    `n_han_chars`     : number of Han characters
    `n_lines`         : number of non-empty lines
    `is_too_short`    : 1 if n_han_chars < MIN_HAN_CHARS, else 0
    `is_truncatable`  : 1 if the text is suspiciously short OR has very
                        few lines OR is mostly punctuation/whitespace
    """
    han = _han_chars(text)
    n = len(han)
    lines = _lines(text)
    n_lines = len(lines)
    is_too_short = 1.0 if n < MIN_HAN_CHARS_FOR_RELIABLE_JUDGMENT else 0.0
    is_truncatable = 1.0 if (n < MIN_HAN_CHARS_FOR_RELIABLE_JUDGMENT
                              or n_lines < 2) else 0.0
    return {
        "n_han_chars": float(n),
        "n_lines": float(n_lines),
        "is_too_short": is_too_short,
        "is_truncatable": is_truncatable,
    }


# Backward-compat: a few downstream scripts ask for `n_chars` directly.
def text_length_features(text: str) -> dict[str, float]:
    """Alias for `text_reliability` kept for legacy imports."""
    return text_reliability(text)
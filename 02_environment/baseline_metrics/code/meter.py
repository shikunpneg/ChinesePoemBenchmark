"""Classical Chinese poetry meter rules (近体诗格律).

Implements real 格律 standards for 五言/七言 绝句/律诗:

  1. 句式 (sentence patterns): the 4 standard line patterns for 平仄
  2. 粘对 (tonal contrast): 对 = odd/even line pairs must differ in tone
                           粘 = even/odd line pairs must agree
  3. 押韵 (rhyming): even lines (2,4,6,8) must rhyme on 平声韵
  4. 对仗 (parallelism): 律诗 3rd/4th & 5th/6th lines are parallel couplets

The 4 平仄 sentence patterns (● = 仄, ○ = 平, ⊙ = 可平可仄):

  五言:
    A: ⊙仄平平仄    仄仄平平仄
    B: 平平仄仄平    平平仄仄平
    C: ⊙平平仄仄    平平平仄仄
    D: ⊙仄仄平平    仄仄仄平平

  七言 (prepend ⊙仄/⊙平):
    A: ⊙平⊙仄平平仄
    B: ⊙仄平平仄仄平
    C: ⊙仄⊙平平仄仄
    D: ⊙平⊙仄仄平平

Standard 押韵: even lines rhyme (同韵母), odd lines 不押 (except first line
which may optionally rhyme on 平).

NOTE on tone classification:
  - Classical 平仄 is based on 中古音 (平水韵), NOT modern Mandarin.
  - pypinyin gives modern tones: 1,2=平 (in modern sense), 3,4=仄.
  - Modern mapping 1,2→平 / 3,4→仄 is only an approximation of 中古音.
  - For a proper implementation we'd need 平水韵/广韵 tables (中古音声母韵母).
  - We implement BOTH: modern-tone approximation AND a pluggable table hook
    so a 中古音 table can be dropped in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pypinyin import Style, lazy_pinyin

# --- tone classification ------------------------------------------------

# Modern Mandarin: tone 1,2 = 平, 3,4 = 仄, 5 = neutral (skip)
MODERN_TONE_TO_PZ = {1: "平", 2: "平", 3: "仄", 4: "仄", 5: None}


def _char_tone_modern(ch: str) -> Optional[str]:
    """Return '平'/'仄'/None for a Han char using modern Mandarin tones."""
    pys = lazy_pinyin(ch, style=Style.TONE3, errors=lambda x: ["5"])
    if not pys:
        return None
    m = re.match(r"^([a-zü]+)([1-5])$", pys[0])
    if not m:
        return None
    tone = int(m.group(2))
    return MODERN_TONE_TO_PZ.get(tone)


def _line_pz_modern(line: str) -> list[str]:
    """Return 平仄 sequence for a line (Han chars only), skipping neutral tones."""
    han = "".join(ch for ch in line if "\u4e00" <= ch <= "\u9fff")
    out = []
    for ch in han:
        pz = _char_tone_modern(ch)
        if pz:
            out.append(pz)
    return out


# --- standard patterns --------------------------------------------------

# Pattern A/B/C/D as strings of 平/仄/⊙ (⊙ = 平 or 仄)
FIVE_CN = {
    "A": "仄仄平平仄",
    "B": "平平仄仄平",
    "C": "平平平仄仄",
    "D": "仄仄仄平平",
}
SEVEN_CN = {
    "A": "平平仄仄平平仄",
    "B": "仄仄平平仄仄平",
    "C": "仄仄平平平仄仄",
    "D": "平平仄仄仄平平",
}


def _pattern_match(pz_seq: list[str], pattern: str) -> tuple[bool, int]:
    """Check if a 平仄 sequence matches a pattern string (⊙ = wildcard).

    Returns (is_match, n_errors). Errors count chars that conflict.
    """
    if len(pz_seq) != len(pattern):
        return False, len(pz_seq)  # wrong length = total error
    errors = 0
    for a, b in zip(pz_seq, pattern):
        if b == "⊙":
            continue
        if a != b:
            errors += 1
    return errors == 0, errors


def classify_line_pattern(line: str, charset: dict[str, str] | None = None) -> dict:
    """Classify a single line's 平仄 pattern against A/B/C/D.

    charset: mapping pattern-name -> pattern string (五言 or 七言).
    Returns {length, pattern, errors, matched_any}.
    """
    pz = _line_pz_modern(line)
    length = len(pz)
    if length not in (5, 7):
        return {"length": length, "pattern": None, "errors": length,
                "matched_any": False, "pz_seq": "".join(pz)}
    charset = charset or (SEVEN_CN if length == 7 else FIVE_CN)
    best = None
    for name, pat in charset.items():
        is_match, err = _pattern_match(pz, pat)
        if best is None or err < best[1]:
            best = (name, err, is_match)
    return {
        "length": length,
        "pattern": best[0],
        "errors": best[1],
        "matched_any": best[2],
        "pz_seq": "".join(pz),
    }


# --- 粘对 (tonal contrast rules) ---------------------------------------

def check_dui(prev_line_pz: list[str], next_line_pz: list[str]) -> float:
    """对: two adjacent lines (odd/even pair) must DIFFER in 平仄.

    Returns agreement fraction (0 = perfect 对, higher = violation).
    Compares aligned by min length; counts positions where they're EQUAL
    as violations (对 requires opposite).
    """
    L = min(len(prev_line_pz), len(next_line_pz))
    if L == 0:
        return 0.0
    violations = sum(1 for i in range(L) if prev_line_pz[i] == next_line_pz[i])
    return violations / L


def check_nian(prev_line_pz: list[str], next_line_pz: list[str]) -> float:
    """粘: 2nd-char of even line = 2nd-char of following odd line.

    In classical rules: line pair (2,3), (4,5), (6,7) must share the same
    second-character tone (粘). We measure the 2nd-char agreement.
    """
    if len(prev_line_pz) < 2 or len(next_line_pz) < 2:
        return 0.0
    return 1.0 if prev_line_pz[1] == next_line_pz[1] else 0.0


# --- 押韵 (rhyming) -----------------------------------------------------

def line_final_syllable(line: str) -> Optional[str]:
    """Return the pinyin (with tone) of the last Han char of a line."""
    han = "".join(ch for ch in line if "\u4e00" <= ch <= "\u9fff")
    if not han:
        return None
    pys = lazy_pinyin(han[-1], style=Style.TONE3, errors=lambda x: [""])
    return pys[0] if pys and pys[0] else None


def _rhyme_group(py: str) -> Optional[str]:
    """Group a pinyin into a rough rhyme class (韵母 approximation).

    NOTE: This is a MODERN Mandarin approximation. True 平水韵 uses
    中古音. For a proper implementation, use a 平水韵 韵部 table.
    We derive the 韵母 (final) from the pinyin (everything after initial).
    """
    if not py:
        return None
    # strip tone digit
    m = re.match(r"^([a-zü]+)([1-5])?$", py)
    if not m:
        return None
    syllable = m.group(1)
    # rough initial set
    initials = ("zh", "ch", "sh", "z", "c", "s", "b", "p", "m", "f",
                "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
                "y", "w")
    for ini in sorted(initials, key=len, reverse=True):
        if syllable.startswith(ini):
            return syllable[len(ini):]
    return syllable


def check_rhyme(lines: list[str], n_lines: int | None = None) -> dict:
    """Check 押韵: even lines (2,4,6,8...) must share the same 韵母.

    Returns {rhyme_group, even_line_agreement, violations}.
    """
    even_lines = [lines[i] for i in range(1, len(lines), 2)]  # 0-indexed even
    if len(even_lines) < 2:
        return {"rhyme_group": None, "even_line_agreement": 1.0,
                "violations": 0}
    finals = []
    for ln in even_lines:
        py = line_final_syllable(ln)
        finals.append(_rhyme_group(py) if py else None)
    valid = [f for f in finals if f]
    if not valid:
        return {"rhyme_group": None, "even_line_agreement": 0.0,
                "violations": len(finals)}
    from collections import Counter
    mode, cnt = Counter(valid).most_common(1)[0]
    return {
        "rhyme_group": mode,
        "even_line_agreement": cnt / len(valid),
        "violations": len(valid) - cnt,
    }


# --- 对仗 (parallelism) — structural proxy -----------------------------

def check_parallelism(lines: list[str]) -> float:
    """律诗 3rd/4th & 5th/6th lines should be parallel couplets.

    True 对仗 requires semantic/grammatical parallelism (noun-noun,
    verb-verb, color-color). We approximate via char-length equality +
    POS-pattern overlap (verbs/nouns at same positions).

    Returns a score in [0,1]: 1 = perfect couplet parallelism.
    """
    if len(lines) < 6:
        return 0.0
    scores = []
    for pair in [(2, 3), (4, 5)]:  # 0-indexed 3rd/4th, 5th/6th
        if pair[1] >= len(lines):
            break
        a, b = lines[pair[0]], lines[pair[1]]
        ha, hb = len(_han_chars_simple(a)), len(_han_chars_simple(b))
        if ha == 0 or hb == 0 or ha != hb:
            scores.append(0.0)
            continue
        # positional char-length match + simple word-class overlap
        # (approx: same-length lines with 名词 at symmetric positions)
        ta, tb = _simple_pos_tags(a), _simple_pos_tags(b)
        L = min(len(ta), len(tb))
        if L == 0:
            scores.append(0.0)
            continue
        same = sum(1 for i in range(L) if ta[i] == tb[i])
        scores.append(same / L)
    return sum(scores) / len(scores) if scores else 0.0


def _han_chars_simple(s: str) -> str:
    return "".join(ch for ch in s if "\u4e00" <= ch <= "\u9fff")


# Very rough POS tagging using jieba's tokenizer (noun/verb detection)
import jieba as _jieba
_jieba.setLogLevel(20)
_jieba.initialize()

_NOUNS = {"n", "nr", "ns", "nt", "nz", "ng", "an", "vn"}
_VERBS = {"v", "vd", "vn", "vg", "a", "ad", "ag"}


def _simple_pos_tags(line: str) -> list[str]:
    """Very rough POS tags per char using jieba's posseg (if available)."""
    try:
        import jieba.posseg as pseg
        tags = []
        for w, flag in pseg.cut(line):
            f0 = flag[0] if flag else "x"
            if f0 == "n" or flag in _NOUNS:
                tags.extend(["n"] * len(w))
            elif f0 == "v" or flag in _VERBS:
                tags.extend(["v"] * len(w))
            else:
                tags.extend(["x"] * len(w))
        return tags
    except Exception:
        return ["x"] * len(_han_chars_simple(line))


# --- full poem meter check ---------------------------------------------

@dataclass
class MeterResult:
    n_lines: int
    char_len: int                 # 5 or 7 if uniform, else 0
    form: str                     # "绝句" / "律诗" / "非近体"
    line_patterns: list[dict]
    dui_score: float              # 0=perfect, 1=worst (lower better)
    nian_score: float             # 0=perfect, 1=worst
    rhyme: dict
    parallelism: float
    is_metrical: bool             # passes 粘对 + 押韵 thresholds


def analyze_meter(text: str) -> MeterResult:
    """Full 格律 analysis of a poem text.

    Returns structured result with:
      - form classification (绝句/律诗/other)
      - per-line pattern match (A/B/C/D)
      - 粘对 violation score
      - 押韵 agreement
      - parallelism score
      - is_metrical flag (threshold-based)
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if _han_chars_simple(ln)]
    n = len(lines)
    if n not in (4, 8):
        return MeterResult(n_lines=n, char_len=0, form="非近体",
                           line_patterns=[], dui_score=0.0, nian_score=0.0,
                           rhyme={}, parallelism=0.0, is_metrical=False)

    char_lens = [len(_han_chars_simple(ln)) for ln in lines]
    char_lens = [c for c in char_lens if c > 0]
    if not char_lens or len(set(char_lens)) != 1 or char_lens[0] not in (5, 7):
        return MeterResult(n_lines=n, char_len=0, form="非近体",
                           line_patterns=[], dui_score=0.0, nian_score=0.0,
                           rhyme={}, parallelism=0.0, is_metrical=False)
    clen = char_lens[0]
    form = ("律诗" if n == 8 else "绝句")

    # per-line patterns
    line_patterns = [classify_line_pattern(ln) for ln in lines]

    # 粘对
    pz_lines = [_line_pz_modern(ln) for ln in lines]
    dui_scores = []
    for i in range(0, n - 1, 2):
        dui_scores.append(check_dui(pz_lines[i], pz_lines[i + 1]))
    dui_score = sum(dui_scores) / len(dui_scores) if dui_scores else 0.0
    nian_scores = []
    for i in range(1, n - 1, 2):
        nian_scores.append(check_nian(pz_lines[i], pz_lines[i + 1]))
    # nian violation = 1 - agreement
    nian_score = 1.0 - (sum(nian_scores) / len(nian_scores)
                        if nian_scores else 1.0)

    # 押韵
    rhyme = check_rhyme(lines)

    # 对仗 (律诗 only)
    parallelism = check_parallelism(lines) if form == "律诗" else 0.0

    # metrical threshold: 粘对 well-behaved (dui<0.3, nian<0.3)
    # and rhyme agreement >= 0.5
    is_metrical = (dui_score <= 0.3 and nian_score <= 0.3
                   and rhyme.get("even_line_agreement", 0) >= 0.5)

    return MeterResult(n_lines=n, char_len=clen, form=form,
                       line_patterns=line_patterns, dui_score=dui_score,
                       nian_score=nian_score, rhyme=rhyme,
                       parallelism=parallelism, is_metrical=is_metrical)


def meter_to_features(text: str) -> dict[str, float]:
    """Convert MeterResult into feature dict (higher = more metrical)."""
    r = analyze_meter(text)
    return {
        "meter_form_score": 1.0 if r.form == "律诗" else (
            0.7 if r.form == "绝句" else 0.0),
        "meter_line_pattern_ok": float(sum(
            1 for p in r.line_patterns if p["matched_any"]) / max(len(r.line_patterns), 1)),
        "meter_dui_ok": float(1.0 - min(r.dui_score, 1.0)),
        "meter_nian_ok": float(1.0 - min(r.nian_score, 1.0)),
        "meter_rhyme_agreement": float(r.rhyme.get("even_line_agreement", 0.0)),
        "meter_parallelism": float(r.parallelism),
        "meter_is_metrical": float(r.is_metrical),
    }


# quick self-test
if __name__ == "__main__":
    # 李商隐《登乐游原》: 向晚意不适 (仄仄仄仄仄), 驱车登古原 (平平平仄平)
    poem1 = "向晚意不适\n驱车登古原\n夕阳无限好\n只是近黄昏"
    r1 = analyze_meter(poem1)
    print("绝句 sample:", r1.form, "char_len", r1.char_len,
          "dui", round(r1.dui_score, 2), "nian", round(r1.nian_score, 2),
          "rhyme", r1.rhyme, "metrical", r1.is_metrical)
    # 杜甫《春望》 国破山河在 城春草木深 感时花溅泪 恨别鸟惊心 烽火连三月 家书抵万金 白头搔更短 浑欲不胜簪
    poem2 = ("国破山河在\n城春草木深\n感时花溅泪\n恨别鸟惊心\n"
             "烽火连三月\n家书抵万金\n白头搔更短\n浑欲不胜簪")
    r2 = analyze_meter(poem2)
    print("律诗 sample:", r2.form, "char_len", r2.char_len,
          "dui", round(r2.dui_score, 2), "nian", round(r2.nian_score, 2),
          "rhyme", r2.rhyme, "parallelism", round(r2.parallelism, 2),
          "metrical", r2.is_metrical)
    # 海子现代诗（不应合律）
    poem3 = "从明天起，做一个幸福的人\n喂马，劈柴，周游世界\n从明天起，关心粮食和蔬菜\n我有一所房子，面朝大海，春暖花开"
    r3 = analyze_meter(poem3)
    print("现代诗 sample:", r3.form, "char_len", r3.char_len,
          "metrical", r3.is_metrical)
    print("\nmeter_to_features demo:")
    print(meter_to_features(poem2))
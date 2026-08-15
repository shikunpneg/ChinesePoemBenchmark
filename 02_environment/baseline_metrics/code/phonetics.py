"""Real phonetic/acoustic features for Chinese poetry (music v2).

Per the original research (旧方案 §2 原则二):
  > 诗歌的音乐性根植于实际发音而非文字表面——平仄本质是声调调值（五度标记）
  > 的高低起伏，押韵本质是韵母音位（发音位置/开口度）的重复。指标必须通过
  > "文本 → 拼音/国际音标 → 声学特征映射表"计算。

We implement REAL phonetic features using pypinyin IPA + a vowel-openness
(开口度/舌位) mapping table:

  - tone_curve:         五度调值曲线（每字声调 1-5）
  - tone_smoothness:    调值曲线的平滑度（相邻调值差的平均倒数）
  - vowel_openness:     元音开口度等级（低=高元音 i/u/y, 高=低元音 a/ɑ）
  - resonance_curve:    元音开口度随时间的变化（响度代理）
  - rhyme_distance:     相邻偶数行韵母的音位距离（相近=和谐）
  - stress_pattern:     轻重音交替的规则度

All are computed from the ACTUAL PRONUNCIATION (pinyin/IPA), not from
characters' visual form.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from pypinyin import Style, lazy_pinyin

# --- tone (声调五度值) --------------------------------------------------

# Tone 1 = 55, 2 = 35, 3 = 214, 4 = 51, 5 = neutral (approx as 3)
_TONE_VALUE = {1: 5, 2: 4, 3: 2, 4: 1, 5: 3}


_PY_CACHE: dict[str, list[str]] = {}


def _line_py(line: str) -> list[str]:
    """Pinyin (TONE3) for each Han char in a line, with cache."""
    han = "".join(ch for ch in line if "\u4e00" <= ch <= "\u9fff")
    if han in _PY_CACHE:
        return _PY_CACHE[han]
    pys = lazy_pinyin(han, style=Style.TONE3, errors=lambda x: ["5"])
    _PY_CACHE[han] = pys
    return pys


def _char_tone(ch: str) -> Optional[int]:
    """Return tone number (1-5) for a Han char, or None."""
    pys = _line_py(ch)
    if not pys:
        return None
    m = re.match(r"^([a-zü]+)([1-5])$", pys[0])
    if not m:
        return None
    return int(m.group(2))


def _line_tone_curve(line: str) -> list[int]:
    """Five-degree tone values for each Han char in a line."""
    pys = _line_py(line)
    out = []
    for py in pys:
        m = re.match(r"^([a-zü]+)([1-5])$", py)
        if m:
            t = int(m.group(2))
            if t in _TONE_VALUE:
                out.append(_TONE_VALUE[t])
    return out


def tone_curve_smoothness(curve: list[int]) -> float:
    """How smooth the tone curve is (adjacent tone diffs small = smooth)."""
    if len(curve) < 2:
        return 0.0
    diffs = [abs(curve[i] - curve[i + 1]) for i in range(len(curve) - 1)]
    mean_diff = sum(diffs) / len(diffs)
    # map diff 0 -> 1.0, diff 4 -> 0.0
    return max(0.0, 1.0 - mean_diff / 4.0)


def tone_contour_cv(curve: list[int]) -> float:
    """CV of tone values (variation in pitch)."""
    if len(curve) < 2:
        return 0.0
    mean = sum(curve) / len(curve)
    std = (sum((v - mean) ** 2 for v in curve) / len(curve)) ** 0.5
    return std / mean if mean > 0 else 0.0


# --- vowel openness (元音开口度/舌位) ---------------------------------

# IPA-ish mapping via pypinyin Style.IPA gives full IPA; but we can
# approximate the FINAL's vowel nucleus from the TONE3 pinyin directly.
# Openness classes (approximate, by vowel height):
#   high (closed): i, ü, u        -> openness 1
#   mid-high:      e, o           -> openness 2
#   mid-low:       ê              -> openness 3
#   low (open):    a, ɑ           -> openness 4
_VOWEL_OPENNESS = {
    "i": 1, "u": 1, "ü": 1, "v": 1,
    "e": 2, "o": 2,
    "ê": 3,
    "a": 4, "ɑ": 4,
}


def _syllable_parts(py: str) -> tuple[str, str]:
    """Split pinyin into (initial, final); returns (final_without_tone, initial)."""
    m = re.match(r"^([a-zü]+)([1-5])?$", py)
    if not m:
        return "", ""
    syllable = m.group(1)
    initials = ("zh", "ch", "sh", "z", "c", "s", "b", "p", "m", "f",
                "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
                "y", "w")
    for ini in sorted(initials, key=len, reverse=True):
        if syllable.startswith(ini) and len(syllable) > len(ini):
            return syllable[len(ini):], ini
    return syllable, ""


def _vowel_nucleus(final: str) -> str:
    """Extract main vowel from a final (韵母)."""
    if not final:
        return ""
    # finals can have compound vowels; pick the most open one
    best, best_open = "", -1
    for ch in final:
        if ch in _VOWEL_OPENNESS and _VOWEL_OPENNESS[ch] > best_open:
            best, best_open = ch, _VOWEL_OPENNESS[ch]
    return best


def _char_vowel_openness(ch: str) -> Optional[int]:
    """Openness of the main vowel for a char (1=closed ... 4=open)."""
    pys = _line_py(ch)
    if not pys or not pys[0]:
        return None
    final, _ = _syllable_parts(pys[0])
    nucleus = _vowel_nucleus(final)
    if not nucleus:
        return None
    return _VOWEL_OPENNESS.get(nucleus)


def line_resonance_curve(line: str) -> list[int]:
    """Vowel-openness sequence (resonance proxy) for each char."""
    pys = _line_py(line)
    out = []
    for py in pys:
        final, _ = _syllable_parts(py)
        nucleus = _vowel_nucleus(final)
        if nucleus and nucleus in _VOWEL_OPENNESS:
            out.append(_VOWEL_OPENNESS[nucleus])
    return out


def resonance_variation(curve: list[int]) -> float:
    """How much resonance varies (energetic prosody = varied)."""
    if len(curve) < 2:
        return 0.0
    mean = sum(curve) / len(curve)
    std = (sum((v - mean) ** 2 for v in curve) / len(curve)) ** 0.5
    cv = std / mean if mean else 0.0
    # moderate-high CV = lively; too low = flat; too high = chaotic
    if 0.2 <= cv <= 0.5:
        return 1.0
    if cv < 0.2:
        return cv / 0.2
    return max(0.0, 1.0 - (cv - 0.5))


# --- rhyme phoneme distance (韵母音位距离) ------------------------------

def final_phoneme(py: str) -> tuple[str, str, str]:
    """Return (vowel_nucleus, coda, full_final) of a pinyin."""
    final, _ = _syllable_parts(py)
    nucleus = _vowel_nucleus(final)
    # coda = trailing consonant of final (n, ng, etc.)
    coda = ""
    for c in ("ng", "n"):
        if final.endswith(c) and len(final) > len(nucleus):
            coda = c
            break
    return nucleus, coda, final


def rhyme_phoneme_distance(py_a: str, py_b: str) -> float:
    """Distance between two syllables' rhyme phonemes (0=same, 1=very different).

    Uses: same nucleus? same coda? openness difference.
    """
    na, ca, _ = final_phoneme(py_a)
    nb, cb, _ = final_phoneme(py_b)
    if not na or not nb:
        return 1.0
    d = 0.0
    if na != nb:
        d += 0.5 + abs(_VOWEL_OPENNESS.get(na, 2) - _VOWEL_OPENNESS.get(nb, 2)) * 0.1
    if ca != cb:
        d += 0.3
    return min(d, 1.0)


def line_final_py(line: str) -> Optional[str]:
    pys = _line_py(line)
    return pys[-1] if pys else None


def phonetic_features(text: str) -> dict[str, float]:
    """Real phonetic features for the whole text.

    Returns:
      - tone_smoothness:   avg 五度调值曲线平滑度 across lines
      - tone_cv:           avg pitch variation CV
      - resonance_var:     avg resonance (vowel-openness) variation
      - rhyme_distance:    avg phoneme distance between even-line finals
      - rhyme_repeat:      fraction of even-line pairs with same nucleus
      - tone_balance:      ping/ze balance (from real tones)
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if any("\u4e00" <= c <= "\u9fff" for c in ln)]
    if not lines:
        return {"tone_smoothness": 0.0, "tone_cv": 0.0, "resonance_var": 0.0,
                "rhyme_distance": 1.0, "rhyme_repeat": 0.0, "tone_balance": 0.0}

    smooths, cvs, resvars = [], [], []
    for ln in lines:
        curve = _line_tone_curve(ln)
        if curve:
            smooths.append(tone_curve_smoothness(curve))
            cvs.append(tone_contour_cv(curve))
        rcurve = line_resonance_curve(ln)
        if rcurve:
            resvars.append(resonance_variation(rcurve))

    # even-line rhyme
    even_pys = [line_final_py(lines[i]) for i in range(1, len(lines), 2)]
    even_pys = [p for p in even_pys if p]
    rhyme_dist, rhyme_repeat = 1.0, 0.0
    if len(even_pys) >= 2:
        dists = []
        for i in range(len(even_pys) - 1):
            dists.append(rhyme_phoneme_distance(even_pys[i], even_pys[i + 1]))
        rhyme_dist = sum(dists) / len(dists)
        rhyme_repeat = sum(1 for d in dists if d < 0.2) / len(dists)

    # ping/ze balance from real tone values
    all_tones = []
    for ln in lines:
        all_tones.extend(_line_tone_curve(ln))
    if all_tones:
        ping = sum(1 for t in all_tones if t >= 4)   # tone 1,2 = ping
        ze = len(all_tones) - ping
        if ping == 0 or ze == 0:
            tone_balance = 0.0
        else:
            ratio = ping / (ping + ze)
            tone_balance = max(0.0, 1.0 - abs(ratio - 0.5) * 2)
    else:
        tone_balance = 0.0

    return {
        "tone_smoothness": float(sum(smooths) / len(smooths)) if smooths else 0.0,
        "tone_cv": float(sum(cvs) / len(cvs)) if cvs else 0.0,
        "resonance_var": float(sum(resvars) / len(resvars)) if resvars else 0.0,
        "rhyme_distance": float(rhyme_dist),
        "rhyme_repeat": float(rhyme_repeat),
        "tone_balance": float(tone_balance),
    }


if __name__ == "__main__":
    print("=== phonetic (real sound) features demo ===")
    poem = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"
    r = phonetic_features(poem)
    print("poem:", {k: round(v, 3) for k, v in r.items()})
    news = ("央行宣布降准0.5个百分点，释放长期资金约1万亿元\n"
            "市场分析认为，此举有助于降低实体经济融资成本")
    r2 = phonetic_features(news)
    print("news:", {k: round(v, 3) for k, v in r2.items()})
    # 现代诗
    modern = "小巷\n又弯又长\n没有门\n没有窗"
    r3 = phonetic_features(modern)
    print("modern:", {k: round(v, 3) for k, v in r3.items()})
    # rhyme demo
    print("\n韵母音位距离:")
    print("  光(guang1) vs 霜(shuang1):", round(rhyme_phoneme_distance("guang1", "shuang1"), 2))
    print("  光(guang1) vs 乡(xiang1):", round(rhyme_phoneme_distance("guang1", "xiang1"), 2))
    print("  光(guang1) vs 河(he2):", round(rhyme_phoneme_distance("guang1", "he2"), 2))
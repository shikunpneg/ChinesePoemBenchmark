"""Data loader v2 (round-2 dataset composition).

Composition:
  + poems:     ALL 1500 (1097 classic + 403 modern) from poems_neutral.jsonl
  + nonpoems:  300 'social' from nonpoems_neutral.jsonl
              + 5 from eval-annotation/data/02_non_poetry.jsonl (hard)
              + 50 Racter nonpoems from eval-annotation/data/samples.js (hardest)

The 'news' category from nonpoems_neutral.jsonl is excluded — it was too easy
(round 1 showed 99%+ accuracy on it). This v2 set focuses on the harder slice.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Absolute import (works when this file is run as a script or imported via package)
try:
    from .js_parser import parse_samples_js
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from js_parser import parse_samples_js  # type: ignore

DATA_ROOT = Path(r"E:\生成诗歌\poetry-judge-train\data\samples")
POEMS_FILE = DATA_ROOT / "poems_neutral.jsonl"
NONPOEMS_FILE = DATA_ROOT / "nonpoems_neutral.jsonl"
EVAL_NONPOETRY = Path(r"E:\生成诗歌\eval-annotation\data\02_non_poetry.jsonl")
EVAL_SAMPLES_JS = Path(r"E:\生成诗歌\eval-annotation\data\samples.js")


@dataclass
class Sample:
    sample_id: str
    text: str
    label: int          # 1 = poem, 0 = non-poem
    source_type: str
    strat: str


# ----- loaders ----------------------------------------------------------

def _read_poems(path: Path = POEMS_FILE) -> Iterator[Sample]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            yield Sample(
                sample_id=f"poem#{i:05d}",
                text=text,
                label=1,
                source_type=str(obj.get("source_type", "?")),
                strat=str(obj.get("strat", "?")),
            )


def _read_nonpoems_filtered(
    path: Path = NONPOEMS_FILE,
    exclude_strat_prefixes: tuple[str, ...] = ("nonpoem:news:",),
) -> Iterator[Sample]:
    """Read nonpoems but skip categories we want to exclude (e.g., news)."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            strat = str(obj.get("strat", ""))
            if any(strat.startswith(p) for p in exclude_strat_prefixes):
                continue
            yield Sample(
                sample_id=f"nonpoem_filtered#{i:05d}",
                text=text,
                label=0,
                source_type=str(obj.get("source_type", "?")),
                strat=strat,
            )


def _read_eval_nonpoetry(path: Path = EVAL_NONPOETRY) -> Iterator[Sample]:
    """Parse the concatenated-JSON file (5 hard nonpoem records)."""
    raw = path.read_text(encoding="utf-8")
    # use brace counting
    i = 0
    n = 0
    while i < len(raw):
        j = raw.find("{", i)
        if j == -1:
            break
        depth = 0
        in_str = False
        escape = False
        k = j
        while k < len(raw):
            ch = raw[k]
            if escape:
                escape = False
                k += 1
                continue
            if ch == "\\":
                escape = True
                k += 1
                continue
            if ch == '"':
                in_str = not in_str
                k += 1
                continue
            if in_str:
                k += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth != 0:
            break
        try:
            obj = json.loads(raw[j:k + 1])
        except Exception:
            i = k + 1
            continue
        if isinstance(obj, dict) and obj.get("label") == 0:
            text = obj.get("text", "").strip()
            if text:
                yield Sample(
                    sample_id=f"eval_nonpoem#{n:03d}",
                    text=text,
                    label=0,
                    source_type=str(obj.get("source_type", "?")),
                    strat=f"eval:hard:{obj.get('genre', '?')}",
                )
                n += 1
        i = k + 1


def _read_racter_nonpoems(path: Path = EVAL_SAMPLES_JS) -> Iterator[Sample]:
    """Parse samples.js and yield the 50 Racter nonpoems as hard negatives."""
    items = parse_samples_js(path)
    n = 0
    for it in items:
        if it.get("genre") != "nonpoem":
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue
        yield Sample(
            sample_id=f"racter#{n:03d}",
            text=text,
            label=0,
            source_type=str(it.get("source_type", "ai")),
            strat=f"racter:{it.get('title', '?')[:20]}",
        )
        n += 1


def load_v2(
    max_poems: int | None = None,
    max_social: int | None = None,
    include_eval_hard: bool = True,
    include_racter: bool = True,
) -> list[Sample]:
    """Build the round-2 dataset."""
    samples: list[Sample] = []
    poems = list(_read_poems())
    if max_poems is not None:
        poems = poems[:max_poems]
    samples.extend(poems)

    nonpoems_social = list(_read_nonpoems_filtered())
    if max_social is not None:
        nonpoems_social = nonpoems_social[:max_social]
    samples.extend(nonpoems_social)

    if include_eval_hard:
        samples.extend(list(_read_eval_nonpoetry()))

    if include_racter:
        samples.extend(list(_read_racter_nonpoems()))

    return samples


def train_val_split(
    samples: list[Sample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Stratified split by label, deterministic."""
    poems = [s for s in samples if s.label == 1]
    nonpoems = [s for s in samples if s.label == 0]
    rng = random.Random(seed)
    rng.shuffle(poems)
    rng.shuffle(nonpoems)
    n_val_p = max(1, int(len(poems) * val_ratio))
    n_val_n = max(1, int(len(nonpoems) * val_ratio))
    val = poems[:n_val_p] + nonpoems[:n_val_n]
    train = poems[n_val_p:] + nonpoems[n_val_n:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# --- baselines support (re-imported for convenience) --------------------

def class_centroid_text(samples: list[Sample], max_chars: int = 8000) -> str:
    """Concatenate Han chars of samples (used as a coarse corpus reference).

    Capped to `max_chars` to avoid O(n^2) blow-up in downstream LCS.
    """
    out: list[str] = []
    total = 0
    for s in samples:
        han = "".join(ch for ch in s.text if "\u4e00" <= ch <= "\u9fff")
        if not han:
            continue
        if total + len(han) > max_chars:
            han = han[: max_chars - total]
        out.append(han)
        total += len(han)
        if total >= max_chars:
            break
    return "".join(out)


def class_char_counts(samples: list[Sample], n: int = 1) -> tuple[Counter, int]:
    """Aggregate char-n-gram counts over a class subset."""
    df: Counter = Counter()
    total = 0
    for s in samples:
        han = "".join(ch for ch in s.text if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams)
        total += len(grams)
    return df, total


# --- diagnostics --------------------------------------------------------

def dataset_summary(samples: list[Sample]) -> dict:
    """Return a small summary useful for reports."""
    by_strat: Counter = Counter()
    by_src: Counter = Counter()
    label_counts: Counter = Counter()
    for s in samples:
        by_strat[s.strat] += 1
        by_src[s.source_type] += 1
        label_counts[s.label] += 1
    return {
        "n_total": len(samples),
        "n_poem": label_counts[1],
        "n_nonpoem": label_counts[0],
        "strat_top": by_strat.most_common(10),
        "source_top": by_src.most_common(10),
    }
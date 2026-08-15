"""Load the poetry-judge-train labeled dataset (read-only, by path).

Per `02_environment/data_registry/README.md`, this is a **read-only**
reference. The loader never writes into the source directory.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


DATA_ROOT = Path(r"E:\生成诗歌\poetry-judge-train\data\samples")
POEMS_FILE = DATA_ROOT / "poems_neutral.jsonl"
NONPOEMS_FILE = DATA_ROOT / "nonpoems_neutral.jsonl"


@dataclass
class Sample:
    sample_id: str
    text: str
    label: int          # 1 = poem, 0 = non-poem
    source_type: str    # e.g. "classic", "news"
    strat: str          # stratification key


def _read_jsonl(path: Path) -> Iterator[Sample]:
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
                sample_id=f"{path.stem}#{i:05d}",
                text=text,
                label=int(obj.get("label", 0)),
                source_type=str(obj.get("source_type", "?")),
                strat=str(obj.get("strat", "?")),
            )


def load_all(max_per_class: int | None = None) -> list[Sample]:
    """Load both classes. Optionally cap each class size for fast iteration."""
    poems = list(_read_jsonl(POEMS_FILE))
    nonpoems = list(_read_jsonl(NONPOEMS_FILE))
    if max_per_class is not None:
        poems = poems[:max_per_class]
        nonpoems = nonpoems[:max_per_class]
    return poems + nonpoems


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
    from collections import Counter
    df: Counter = Counter()
    total = 0
    for s in samples:
        han = "".join(ch for ch in s.text if "\u4e00" <= ch <= "\u9fff")
        grams = [han[i:i + n] for i in range(len(han) - n + 1)]
        df.update(grams)
        total += len(grams)
    return df, total


# Late import to keep stdlib-only where possible
from collections import Counter  # noqa: E402
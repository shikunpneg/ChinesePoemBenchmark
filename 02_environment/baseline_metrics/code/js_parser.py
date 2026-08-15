"""Parse `eval-annotation/data/samples.js` (JS module exporting a CORPUS array).

The file contains 100 fixed items: 50 real poems (顾城/海子/张枣/唐诗) +
50 Racter-generated nonpoems. We use them as a labeled hard-negative set
for stage 1 round 2.

Each item is on a single line:
    { title: "...", author: "...", text: "...\\n...\\n...", genre: "poem"|"nonpoem", source_type: "..." },

We extract each object with a simple brace-counter parser, then convert
the JS object literal to JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

JS_PATH = Path(r"E:\生成诗歌\eval-annotation\data\samples.js")


_JS_TO_JSON_KEY = re.compile(r"([{,]\s*)([A-Za-z_]\w*)(\s*:)")


def _js_to_json(js_obj_text: str) -> str:
    """Convert JS object literal (with unquoted keys) to JSON-compatible."""
    return _JS_TO_JSON_KEY.sub(r'\1"\2"\3', js_obj_text)


def parse_samples_js(path: Path = JS_PATH) -> list[dict]:
    """Return list of dicts with title, author, text, genre, source_type."""
    raw = path.read_text(encoding="utf-8")
    # Find every top-level {...} in the file
    # The CORPUS array entries are objects separated by commas and whitespace
    # We use brace counting.
    objs = []
    i = 0
    while i < len(raw):
        # find next '{'
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
        obj_text = raw[j:k + 1]
        try:
            json_text = _js_to_json(obj_text)
            obj = json.loads(json_text)
        except Exception as e:
            print(f"[parser] skip malformed object at {j}: {e}")
            i = k + 1
            continue
        if isinstance(obj, dict) and "genre" in obj:
            objs.append(obj)
        i = k + 1
    return objs


if __name__ == "__main__":
    items = parse_samples_js()
    by_genre = {}
    for it in items:
        by_genre.setdefault(it.get("genre", "?"), []).append(it)
    print(f"Parsed {len(items)} items")
    for g, lst in by_genre.items():
        print(f"  genre={g}: {len(lst)}")
        print(f"    sample: {lst[0]['title']!r} by {lst[0].get('author', '?')!r}")
    # split poems vs nonpoems
    poems = [it for it in items if it.get("genre") == "poem"]
    nonpoems = [it for it in items if it.get("genre") == "nonpoem"]
    print(f"\npoems: {len(poems)}")
    print(f"nonpoems: {len(nonpoems)}")
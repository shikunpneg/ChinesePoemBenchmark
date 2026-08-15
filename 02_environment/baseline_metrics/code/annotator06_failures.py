"""Show the 31 cases where annotator_06 said 'not poem' but metric said 'poem'."""
import csv
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze  # noqa: E402

EXPORT6 = Path(r"E:\生成诗歌\annotations_export6.csv")
HARD_FILES = [
    ("hard_gen_LiBai", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl")),
    ("hard_gen_GuCheng", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl")),
    ("hard_gen_Haizi", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl")),
    ("hard_gen_Haizi-CN", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl")),
    ("to_annotate_near", Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl")),
]

# load annotator_06
ann = []
with EXPORT6.open("r", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        r["sample_id"] = int(r["sample_id"])
        r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
        ann.append(r)

# load datasets and index
title_author_to_item = {}
for name, path in HARD_FILES:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            key = (obj.get("title", ""), obj.get("model", ""))
            title_author_to_item.setdefault(key, []).append({**obj, "dataset": name})

# match
matched = []
for a in ann:
    key = (a["title"], a["author"])
    if key in title_author_to_item:
        matched.append({**a, **title_author_to_item[key][0]})
print(f"matched {len(matched)}")

fm = build_and_freeze(seed=42, val_ratio=0.2)

# find failure cases
print("\n=== 31 cases: human=NOT-poem, metric=poem ===")
failures = []
for m in matched:
    if m["is_poetry"]:
        continue
    pred = fm.apply(m["generated"])
    if pred.pred == 1:
        failures.append({**m, "prob": pred.prob_poem})

failures.sort(key=lambda x: -x["prob"])
print(f"count: {len(failures)}\n")
for i, m in enumerate(failures, 1):
    print(f"--- [{i}/{len(failures)}] {m['title']!r} by {m['author']} "
          f"(sim_jaccard={m.get('sim_jaccard', 0):.3f}, prob={m['prob']:.3f}) ---")
    print(m["generated"][:250].replace("\n", " / "))
    print()
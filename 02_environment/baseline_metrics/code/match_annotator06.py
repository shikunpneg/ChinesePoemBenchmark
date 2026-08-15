"""Try to match annotator_06's annotations with hard_gen_LiBai.jsonl by title/author."""
import csv
import json
from pathlib import Path

EXPORT6 = Path(r"E:\生成诗歌\annotations_export6.csv")
HARD_LIBAI = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl")
HARD_GUCHENG = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl")
HARD_HAIZI = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl")
HARD_HAIZI_CN = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl")
NEAR = Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl")

# Load annotator_06 annotations
ann_06 = []
with EXPORT6.open("r", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        r["sample_id"] = int(r["sample_id"])
        r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
        ann_06.append(r)
print(f"annotator_06: {len(ann_06)} rows")

# Build a (title, author) -> text index from each file
def load_hard(path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            items.append({
                "title": obj.get("title", ""),
                "author": obj.get("model", ""),  # 'model' field is the model name (LiBai etc.)
                "genre": obj.get("genre", ""),
                "generated": obj.get("generated", ""),
                "real_text": obj.get("real_text", ""),
                "sim_jaccard": obj.get("sim_jaccard", 0),
                "sim_cosine": obj.get("sim_cosine", 0),
                "line_idx": i,
            })
    return items

datasets = {
    "hard_gen_LiBai": load_hard(HARD_LIBAI),
    "hard_gen_GuCheng": load_hard(HARD_GUCHENG),
    "hard_gen_Haizi": load_hard(HARD_HAIZI),
    "hard_gen_Haizi-CN": load_hard(HARD_HAIZI_CN),
    "to_annotate_near": load_hard(NEAR),
}
for name, items in datasets.items():
    print(f"  {name}: {len(items)} items")

# Try to match annotator_06 by (title, author) or just title
matches = []
for a in ann_06:
    title = a["title"]
    author = a["author"]
    # first try exact (title, author)
    found = None
    for ds_name, items in datasets.items():
        for it in items:
            if it["title"] == title and it["author"] == author:
                found = (ds_name, it)
                break
        if found: break
    if found:
        matches.append({"ann": a, "ds": found[0], "item": found[1]})

print(f"\nmatched by (title, author): {len(matches)} of {len(ann_06)}")

# show sample matches
for m in matches[:5]:
    print(f"  annotator_06 sample#{m['ann']['sample_id']} ({m['ann']['title']!r}/{m['ann']['author']!r})")
    print(f"    -> matched in {m['ds']} line {m['item']['line_idx']}")
    print(f"    annot_vote={m['ann']['is_poetry']}  genre={m['item']['genre']}  sim={m['item']['sim_jaccard']:.3f}")
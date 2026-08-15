"""Print a batch of disagreement samples for manual judgment.

Usage: pass batch name as sys.argv[1].
"""
import json
import sys
from pathlib import Path

DIS_PATH = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_004\disagreements_to_review.json")

batch = sys.argv[1] if len(sys.argv) > 1 else "all"
items = json.loads(DIS_PATH.read_text(encoding="utf-8"))


def is_in_batch(d, batch):
    g = d["genre"]; s = d["source_type"]
    h = d["majority_label"]; i = d["indicator_pred"]
    if batch == "hum1_ind0":
        return h == 1 and i == 0
    if batch == "hum0_ind1_poem_classic":
        return h == 0 and i == 1 and g == "poem" and s == "classic"
    if batch == "hum0_ind1_poem_modern":
        return h == 0 and i == 1 and g == "poem" and s == "modern"
    if batch == "hum0_ind1_nonpoem_news":
        return h == 0 and i == 1 and g == "nonpoem" and s == "news"
    if batch == "hum0_ind1_nonpoem_social":
        return h == 0 and i == 1 and g == "nonpoem" and s == "social"
    if batch == "sample_poem_modern_15":
        return h == 0 and i == 1 and g == "poem" and s == "modern"
    if batch == "sample_news_15":
        return h == 0 and i == 1 and g == "nonpoem" and s == "news"
    if batch == "all":
        return True
    return False


selected = [d for d in items if is_in_batch(d, batch)]
if batch.startswith("sample_"):
    # sample randomly with deterministic seed
    import random
    random.seed(42)
    selected = random.sample(selected, min(15, len(selected)))
    selected = sorted(selected, key=lambda x: x["sample_id"])

print(f"=== batch '{batch}': {len(selected)} items ===\n")
for idx, d in enumerate(selected, 1):
    print(f"--- [{idx}/{len(selected)}] sample#{d['sample_id']} ---")
    print(f"title    : {d['title']!r}")
    print(f"author   : {d['author']!r}")
    print(f"genre    : {d['genre']}   source_type: {d['source_type']}")
    print(f"hum_major: {d['majority_label']}  indicator_pred: {d['indicator_pred']}  indicator_prob: {d['indicator_prob']:.3f}")
    print(f"n_raters : {d['n_raters']}  dissenters: {d['dissenters']}")
    print(f"TEXT:")
    print(d['text_full'])
    print()
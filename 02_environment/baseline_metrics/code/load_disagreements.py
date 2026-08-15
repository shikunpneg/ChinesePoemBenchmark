"""Load and stratify the 215 disagreements for manual review."""
import json
from pathlib import Path

LOG_PATH = Path(r"E:\ai4s\poetry-poetricity\04_memory\experiment_logs\stage2_round_003.json")
OUT_DIR = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_004")
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = json.loads(LOG_PATH.read_text(encoding="utf-8"))
disagreements = log["disagreements"]

# Stratify by genre x source_type
from collections import Counter, defaultdict
by_cat = defaultdict(list)
for d in disagreements:
    key = (d["genre"], d["source_type"])
    by_cat[key].append(d)

print(f"=== disagreement distribution by genre x source_type ===")
print(f"total: {len(disagreements)}")
for key, lst in sorted(by_cat.items()):
    print(f"  {key}: {len(lst)}")

# distribution by direction (hum vs indicator)
dir_counter = Counter()
for d in disagreements:
    if d["majority_label"] == 0 and d["indicator_pred"] == 1:
        dir_counter["hum=0, ind=1"] += 1
    elif d["majority_label"] == 1 and d["indicator_pred"] == 0:
        dir_counter["hum=1, ind=0"] += 1
    else:
        dir_counter["other"] += 1
print()
print(f"=== direction ===")
for k, v in dir_counter.most_common():
    print(f"  {k}: {v}")

# n_raters distribution
print()
print(f"=== n_raters ===")
n_raters_counter = Counter(d["n_raters"] for d in disagreements)
for k, v in sorted(n_raters_counter.items()):
    print(f"  n_raters={k}: {v}")

# Save stratified sample
samples_to_review = []
# Take ALL disagreements, save in a review-friendly format
review_data = []
for d in disagreements:
    review_data.append({
        "sample_id": d["sample_id"],
        "title": d["title"],
        "author": d["author"],
        "genre": d["genre"],
        "source_type": d["source_type"],
        "majority_label": d["majority_label"],
        "indicator_pred": d["indicator_pred"],
        "indicator_prob": d["indicator_prob"],
        "n_raters": d["n_raters"],
        "dissenters": d["dissenters"],
    })

OUT_PATH = OUT_DIR / "disagreements_to_review.json"
OUT_PATH.write_text(json.dumps(review_data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
print(f"\nsaved {len(review_data)} disagreements to {OUT_PATH}")
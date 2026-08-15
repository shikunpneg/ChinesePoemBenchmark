"""Load all 4 annotation CSVs, dedupe, analyze."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATHS = [
    Path(r"E:\生成诗歌\annotations_export.csv"),
    Path(r"E:\生成诗歌\annotations_export2.csv"),
    Path(r"E:\生成诗歌\annotations_export6.csv"),
    Path(r"E:\生成诗歌\eval-annotation\backups\annotations_hk.csv"),
]

all_rows = []
for path in CSV_PATHS:
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["sample_id"] = int(r["sample_id"])
            except (ValueError, TypeError):
                continue
            r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
            r["source"] = path.stem
            all_rows.append(r)

print(f"total raw rows: {len(all_rows)}")

# Dedupe by (username, sample_id) -- keep latest
by_user_sample = {}
for r in all_rows:
    key = (r["username"], r["sample_id"])
    existing = by_user_sample.get(key)
    if existing is None or r["updated_at"] > existing["updated_at"]:
        by_user_sample[key] = r

print(f"unique (user, sample_id) annotations: {len(by_user_sample)}")

# Per-source counts
by_source = Counter(r["source"] for r in by_user_sample.values())
print(f"by source: {dict(by_source)}")

# Per-user counts
by_user = Counter(r["username"] for r in by_user_sample.values())
print(f"by user: {dict(by_user)}")

# Save merged CSV
import json
out_path = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_006\all_annotations.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
merged = []
for r in by_user_sample.values():
    merged.append({
        "username": r["username"],
        "user_role": r["user_role"],
        "sample_id": r["sample_id"],
        "is_poetry": r["is_poetry"],
        "quality_grade": (r.get("quality_grade") or "").strip() or None,
        "truth_genre": r["truth_genre"],
        "source_type": r["source_type"],
        "title": r["title"],
        "author": r["author"],
        "updated_at": r["updated_at"],
        "source_csv": r["source"],
    })
out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                    encoding="utf-8")
print(f"\nsaved {len(merged)} merged annotations to {out_path}")

# Quick per-user stats
print("\n=== per-user overall (all samples) ===")
by_user_rows = defaultdict(list)
for r in by_user_sample.values():
    by_user_rows[r["username"]].append(r)
for u in sorted(by_user_rows):
    ann = by_user_rows[u]
    n = len(ann)
    n_yes = sum(1 for r in ann if r["is_poetry"])
    n_db_poem = sum(1 for r in ann if r["truth_genre"] == "poem")
    n_db_nonpoem = sum(1 for r in ann if r["truth_genre"] == "nonpoem")
    n_yes_on_db_poem = sum(1 for r in ann if r["truth_genre"] == "poem" and r["is_poetry"])
    n_yes_on_db_nonpoem = sum(1 for r in ann if r["truth_genre"] == "nonpoem" and r["is_poetry"])
    n_no_on_db_poem = sum(1 for r in ann if r["truth_genre"] == "poem" and not r["is_poetry"])
    print(f"  {u:20s}  n={n:>4d}  is_poem=true: {n_yes:>4d} ({n_yes/n*100:.0f}%)")
    print(f"    vs DB=poem ({n_db_poem}):   said yes={n_yes_on_db_poem}  said no={n_no_on_db_poem}  (yes-rate={n_yes_on_db_poem/max(n_db_poem,1)*100:.0f}%)")
    print(f"    vs DB=nonpoem ({n_db_nonpoem}): said yes={n_yes_on_db_nonpoem}  said no={n_db_nonpoem-n_yes_on_db_nonpoem}  (yes-rate={n_yes_on_db_nonpoem/max(n_db_nonpoem,1)*100:.0f}%)")
"""Analyze annotations_task6_r1.csv and annotations_task7_r1.csv."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

F6 = Path(r"E:\生成诗歌\annotations_task6_r1.csv")
F7 = Path(r"E:\生成诗歌\annotations_task7_r1.csv")


def load(p):
    rows = []
    with p.open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["sample_id"] = int(r["sample_id"])
            r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
            qg = (r.get("quality_grade") or "").strip()
            r["quality_grade"] = int(qg) if qg.isdigit() else None
            rows.append(r)
    return rows


def summarize(name, rows):
    print(f"\n{'='*70}")
    print(f"=== {name} ===")
    print(f"rows: {len(rows)}")
    users = Counter(r["username"] for r in rows)
    print(f"users: {dict(users)}")
    n_yes = sum(1 for r in rows if r["is_poetry"])
    print(f"is_poetry: yes={n_yes} no={len(rows)-n_yes}")
    # per-user yes rate
    by_user = defaultdict(list)
    for r in rows:
        by_user[r["username"]].append(r)
    for u, rs in sorted(by_user.items()):
        y = sum(1 for r in rs if r["is_poetry"])
        qgs = [r["quality_grade"] for r in rs if r["quality_grade"]]
        qmean = sum(qgs)/len(qgs) if qgs else None
        print(f"  {u}: n={len(rs)} yes={y} yes_rate={y/len(rs)*100:.0f}% "
              f"quality_mean={qmean:.2f}" if qmean else f"  {u}: n={len(rs)} yes={y}")
    # origin / person / source_type
    print(f"origin: {dict(Counter(r['origin'] for r in rows))}")
    print(f"source_type: {dict(Counter(r['source_type'] for r in rows))}")
    print(f"person: {dict(Counter(r['person'] for r in rows))}")
    # quality distribution
    qdist = Counter(r["quality_grade"] for r in rows)
    print(f"quality_grade dist: {dict(sorted(qdist.items(), key=lambda x: (x[0] is None, x[0])))}")
    return rows


r6 = summarize("annotations_task6_r1.csv", load(F6))
r7 = summarize("annotations_task7_r1.csv", load(F7))

# Cross-task: do the same sample_ids appear in both?
ids6 = {r["sample_id"] for r in r6}
ids7 = {r["sample_id"] for r in r7}
print(f"\n=== cross-task ===")
print(f"task6 ids: {len(ids6)}, task7 ids: {len(ids7)}, overlap: {len(ids6 & ids7)}")
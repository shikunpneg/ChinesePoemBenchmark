"""Per-annotator breakdown of disagreements.

The CSV has usernames for each annotation. Let's see WHICH annotators
voted is_poem=true on news text and is_poem=false on classical poetry.
"""

import json
import collections
from pathlib import Path

DIS_PATH = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_004\disagreements_to_review.json")
ROUND3_LOG = Path(r"E:\ai4s\poetry-poetricity\04_memory\experiment_logs\stage2_round_003.json")

# Load disagreements + per-rater info (from round 3)
items = json.loads(DIS_PATH.read_text(encoding="utf-8"))
log = json.loads(ROUND3_LOG.read_text(encoding="utf-8"))

# Per-sample rater list from log
by_sample_raters = {}
for d in log["disagreements"]:
    sid = d["sample_id"]
    # Need to fetch rater details - reconstruct from CSV via DB
    pass

# Better approach: re-query CSV joined to text
import csv
import psycopg2

CSV_PATH = Path(r"E:\生成诗歌\eval-annotation\backups\annotations_hk.csv")
rows = []
with CSV_PATH.open("r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["sample_id"] = int(r["sample_id"])
        r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
        rows.append(r)

# Get texts
conn = psycopg2.connect(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
    dbname="neondb", sslmode="require", connect_timeout=15)
cur = conn.cursor()
sample_ids = sorted({r["sample_id"] for r in rows})
texts = {}
for i in range(0, len(sample_ids), 500):
    cur.execute("SELECT id, text, title, author, genre, source_type FROM samples WHERE id = ANY(%s)",
                (sample_ids[i:i+500],))
    for sid, t, ti, au, g, st in cur.fetchall():
        texts[int(sid)] = {"text": t, "title": ti, "author": au, "genre": g, "source_type": st}
conn.close()

# join
joined = []
for r in rows:
    sid = r["sample_id"]
    if sid in texts:
        joined.append({
            "username": r["username"],
            "sample_id": sid,
            "is_poetry": r["is_poetry"],
            "title": texts[sid]["title"],
            "author": texts[sid]["author"],
            "genre": texts[sid]["genre"],
            "source_type": texts[sid]["source_type"],
        })

print(f"total joined rows: {len(joined)}")

# Per-annotator stats
by_user = collections.defaultdict(list)
for r in joined:
    by_user[r["username"]].append(r)

print()
print("=== per-annotator overall (on all samples they labeled) ===")
print(f"{'annotator':20s} {'n':>5s} {'is_poem=true rate':>20s} {'true_pos_rate':>15s}")
for u in sorted(by_user):
    ann = by_user[u]
    n = len(ann)
    n_true = sum(1 for r in ann if r["is_poetry"])
    n_db_poem = sum(1 for r in ann if r["genre"] == "poem")
    n_human_said_poem_among_db_poem = sum(
        1 for r in ann if r["genre"] == "poem" and r["is_poetry"])
    print(f"  {u:20s} {n:>5d} {n_true/n*100:>19.1f}% "
          f"{n_human_said_poem_among_db_poem/max(n_db_poem,1)*100:>14.1f}%")

# Now, only on disagreements: 215 samples, indicator disagrees with majority
disagreement_ids = {d["sample_id"] for d in items}
print()
print(f"=== disagreement distribution by annotator ===")
ann_in_disag = collections.Counter()
for r in joined:
    if r["sample_id"] in disagreement_ids:
        ann_in_disag[r["username"]] += 1
for u, n in ann_in_disag.most_common():
    print(f"  {u:20s} {n}")

# Categorize disagreements by annotator + direction
# We need each annotation row that's a disagreement (where this annotator's vote != indicator's pred)
# Get indicator's pred per sample
ind_per_sample = {}
for d in items:
    sid = d["sample_id"]
    ind_per_sample[sid] = {
        "pred": d["indicator_pred"],
        "prob": d["indicator_prob"],
        "hum_major": d["majority_label"],
    }

# per-annotator agreement with indicator, broken down by direction
print()
print("=== per-annotator agreement with indicator on disagreements ===")
print(f"{'annotator':20s} {'direction':20s} {'n':>4s}  {'hum=1,ind=0':>15s} {'hum=0,ind=1':>15s}")
for u in sorted(by_user):
    ann = [r for r in by_user[u] if r["sample_id"] in disagreement_ids]
    if not ann:
        continue
    n_h1i0 = sum(1 for r in ann if ind_per_sample[r["sample_id"]]["pred"] == 0 and r["is_poetry"])
    n_h0i1 = sum(1 for r in ann if ind_per_sample[r["sample_id"]]["pred"] == 1 and not r["is_poetry"])
    print(f"  {u:20s} {'(any disagreement)':20s} {len(ann):>4d}  "
          f"{n_h1i0:>15d} {n_h0i1:>15d}")

# Now show per-annotator breakdown of hum=1,ind=0 (humans say poem, indicator says no)
# These are the cases where annotators claim text is a poem but indicator says no
# Earlier I saw many were news/social despite title saying poem.
# Without title visible, the annotator is just wrong on the text.
print()
print("=== hum=1, ind=0 (annotator says poem, indicator says no) ===")
print("    These are cases where the annotator voted is_poem=true on a text that the")
print("    indicator says is NOT a poem. Annotators DO NOT see title per user.")
print()
for u in sorted(by_user):
    cases = [r for r in by_user[u]
             if r["sample_id"] in disagreement_ids
             and ind_per_sample[r["sample_id"]]["pred"] == 0
             and r["is_poetry"]]
    if not cases:
        continue
    print(f"  {u}: {len(cases)} cases")
    for c in cases[:3]:
        # truncate text
        text_preview = ""
        conn2 = psycopg2.connect(
            host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
            port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
            dbname="neondb", sslmode="require", connect_timeout=15)
        cur2 = conn2.cursor()
        cur2.execute("SELECT text FROM samples WHERE id = %s", (c["sample_id"],))
        row = cur2.fetchone()
        text_preview = (row[0] if row else "")[:80].replace("\n", " ")
        conn2.close()
        print(f"    #{c['sample_id']} DBgenre={c['genre']}/{c['source_type']:8s}  '{text_preview}...'")

print()
print("=== hum=0, ind=1 (annotator says nonpoem, indicator says poem) ===")
print("    These are cases where the annotator voted is_poem=false on a text that")
print("    the indicator says IS a poem (typically classical poetry).")
print()
for u in sorted(by_user):
    cases = [r for r in by_user[u]
             if r["sample_id"] in disagreement_ids
             and ind_per_sample[r["sample_id"]]["pred"] == 1
             and not r["is_poetry"]]
    if not cases:
        continue
    print(f"  {u}: {len(cases)} cases")
    for c in cases[:3]:
        conn2 = psycopg2.connect(
            host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
            port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
            dbname="neondb", sslmode="require", connect_timeout=15)
        cur2 = conn2.cursor()
        cur2.execute("SELECT text FROM samples WHERE id = %s", (c["sample_id"],))
        row = cur2.fetchone()
        text_preview = (row[0] if row else "")[:80].replace("\n", " ")
        conn2.close()
        print(f"    #{c['sample_id']} DBgenre={c['genre']}/{c['source_type']:8s}  '{text_preview}...'")
"""Check why annotator_06's IDs aren't in Neon DB."""
import csv
import json
import psycopg2
from pathlib import Path
from collections import Counter

# Load annotator_06 specifically
EXPORT6 = Path(r"E:\生成诗歌\annotations_export6.csv")
rows_06 = []
with EXPORT6.open("r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["sample_id"] = int(r["sample_id"])
        r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
        rows_06.append(r)
print(f"annotator_06 total: {len(rows_06)}")
print(f"unique sample_ids: {len(set(r['sample_id'] for r in rows_06))}")
print(f"sample_id range: {min(r['sample_id'] for r in rows_06)} - {max(r['sample_id'] for r in rows_06)}")

# show first 10 sample_ids
ids = sorted(set(r["sample_id"] for r in rows_06))
print(f"first 10: {ids[:10]}")
print(f"last 10: {ids[-10:]}")

# Check if these IDs exist in Neon
conn = psycopg2.connect(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
    dbname="neondb", sslmode="require", connect_timeout=15)
cur = conn.cursor()
# batch check
for i in range(0, len(ids), 500):
    batch = ids[i:i+500]
    cur.execute("SELECT id FROM samples WHERE id = ANY(%s)", (batch,))
    found = {r[0] for r in cur.fetchall()}
    print(f"  batch {i}: requested {len(batch)}, found {len(found)}")
conn.close()

# look at one example - annotator_06's labels
print("\n=== annotator_06 examples ===")
print("first 5 rows:")
for r in rows_06[:5]:
    print(f"  sample#{r['sample_id']} '{r['title']}' by '{r['author']}'  "
          f"genre={r['truth_genre']} src={r['source_type']}  is_poetry={r['is_poetry']}")

# also check annotator_01
EXPORT = Path(r"E:\生成诗歌\annotations_export.csv")
rows_01 = []
with EXPORT.open("r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["sample_id"] = int(r["sample_id"])
        r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
        rows_01.append(r)
print(f"\nannotator_01 total: {len(rows_01)}")
print(f"annotator_01 sample_id range: "
      f"{min(r['sample_id'] for r in rows_01)} - {max(r['sample_id'] for r in rows_01)}")
ids_01 = sorted(set(r["sample_id"] for r in rows_01))
print(f"first 10: {ids_01[:10]}")
print(f"last 10: {ids_01[-10:]}")

# Check if these IDs exist in Neon
conn = psycopg2.connect(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
    dbname="neondb", sslmode="require", connect_timeout=15)
cur = conn.cursor()
cur.execute("SELECT id FROM samples WHERE id = ANY(%s)", (ids_01,))
found_01 = {r[0] for r in cur.fetchall()}
print(f"annotator_01: requested {len(ids_01)}, found in DB {len(found_01)}")
conn.close()
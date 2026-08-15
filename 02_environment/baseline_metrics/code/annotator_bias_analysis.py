"""Annotator-level bias analysis.

Key questions:
  - Each annotator's bias direction (lean yes / lean no)
  - Each annotator's tendency to vote "yes poem" on non-poem text
  - Each annotator's tendency to vote "no poem" on real poetry text
"""

import csv
import collections
import psycopg2
from pathlib import Path

CSV_PATH = Path(r"E:\生成诗歌\eval-annotation\backups\annotations_hk.csv")

# Load CSV + texts
rows = []
with CSV_PATH.open("r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["sample_id"] = int(r["sample_id"])
        r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
        rows.append(r)

conn = psycopg2.connect(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
    dbname="neondb", sslmode="require", connect_timeout=15)
cur = conn.cursor()
sample_ids = sorted({r["sample_id"] for r in rows})
texts = {}
for i in range(0, len(sample_ids), 500):
    cur.execute("SELECT id, text, title, genre, source_type FROM samples WHERE id = ANY(%s)",
                (sample_ids[i:i+500],))
    for sid, t, ti, g, st in cur.fetchall():
        texts[int(sid)] = {"text": t, "title": ti, "genre": g, "source_type": st}
conn.close()

# Per-annotator breakdown
print("=" * 70)
print("PER-ANNOTATOR BREAKDOWN")
print("=" * 70)

# Group rows by annotator
by_user = collections.defaultdict(list)
for r in rows:
    if r["sample_id"] in texts:
        r["title"] = texts[r["sample_id"]]["title"]
        r["genre"] = texts[r["sample_id"]]["genre"]
        r["source_type"] = texts[r["sample_id"]]["source_type"]
        by_user[r["username"]].append(r)

for u in sorted(by_user):
    ann = by_user[u]
    n = len(ann)
    n_yes = sum(1 for r in ann if r["is_poetry"])
    # When DB says poem, how often did annotator agree?
    db_poem = [r for r in ann if r["genre"] == "poem"]
    db_nonpoem = [r for r in ann if r["genre"] == "nonpoem"]
    n_yes_on_db_poem = sum(1 for r in db_poem if r["is_poetry"])
    n_yes_on_db_nonpoem = sum(1 for r in db_nonpoem if r["is_poetry"])

    print(f"\n--- {u} (n={n}) ---")
    print(f"  overall: {n_yes}/{n} = {n_yes/n*100:.1f}% is_poem=true")
    print(f"  when DB=poem ({len(db_poem)} cases):    "
          f"{n_yes_on_db_poem}/{len(db_poem)} = "
          f"{n_yes_on_db_poem/max(len(db_poem),1)*100:.1f}% agreed")
    print(f"  when DB=nonpoem ({len(db_nonpoem)} cases): "
          f"{n_yes_on_db_nonpoem}/{len(db_nonpoem)} = "
          f"{n_yes_on_db_nonpoem/max(len(db_nonpoem),1)*100:.1f}% disagreed")

# Show specific examples per annotator
print()
print("=" * 70)
print("SPECIFIC EXAMPLES: When each annotator disagreed with DB")
print("=" * 70)

for u in sorted(by_user):
    ann = by_user[u]
    # cases where annotator voted is_poem=true but DB says nonpoem
    yes_on_nonpoem = [r for r in ann if r["genre"] == "nonpoem" and r["is_poetry"]]
    # cases where annotator voted is_poem=false but DB says poem
    no_on_poem = [r for r in ann if r["genre"] == "poem" and not r["is_poetry"]]

    if yes_on_nonpoem:
        print(f"\n--- {u}: voted is_poem=true on NON-POEMS ({len(yes_on_nonpoem)} cases) ---")
        for r in yes_on_nonpoem[:3]:
            t = texts[r["sample_id"]]["text"][:80].replace("\n", " ")
            print(f"  #{r['sample_id']} src={r['source_type']:8s}  '{t}...'")

    if no_on_poem:
        print(f"\n--- {u}: voted is_poem=false on POEMS ({len(no_on_poem)} cases) ---")
        for r in no_on_poem[:3]:
            t = texts[r["sample_id"]]["text"][:80].replace("\n", " ")
            print(f"  #{r['sample_id']} genre={r['genre']}/src={r['source_type']:8s}  '{t}...'")

# DB label vs annotator label crosstab
print()
print("=" * 70)
print("DB GENRE × ANNOTATOR VOTE CROSSTAB")
print("=" * 70)
print(f"{'annotator':20s} | DB=poem | DB=nonpoem | Total")
print(f"{'':20s} | yes/no  | yes/no    |")
for u in sorted(by_user):
    ann = by_user[u]
    db_poem = [r for r in ann if r["genre"] == "poem"]
    db_non = [r for r in ann if r["genre"] == "nonpoem"]
    yp = sum(1 for r in db_poem if r["is_poetry"])
    np = len(db_poem) - yp
    yn = sum(1 for r in db_non if r["is_poetry"])
    nn = len(db_non) - yn
    print(f"{u:20s} | {yp:3d}/{np:3d} | {yn:3d}/{nn:3d}  | {len(ann)}")

# Where annotator_01 votes yes on news/social (specific examples)
print()
print("=" * 70)
print("annotator_01's most striking cases (votes yes on news/social):")
print("=" * 70)
ann_01 = by_user.get("annotator_01", [])
strange_yes = [r for r in ann_01
              if r["is_poetry"] and r["genre"] == "nonpoem"
              and r["source_type"] in ("news", "social")]
print(f"Total: {len(strange_yes)} cases of annotator_01 saying 'yes poem' on news/social")
for r in strange_yes[:5]:
    t = texts[r["sample_id"]]["text"][:120].replace("\n", " ")
    print(f"  #{r['sample_id']} ({r['source_type']})  '{t}'")

# annotator_06 / 04 specific cases
print()
print("=" * 70)
print("annotator_06 / annotator_04's cases (votes no on actual poetry):")
print("=" * 70)
for u in ("annotator_06", "annotator_04"):
    if u not in by_user:
        continue
    ann = by_user[u]
    no_on_poem = [r for r in ann if r["genre"] == "poem" and not r["is_poetry"]]
    print(f"\n--- {u} voted NO on POEMS ({len(no_on_poem)} cases) ---")
    for r in no_on_poem:
        t = texts[r["sample_id"]]["text"][:120].replace("\n", " ")
        print(f"  #{r['sample_id']} ({r['genre']}/{r['source_type']})  '{t}'")
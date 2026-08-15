"""Pull all disagreements with full annotator info for review.

Output: a clean table of each disagreement, showing:
  - sample_id, text preview
  - who labeled it (annotator)
  - what they voted
  - what indicator predicted
  - what DB says
"""

import csv
import json
import psycopg2
from pathlib import Path

CSV_PATH = Path(r"E:\生成诗歌\eval-annotation\backups\annotations_hk.csv")
ROUND3_LOG = Path(r"E:\ai4s\poetry-poetricity\04_memory\experiment_logs\stage2_round_003.json")
OUT_DIR = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_005")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load indicator's predictions on disagreement samples
log = json.loads(ROUND3_LOG.read_text(encoding="utf-8"))
disagreements = log["disagreements"]
ind_by_sid = {d["sample_id"]: d for d in disagreements}

# Load all annotations + texts
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

# For each annotator, pull the disagreements they caused
print("=" * 80)
print("DISAGREEMENTS BY ANNOTATOR (full text per case)")
print("=" * 80)

# We need: which annotations by each user are disagreements
# An annotation is a "disagreement" if:
#   indicator_pred != annotator's vote
# OR if it contributes to a majority that's != indicator_pred
# Simplest: if sample_id is in disagreements list and this annotator voted != indicator

# Build per-sample rater map
from collections import defaultdict
per_sample_raters = defaultdict(list)
for r in rows:
    if r["sample_id"] in ind_by_sid:
        per_sample_raters[r["sample_id"]].append(r)

# Group disagreements by annotator
ann_disag = defaultdict(list)
for sid, raters in per_sample_raters.items():
    ind = ind_by_sid[sid]
    for r in raters:
        # Is this annotator's vote different from indicator?
        annot_voted_poem = r["is_poetry"]
        if annot_voted_poem != ind["indicator_pred"]:
            ann_disag[r["username"]].append({
                "sample_id": sid,
                "annot_voted_poem": annot_voted_poem,
                "ind_pred": ind["indicator_pred"],
                "ind_prob": ind["indicator_prob"],
                "hum_major": ind["majority_label"],
                "db_genre": texts[sid]["genre"],
                "db_source_type": texts[sid]["source_type"],
                "db_title": texts[sid]["title"],
                "text": texts[sid]["text"],
            })

for u in sorted(ann_disag):
    cases = ann_disag[u]
    n_yes_on_ind_no = sum(1 for c in cases
                            if c["annot_voted_poem"] and not c["ind_pred"])
    n_no_on_ind_yes = sum(1 for c in cases
                            if not c["annot_voted_poem"] and c["ind_pred"])
    print(f"\n{'='*80}")
    print(f"### {u}: {len(cases)} disagreements total")
    print(f"    - annotator voted YES but indicator NO: {n_yes_on_ind_no}")
    print(f"    - annotator voted NO  but indicator YES: {n_no_on_ind_yes}")
    print(f"{'='*80}")

    # Show each case with text
    for c in cases:
        direction = "ANNOT_YES_ind_NO" if c["annot_voted_poem"] else "ANNOT_NO_ind_YES"
        print(f"\n  [{direction}] sample#{c['sample_id']}  "
              f"DB:{c['db_genre']}/{c['db_source_type']:8s}  "
              f"ind_prob={c['ind_prob']:.2f}")
        print(f"  TITLE (DB): {c['db_title']!r}")
        print(f"  TEXT: {c['text'][:160].replace(chr(10), ' ')}...")

# Save structured output
ann_disag_full = []
for u, cases in ann_disag.items():
    for c in cases:
        ann_disag_full.append({
            "annotator": u,
            **c,
        })
(OUT_DIR / "disagreements_by_annotator.json").write_text(
    json.dumps(ann_disag_full, ensure_ascii=False, indent=2),
    encoding="utf-8")
print(f"\n\nsaved {len(ann_disag_full)} disagreements-by-annotator records")
"""Identify annotator_01's clearly bad labels (likely keyword-matching bias)."""
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze  # noqa: E402

ALL_PATH = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_006\all_annotations.json")
OUT_DIR = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_006")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = json.loads(ALL_PATH.read_text(encoding="utf-8"))

# filter annotator_01 only
ann_01 = [r for r in rows if r["username"] == "annotator_01"]
print(f"annotator_01 total: {len(ann_01)}")

# get texts (we already have them in all_annotations; need to fetch)
import psycopg2
sample_ids = sorted({r["sample_id"] for r in ann_01})
conn = psycopg2.connect(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
    dbname="neondb", sslmode="require", connect_timeout=15)
cur = conn.cursor()
texts = {}
for i in range(0, len(sample_ids), 500):
    cur.execute("SELECT id, text FROM samples WHERE id = ANY(%s)", (sample_ids[i:i+500],))
    for sid, text in cur.fetchall():
        texts[int(sid)] = text
conn.close()

# apply frozen metric
print("\n[metric] applying frozen metric to annotator_01's samples ...", flush=True)
fm = build_and_freeze(seed=42, val_ratio=0.2)

# cases to inspect: where annotator_01 voted differently from indicator
candidates = []
for r in ann_01:
    sid = r["sample_id"]
    if sid not in texts:
        continue
    text = texts[sid]
    pred = fm.apply(text)
    ind_pred = pred.pred
    ind_prob = pred.prob_poem
    annot_vote = 1 if r["is_poetry"] else 0
    if ind_pred != annot_vote:
        candidates.append({
            "sample_id": sid,
            "annotator": "annotator_01",
            "annot_vote_poem": bool(annot_vote),
            "indicator_pred": int(ind_pred),
            "indicator_prob": float(ind_prob),
            "db_genre": r["truth_genre"],
            "db_source_type": r["source_type"],
            "db_title": r["title"],
            "text": text,
        })

# separate by direction
yes_on_ind_no = [c for c in candidates if c["annot_vote_poem"] and not c["indicator_pred"]]
no_on_ind_yes = [c for c in candidates if not c["annot_vote_poem"] and c["indicator_pred"]]
print(f"\n=== annotator_01 disagreements with indicator ===")
print(f"  ANNOT voted YES but indicator NO: {len(yes_on_ind_no)}")
print(f"  ANNOT voted NO  but indicator YES: {len(no_on_ind_yes)}")

# show top examples by indicator confidence
print("\n=== ANNOT voted YES but indicator NO (annotator_01 likely wrong) ===")
yes_on_ind_no.sort(key=lambda c: -c["indicator_prob"])
for c in yes_on_ind_no[:10]:
    t = c["text"][:120].replace("\n", " ")
    print(f"  #{c['sample_id']} DB={c['db_genre']}/{c['db_source_type']:8s}  "
          f"ind_prob={c['indicator_prob']:.2f}  '{t}...'")

print("\n=== ANNOT voted NO but indicator YES (annotator_01 likely wrong) ===")
no_on_ind_yes.sort(key=lambda c: -c["indicator_prob"])
for c in no_on_ind_yes[:10]:
    t = c["text"][:120].replace("\n", " ")
    print(f"  #{c['sample_id']} DB={c['db_genre']}/{c['db_source_type']:8s}  "
          f"ind_prob={c['indicator_prob']:.2f}  '{t}...'")

# apply heuristics: annotator_01's votes that are HIGH-CONFIDENCE disagreements
# with the indicator (prob > 0.85) are very likely keyword-matching noise
high_conf_yes_on_ind_no = [c for c in yes_on_ind_no if c["indicator_prob"] > 0.85]
high_conf_no_on_ind_yes = [c for c in no_on_ind_yes if c["indicator_prob"] > 0.85]
print(f"\n=== high-confidence (ind_prob > 0.85) annotator_01 wrong-likely labels ===")
print(f"  YES on indicator NO  (annotator voted yes on news): {len(high_conf_yes_on_ind_no)}")
print(f"  NO  on indicator YES (annotator voted no on poem):  {len(high_conf_no_on_ind_yes)}")

# Save candidates for inspection
OUT_DIR.joinpath("annotator01_candidates.json").write_text(
    json.dumps(candidates, ensure_ascii=False, indent=2),
    encoding="utf-8")
print(f"\nsaved {len(candidates)} candidates to {OUT_DIR/'annotator01_candidates.json'}")
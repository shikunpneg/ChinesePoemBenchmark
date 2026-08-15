"""Fetch full texts for the 215 disagreement samples."""
import json
import psycopg2
from pathlib import Path

DIS_PATH = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_004\disagreements_to_review.json")
OUT_PATH = DIS_PATH  # overwrite with full text

disagreements = json.loads(DIS_PATH.read_text(encoding="utf-8"))
sample_ids = sorted({d["sample_id"] for d in disagreements})
print(f"need {len(sample_ids)} unique sample_ids")

conn = psycopg2.connect(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
    dbname="neondb", sslmode="require", connect_timeout=15)
cur = conn.cursor()
texts = {}
chunk = 500
for i in range(0, len(sample_ids), chunk):
    batch = sample_ids[i:i + chunk]
    cur.execute("SELECT id, text FROM samples WHERE id = ANY(%s)", (batch,))
    for sid, text in cur.fetchall():
        texts[int(sid)] = text
conn.close()
print(f"got {len(texts)} texts")

# attach full text
out = []
for d in disagreements:
    sid = d["sample_id"]
    full = texts.get(sid, "")
    out.append({**d, "text_full": full})
OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
print(f"saved {len(out)} entries with full text to {OUT_PATH}")
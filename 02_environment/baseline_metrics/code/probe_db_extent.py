"""Investigate the ID range discrepancy."""
import psycopg2
import csv
from pathlib import Path

# Check DB extent
conn = psycopg2.connect(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
    dbname="neondb", sslmode="require", connect_timeout=15)
cur = conn.cursor()
cur.execute("SELECT MIN(id), MAX(id), COUNT(*) FROM samples")
min_id, max_id, count = cur.fetchone()
print(f"DB samples: {count} rows, id range {min_id} - {max_id}")

# breakdown by source_type
cur.execute("SELECT source_type, COUNT(*) FROM samples GROUP BY source_type")
for s, c in cur.fetchall():
    print(f"  source_type={s}: {c}")

# breakdown by genre
cur.execute("SELECT genre, COUNT(*) FROM samples GROUP BY genre")
for g, c in cur.fetchall():
    print(f"  genre={g}: {c}")

# Check if there are other tables
cur.execute("""
    SELECT table_schema, table_name FROM information_schema.tables
    WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
    AND table_name NOT LIKE 'pg_%'
    ORDER BY table_schema, table_name
""")
print("\n=== all tables ===")
for s, t in cur.fetchall():
    print(f"  {s}.{t}")

# Check if there's an alternate samples-related table
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_name ILIKE '%sample%' OR table_name ILIKE '%annot%'
""")
print("\n=== sample/annotation related tables ===")
for (t,) in cur.fetchall():
    print(f"  {t}")

# Try ai samples count
cur.execute("SELECT COUNT(*) FROM samples WHERE source_type='ai'")
print(f"\nai samples in DB: {cur.fetchone()[0]}")

# IDs of ai samples
cur.execute("SELECT MIN(id), MAX(id) FROM samples WHERE source_type='ai'")
mn, mx = cur.fetchone()
print(f"  ai sample id range: {mn} - {mx}")

# Check ID 109953 specifically
cur.execute("SELECT id, title, author, genre, source_type FROM samples WHERE id = 109953")
row = cur.fetchone()
if row:
    print(f"\nid=109953 exists: {row}")
else:
    print(f"\nid=109953 DOES NOT exist in DB")

# look at IDs around annotator_06's range
cur.execute("SELECT MIN(id), MAX(id) FROM samples WHERE id >= 109500")
mn, mx = cur.fetchone()
print(f"DB samples with id >= 109500: {mn} - {mx}")

# Check if the IDs come from a different DB altogether
print("\n=== probe: try connecting to other servers ===")
# check wirter.com (production in HK)
try:
    c2 = psycopg2.connect(
        host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
        port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
        dbname="neondb", sslmode="require", connect_timeout=15)
    cur2 = c2.cursor()
    cur2.execute("SELECT COUNT(*), MAX(id) FROM samples")
    print(f"  current DB: {cur2.fetchone()}")
    c2.close()
except Exception as e:
    print(f"  failed: {e}")

conn.close()
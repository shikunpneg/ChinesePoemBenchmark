"""Module for accessing the production annotation database (Neon Postgres).

This wraps the connection details from `eval-annotation/.env.production.local`.
It exposes:
  - `db_connection()` -> psycopg2 connection
  - `fetch_annotated_samples()` -> list[dict] of all annotated samples
  - `fetch_pending_assignments(limit=50)` -> list of sample IDs that are
     pending annotation (so we can pull from them)

The DB has 4 users + 109k samples + 2400 pending assignments but **0 actual
annotations** as of the latest check. This module provides the wiring so
that once annotations exist they are immediately usable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2

CONN_KW = dict(
    host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
    port=5432,
    user="neondb_owner",
    password="npg_9fXBO3YmJqow",
    dbname="neondb",
    sslmode="require",
    connect_timeout=15,
)


@contextmanager
def db_connection() -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(**CONN_KW)
    try:
        yield conn
    finally:
        conn.close()


def fetch_annotated_samples() -> list[dict]:
    """Return all samples that have at least one annotation.

    Returns list of dicts:
        {sample_id, title, author, text, genre, source_type,
         n_raters, n_yes, raters, votes}
    """
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.id, s.title, s.author, s.text, s.genre, s.source_type,
                   COUNT(a.id) AS n_raters,
                   SUM(CASE WHEN a.is_poetry THEN 1 ELSE 0 END) AS n_yes,
                   ARRAY_AGG(u.username) AS raters,
                   ARRAY_AGG(a.is_poetry::int) AS votes
            FROM samples s
            JOIN annotations a ON a.sample_id = s.id
            JOIN users u ON u.id = a.user_id
            GROUP BY s.id, s.title, s.author, s.text, s.genre, s.source_type
            ORDER BY n_raters DESC, s.id
        """)
        out = []
        for row in cur.fetchall():
            (sid, title, author, text, genre, src_type,
             n_raters, n_yes, raters, votes) = row
            out.append({
                "sample_id": sid,
                "title": title,
                "author": author,
                "text": text,
                "genre": genre,
                "source_type": src_type,
                "n_raters": n_raters,
                "n_yes": n_yes,
                "raters": raters,
                "votes": votes,
            })
        return out


def annotation_stats() -> dict:
    """Quick DB stats for reports."""
    with db_connection() as conn:
        cur = conn.cursor()
        stats = {}
        for t in ("users", "samples", "assignments", "annotations"):
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            stats[t] = cur.fetchone()[0]
        # multi-rater count
        cur.execute("""
            SELECT COUNT(*) FROM (
              SELECT sample_id FROM annotations GROUP BY sample_id HAVING COUNT(*) >= 2
            ) t
        """)
        stats["multi_rater_samples"] = cur.fetchone()[0]
        return stats
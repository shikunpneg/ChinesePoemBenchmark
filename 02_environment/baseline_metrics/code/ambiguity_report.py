"""Fetch texts for the ambiguous samples and build the full ambiguity report.

Ambiguity categories:
  A. 多标注者分歧 (annotators disagree on is_poetry)
  B. human 真诗被判非诗 (likely annotation error)
  C. AI 仿诗被判为诗 + quality>=4 (AI fooled annotator)
  D. 低质量 (quality 1-2)
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import psycopg2

F6 = Path(r"E:\生成诗歌\annotations_task6_r1.csv")
F7 = Path(r"E:\生成诗歌\annotations_task7_r1.csv")
HARD_FILES = [
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl"),
]
EXPERT_JS = Path(r"E:\生成诗歌\eval-annotation\data\samples.js")


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


def fetch_texts(sample_ids):
    """Fetch texts from Neon DB by sample_id."""
    texts = {}
    conn = psycopg2.connect(
        host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
        port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
        dbname="neondb", sslmode="require", connect_timeout=15)
    cur = conn.cursor()
    ids = sorted(sample_ids)
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        cur.execute("SELECT id, text, title, author FROM samples WHERE id = ANY(%s)", (batch,))
        for sid, text, title, author in cur.fetchall():
            texts[int(sid)] = {"text": text, "title": title, "author": author}
    conn.close()
    return texts


def main():
    r6, r7 = load(F6), load(F7)
    all_rows = r6 + r7
    per_s = defaultdict(list)
    for r in all_rows:
        per_s[r["sample_id"]].append(r)
    ids = set(per_s.keys())
    print(f"total unique sample_ids: {len(ids)}")

    # fetch texts from Neon
    db_texts = fetch_texts(ids)
    print(f"fetched {len(db_texts)} texts from Neon")

    # also index AI texts by (title, author)
    ai_texts = {}
    for path in HARD_FILES:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = (obj.get("title", ""), obj.get("model", ""))
                ai_texts.setdefault(key, []).append(obj.get("generated", ""))
    # expert js human texts by title
    human_texts = {}
    with EXPERT_JS.open("r", encoding="utf-8") as f:
        raw = f.read()
    for m in re.finditer(r"\{ title: \"([^\"]+)\".*?text: \"((?:[^\"\\]|\\.)*)\".*?genre: \"(poem|nonpoem)\"", raw, re.DOTALL):
        title, text, genre = m.group(1), m.group(2), m.group(3)
        text = text.replace("\\n", "\n").replace('\\"', '"')
        human_texts[title] = text

    def get_text(sid, row):
        if sid in db_texts:
            return db_texts[sid]["text"]
        # try AI by title+author
        key = (row["title"], row["author"])
        if key in ai_texts and ai_texts[key]:
            return ai_texts[key][0]
        if row["title"] in human_texts:
            return human_texts[row["title"]]
        return None

    # ---- category A: multi-annotator disagreement ----
    print("\n" + "=" * 76)
    print("类别 A: 多标注者分歧（同一 AI 诗，有人说是诗有人说不是）")
    print("=" * 76)
    for sid in sorted(per_s):
        anns = per_s[sid]
        if len(anns) < 2:
            continue
        yes = sum(1 for a in anns if a["is_poetry"])
        if yes == 0 or yes == len(anns):
            continue
        row0 = anns[0]
        votes = " / ".join(f"{a['username'][-2:]}:{'诗' if a['is_poetry'] else '非'}"
                           for a in sorted(anns, key=lambda x: x["username"]))
        txt = get_text(sid, row0)
        preview = (txt[:80].replace("\n", " / ") + "…") if txt else "(no text)"
        print(f"  #{sid} {row0['person']}「{row0['title']}」[{votes}] origin={row0['origin']}")
        print(f"      {preview}")

    # ---- category B: human poem judged non-poem ----
    print("\n" + "=" * 76)
    print("类别 B: human 真诗被判为非诗（可能标注错误）")
    print("=" * 76)
    for sid in sorted(per_s):
        anns = per_s[sid]
        row0 = anns[0]
        if row0["origin"] == "human" and not row0["is_poetry"]:
            txt = get_text(sid, row0)
            preview = (txt[:100].replace("\n", " / ") + "…") if txt else "(no text)"
            print(f"  #{sid} {row0['person']}「{row0['title']}」标注者={row0['username']}")
            print(f"      {preview}")

    # ---- category C: AI poem judged poem + quality>=4 ----
    print("\n" + "=" * 76)
    print("类别 C: AI 仿诗被判为诗 且 quality>=4（AI 骗过标注者）")
    print("=" * 76)
    for sid in sorted(per_s):
        anns = per_s[sid]
        for a in anns:
            if a["origin"] == "ai" and a["is_poetry"] and a["quality_grade"] and a["quality_grade"] >= 4:
                txt = get_text(sid, a)
                preview = (txt[:80].replace("\n", " / ") + "…") if txt else "(no text)"
                print(f"  #{sid} {a['person']}「{a['title']}」q={a['quality_grade']} "
                      f"标注者={a['username']}")
                print(f"      {preview}")
                break


if __name__ == "__main__":
    main()
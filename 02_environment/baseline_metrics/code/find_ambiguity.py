"""Find ambiguous samples: disagreement between annotators on same sample,
human poems judged as non-poem, low quality grades, etc."""
import csv
from collections import defaultdict
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


def per_sample(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["sample_id"]].append(r)
    return d


def ambiguity_report(name, rows):
    print(f"\n{'='*72}")
    print(f"=== {name} 歧义样本分析 ===")
    print(f"{'='*72}")
    per_s = per_sample(rows)

    # 1) same sample annotated by multiple users -> disagreement
    multi = {sid: v for sid, v in per_s.items() if len(v) >= 2}
    print(f"\n[1] 多标注者样本: {len(multi)} 个")
    for sid, anns in sorted(multi.items()):
        votes = [(a["username"], 1 if a["is_poetry"] else 0,
                  a["quality_grade"]) for a in anns]
        yes = sum(1 for a in anns if a["is_poetry"])
        if 0 < yes < len(anns):  # disagreement
            print(f"  sample#{sid} 分歧! {votes}  "
                  f"origin={anns[0]['origin']} person={anns[0]['person']}")

    # 2) human-origin poems judged as non-poem
    print(f"\n[2] human 真诗被判为非诗 (歧义/可能标注错误):")
    for sid, anns in sorted(per_s.items()):
        a = anns[0]
        if a["origin"] == "human" and not a["is_poetry"]:
            print(f"  sample#{sid} {a['person']} {a['title']!r} "
                  f"annotator={a['username']}")

    # 3) AI poems judged as poem with HIGH quality (AI 骗过)
    print(f"\n[3] AI 仿诗被判为是诗 且 quality>=4 (AI 骗过标注者):")
    for sid, anns in sorted(per_s.items()):
        a = anns[0]
        if a["origin"] == "ai" and a["is_poetry"] and a["quality_grade"] and a["quality_grade"] >= 4:
            print(f"  sample#{sid} {a['person']} {a['title']!r} q={a['quality_grade']} "
                  f"annotator={a['username']}")

    # 4) low quality (quality 1-2) - arguably not good poetry
    print(f"\n[4] 低质量评分 (quality 1-2):")
    for sid, anns in sorted(per_s.items()):
        a = anns[0]
        if a["quality_grade"] and a["quality_grade"] <= 2:
            print(f"  sample#{sid} {a['person']} {a['title']!r} "
                  f"q={a['quality_grade']} origin={a['origin']} is_poetry={a['is_poetry']}")


r6 = load(F6)
r7 = load(F7)
ambiguity_report("annotations_task6_r1.csv", r6)
ambiguity_report("annotations_task7_r1.csv", r7)
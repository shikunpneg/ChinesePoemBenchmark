"""Save ambiguity analysis results to memory as structured JSON."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import psycopg2

F6 = Path(r"E:\生成诗歌\annotations_task6_r1.csv")
F7 = Path(r"E:\生成诗歌\annotations_task7_r1.csv")
OUT = Path(r"E:\ai4s\poetry-poetricity\04_memory\failures\ambiguity_task67.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

HARD_FILES = [
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_LiBai.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_GuCheng.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\hard_gen_Haizi-CN.jsonl"),
    Path(r"E:\生成诗歌\ChineseHardJudgePoem\data\to_annotate_near.jsonl"),
]


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


def main():
    r6, r7 = load(F6), load(F7)
    all_rows = r6 + r7
    per_s = defaultdict(list)
    for r in all_rows:
        per_s[r["sample_id"]].append(r)

    # index AI texts
    ai_texts = {}
    for path in HARD_FILES:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = (obj.get("title", ""), obj.get("model", ""))
                ai_texts.setdefault(key, []).append(obj.get("generated", ""))

    # category A: disagreement
    cat_a = []
    for sid in sorted(per_s):
        anns = per_s[sid]
        if len(anns) < 2:
            continue
        yes = sum(1 for a in anns if a["is_poetry"])
        if yes == 0 or yes == len(anns):
            continue
        a0 = anns[0]
        cat_a.append({
            "sample_id": sid,
            "title": a0["title"],
            "person": a0["person"],
            "origin": a0["origin"],
            "votes": {a["username"]: bool(a["is_poetry"]) for a in anns},
            "quality": {a["username"]: a["quality_grade"] for a in anns},
        })

    # category B: human non-poem
    cat_b = []
    for sid in sorted(per_s):
        anns = per_s[sid]
        a0 = anns[0]
        if a0["origin"] == "human" and not a0["is_poetry"]:
            cat_b.append({
                "sample_id": sid, "title": a0["title"], "person": a0["person"],
                "annotator": a0["username"],
            })

    # category C: AI poem q>=4
    cat_c = []
    for sid in sorted(per_s):
        for a in per_s[sid]:
            if a["origin"] == "ai" and a["is_poetry"] and a["quality_grade"] and a["quality_grade"] >= 4:
                cat_c.append({
                    "sample_id": sid, "title": a["title"], "person": a["person"],
                    "quality": a["quality_grade"], "annotator": a["username"],
                })
                break

    # category D: low quality 1-2
    cat_d = []
    for sid in sorted(per_s):
        for a in per_s[sid]:
            if a["quality_grade"] and a["quality_grade"] <= 2:
                cat_d.append({
                    "sample_id": sid, "title": a["title"], "person": a["person"],
                    "quality": a["quality_grade"], "origin": a["origin"],
                })
                break

    report = {
        "source": ["annotations_task6_r1.csv", "annotations_task7_r1.csv"],
        "total_rows": len(all_rows),
        "unique_samples": len(per_s),
        "cat_a_disagreement": {"n": len(cat_a), "samples": cat_a},
        "cat_b_human_nonpoem": {"n": len(cat_b), "samples": cat_b},
        "cat_c_ai_high_quality": {"n": len(cat_c), "samples": cat_c},
        "cat_d_low_quality": {"n": len(cat_d), "samples": cat_d},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT}")
    print(f"  cat_a (disagreement): {len(cat_a)}")
    print(f"  cat_b (human nonpoem): {len(cat_b)}")
    print(f"  cat_c (AI q>=4): {len(cat_c)}")
    print(f"  cat_d (low quality): {len(cat_d)}")


if __name__ == "__main__":
    main()
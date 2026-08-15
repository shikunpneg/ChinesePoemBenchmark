"""Stage 2 · Round 3: REAL Human-IAA + indicator agreement.

Uses:
  - 1,764 annotation records from eval-annotation/backups/annotations_hk.csv
  - sample texts from the production Neon database (joined on sample_id)
  - the frozen Stage-1 metric

Computes:
  (1) Real inter-annotator agreement (Cohen's / Fleiss' Kappa on overlapping samples)
  (2) Indicator vs human agreement, per-annotator
  (3) Where indicator AND humans disagree (the true failure boundary)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze  # noqa: E402
from code.neon_data import db_connection  # noqa: E402

CSV_PATH = Path(r"E:\生成诗歌\eval-annotation\backups\annotations_hk.csv")
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / "round_003"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / "stage2_round_003.json"


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["sample_id"] = int(r["sample_id"])
            r["is_poetry"] = (r["is_poetry"].strip().lower() == "true")
            r["quality_grade"] = (r.get("quality_grade") or "").strip() or None
            rows.append(r)
    return rows


def fetch_texts_from_neon(sample_ids: list[int]) -> dict[int, str]:
    """Pull texts for given sample IDs from production Neon DB."""
    out = {}
    chunk_size = 500
    with db_connection() as conn:
        cur = conn.cursor()
        for i in range(0, len(sample_ids), chunk_size):
            chunk = sample_ids[i:i + chunk_size]
            cur.execute(
                "SELECT id, text FROM samples WHERE id = ANY(%s)",
                (chunk,))
            for sid, text in cur.fetchall():
                out[int(sid)] = text
    return out


def cohen_kappa(rater_a: list[int], rater_b: list[int]) -> dict:
    """Cohen's kappa (quadratic-weighted) between two raters on overlapping items."""
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        return {"n": 0, "kappa": None, "agree_rate": None}
    a = np.asarray(rater_a, dtype=np.int64)
    b = np.asarray(rater_b, dtype=np.int64)
    from sklearn.metrics import cohen_kappa_score
    kap = cohen_kappa_score(a, b, weights="quadratic")
    agree = float((a == b).mean())
    return {"n": int(len(a)), "kappa": float(kap), "agree_rate": agree}


def fleiss_kappa(items_matrix: list[list[int]], n_categories: int = 2) -> dict:
    """Fleiss' kappa for N raters x M items.

    items_matrix[i] is the list of votes (length n_raters_i) on item i.
    """
    if not items_matrix:
        return {"n_items": 0, "kappa": None}
    n_items = len(items_matrix)
    n_raters_per_item = [len(v) for v in items_matrix]
    if not all(n > 0 for n in n_raters_per_item):
        return {"n_items": n_items, "kappa": None, "note": "items with 0 raters"}
    # P_i: per-item agreement
    P = []
    N = []
    for votes in items_matrix:
        n = len(votes)
        N.append(n)
        counts = [0] * n_categories
        for v in votes:
            if 0 <= v < n_categories:
                counts[v] += 1
        if n < 2:
            P.append(1.0)
            continue
        s2 = sum(c * c for c in counts)
        P.append((s2 - n) / (n * (n - 1)))
    Pbar = float(np.mean(P))
    # p_j: marginal proportion of each category
    all_votes = [v for votes in items_matrix for v in votes]
    p = []
    for c in range(n_categories):
        p.append(sum(1 for v in all_votes if v == c) / len(all_votes))
    Pe = float(sum(pj * pj for pj in p))
    if Pe >= 1.0:
        return {"n_items": n_items, "kappa": None, "note": "Pe=1 (degenerate)"}
    kappa = (Pbar - Pe) / (1 - Pe)
    return {
        "n_items": n_items,
        "n_raters_min": int(min(N)),
        "n_raters_max": int(max(N)),
        "n_raters_mean": float(np.mean(N)),
        "P_bar": float(Pbar),
        "Pe": Pe,
        "kappa": float(kappa),
    }


def main() -> int:
    t0 = time.time()
    print("[load] reading CSV ...", flush=True)
    rows = load_csv(CSV_PATH)
    print(f"[load] {len(rows)} annotation rows", flush=True)
    by_user = Counter(r["username"] for r in rows)
    print(f"[load] by user: {dict(by_user)}", flush=True)

    # all sample_ids needed
    sample_ids = sorted({r["sample_id"] for r in rows})
    print(f"[load] unique sample_ids: {len(sample_ids)}", flush=True)

    # pull texts from Neon
    print("\n[neon] fetching texts for {} samples ...".format(len(sample_ids)),
          flush=True)
    texts = fetch_texts_from_neon(sample_ids)
    print(f"[neon] got {len(texts)} texts  (missing={len(sample_ids) - len(texts)})",
          flush=True)

    # join: only keep rows where text exists
    joined = []
    missing_ids = []
    for r in rows:
        sid = r["sample_id"]
        if sid in texts and texts[sid]:
            joined.append({**r, "text": texts[sid]})
        else:
            missing_ids.append(sid)
    print(f"[join] joined={len(joined)}  missing_ids={len(set(missing_ids))}",
          flush=True)
    if missing_ids:
        print(f"[join] first missing: {sorted(set(missing_ids))[:10]}", flush=True)

    # ---------- per-user / per-sample aggregation ----------
    by_sample = defaultdict(list)
    by_user_sample = defaultdict(dict)
    for r in joined:
        lab = 1 if r["is_poetry"] else 0
        by_sample[r["sample_id"]].append((r["username"], lab))
        by_user_sample[r["username"]][r["sample_id"]] = lab

    # ---------- multi-rater overlap ----------
    multi_sample = [(sid, v) for sid, v in by_sample.items() if len(v) >= 2]
    print(f"\n[multi-rater] samples with >=2 annotations: {len(multi_sample)}",
          flush=True)

    # ---------- pairwise Cohen ----------
    users = sorted({r["username"] for r in joined})
    print(f"\n[users] {len(users)} annotators: {users}", flush=True)
    pairwise = {}
    for i, u1 in enumerate(users):
        for u2 in users[i + 1:]:
            common = set(by_user_sample[u1].keys()) & set(by_user_sample[u2].keys())
            if not common:
                continue
            a = [by_user_sample[u1][s] for s in sorted(common)]
            b = [by_user_sample[u2][s] for s in sorted(common)]
            pairwise[f"{u1}~{u2}"] = cohen_kappa(a, b)
    print("\n[pairwise Cohen's Kappa (quadratic):]", flush=True)
    for pair, r in sorted(pairwise.items()):
        print(f"  {pair:35s}  n={r['n']:>4d}  kappa={r['kappa']:.4f}  "
              f"agree={r['agree_rate']:.4f}", flush=True)

    # ---------- Fleiss' kappa (all multi-rater samples) ----------
    items_matrix = [[v for _, v in votes] for _, votes in multi_sample]
    fleiss = fleiss_kappa(items_matrix, n_categories=2)
    print(f"\n[fleiss] {fleiss}", flush=True)

    # ---------- per-pair agreement for raters who overlap a lot ----------
    # (>=50 common samples)
    strong_pairs = {k: v for k, v in pairwise.items() if v["n"] >= 50}
    print(f"\n[strong-pair] pairs with >=50 common items:", flush=True)
    for k, v in strong_pairs.items():
        print(f"  {k:35s}  n={v['n']:>4d}  kappa={v['kappa']:.4f}", flush=True)

    # ---------- frozen indicator on the same texts ----------
    print("\n[metric] frozen metric ...", flush=True)
    fm = build_and_freeze(seed=42, val_ratio=0.2)
    # use unique texts (one prediction per sample)
    unique_samples = {}
    for r in joined:
        unique_samples.setdefault(r["sample_id"], []).append(r)
    print(f"[metric] unique samples with annotations+text: {len(unique_samples)}",
          flush=True)

    indicator_per_sample = {}
    for sid, ann_list in unique_samples.items():
        text = ann_list[0]["text"]
        if not text or not text.strip():
            continue
        pred = fm.apply(text)
        indicator_per_sample[sid] = {
            "prob_poem": pred.prob_poem,
            "pred": pred.pred,
        }

    # ---------- indicator vs human agreement ----------
    # For each sample, take the **majority** vote of humans
    majority_per_sample = {}
    for sid, ann_list in unique_samples.items():
        votes = [1 if r["is_poetry"] else 0 for r in ann_list]
        majority_per_sample[sid] = int(round(np.mean(votes)))
    sids_with_indicator = sorted(
        s for s in indicator_per_sample if s in majority_per_sample)
    y_ind = np.asarray([indicator_per_sample[s]["pred"] for s in sids_with_indicator],
                       dtype=np.int64)
    y_hum = np.asarray([majority_per_sample[s] for s in sids_with_indicator],
                       dtype=np.int64)
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    ind_acc = accuracy_score(y_hum, y_ind)
    ind_kap = cohen_kappa_score(y_hum, y_ind, weights="quadratic")
    ind_f1 = f1_score(y_hum, y_ind, average="macro")
    print(f"\n[ind vs human] acc={ind_acc:.4f} kappa={ind_kap:.4f} f1={ind_f1:.4f}",
          flush=True)

    # per-annotator agreement with indicator
    per_ann = {}
    for u in users:
        sids = sorted(set(by_user_sample[u].keys()) & set(indicator_per_sample.keys()))
        if not sids:
            continue
        yh = np.asarray([by_user_sample[u][s] for s in sids], dtype=np.int64)
        yi = np.asarray([indicator_per_sample[s]["pred"] for s in sids],
                        dtype=np.int64)
        per_ann[u] = {
            "n": len(sids),
            "acc": float(accuracy_score(yh, yi)),
            "kappa_quadratic": float(cohen_kappa_score(yh, yi, weights="quadratic")),
        }
    print("\n[ind vs each annotator]:", flush=True)
    for u, r in sorted(per_ann.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {u:20s}  n={r['n']:>4d}  acc={r['acc']:.4f}  kappa={r['kappa_quadratic']:.4f}",
              flush=True)

    # ---------- failure modes ----------
    # Where indicator AND majority-human disagree -> real failure
    disagreements = []
    for s in sids_with_indicator:
        h = majority_per_sample[s]
        i = indicator_per_sample[s]["pred"]
        if h != i:
            # find annotators who disagreed with majority
            dissenting = []
            ann_list = unique_samples[s]
            for r in ann_list:
                v = 1 if r["is_poetry"] else 0
                if v != h:
                    dissenting.append((r["username"], v))
            disagreements.append({
                "sample_id": s,
                "majority_label": h,
                "indicator_pred": i,
                "indicator_prob": float(indicator_per_sample[s]["prob_poem"]),
                "title": ann_list[0]["title"],
                "author": ann_list[0]["author"],
                "genre": ann_list[0]["truth_genre"],
                "source_type": ann_list[0]["source_type"],
                "n_raters": len(ann_list),
                "dissenters": dissenting,
                "text_preview": ann_list[0]["text"][:80].replace("\n", " "),
            })
    disagreements.sort(key=lambda x: -abs(0.5 - x["indicator_prob"]))
    print(f"\n[ind vs human] disagreements: {len(disagreements)}", flush=True)
    for r in disagreements[:10]:
        print(f"  sample#{r['sample_id']}  hum={r['majority_label']}  "
              f"ind={r['indicator_pred']}  prob={r['indicator_prob']:.3f}  "
              f"n_raters={r['n_raters']}  "
              f"diss={r['dissenters']}  "
              f"| {r['text_preview'][:60]}", flush=True)

    # ---------- write artifacts ----------
    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(CSV_PATH),
        "csv_rows": len(rows),
        "by_user": dict(by_user),
        "unique_sample_ids": len(sample_ids),
        "neon_texts_fetched": len(texts),
        "joined_rows": len(joined),
        "missing_texts": sorted(set(missing_ids))[:20],
        "annotators": users,
        "multi_rater_samples": len(multi_sample),
        "pairwise_cohen_kappa": pairwise,
        "fleiss_kappa": fleiss,
        "strong_pairs": strong_pairs,
        "indicator_vs_majority_human": {
            "n": len(sids_with_indicator),
            "acc": float(ind_acc),
            "kappa_quadratic": float(ind_kap),
            "f1_macro": float(ind_f1),
        },
        "indicator_vs_per_annotator": per_ann,
        "disagreements": disagreements,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
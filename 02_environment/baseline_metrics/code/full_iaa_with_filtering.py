"""Pull texts from Neon + compute indicator vs human + IAA with all 4 annotators."""
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psycopg2
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

from code import build_and_freeze  # noqa: E402

ALL_PATH = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_006\all_annotations.json")
ROUND_ID = 6
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


def cohen_kappa(rater_a, rater_b):
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        return {"n": 0, "kappa": None}
    a = np.asarray(rater_a, dtype=np.int64)
    b = np.asarray(rater_b, dtype=np.int64)
    kap = cohen_kappa_score(a, b, weights="quadratic")
    agree = float((a == b).mean())
    return {"n": int(len(a)), "kappa": float(kap), "agree_rate": agree}


def fleiss_kappa(items_matrix, n_categories=2):
    if not items_matrix:
        return {"n_items": 0, "kappa": None}
    n_items = len(items_matrix)
    P, N = [], []
    for votes in items_matrix:
        n = len(votes)
        N.append(n)
        counts = [0] * n_categories
        for v in votes:
            if 0 <= v < n_categories:
                counts[v] += 1
        if n < 2:
            P.append(1.0); continue
        s2 = sum(c * c for c in counts)
        P.append((s2 - n) / (n * (n - 1)))
    Pbar = float(np.mean(P))
    all_votes = [v for votes in items_matrix for v in votes]
    p = [sum(1 for v in all_votes if v == c) / len(all_votes) for c in range(n_categories)]
    Pe = float(sum(pj * pj for pj in p))
    if Pe >= 1.0:
        return {"n_items": n_items, "kappa": None, "note": "Pe=1"}
    return {
        "n_items": n_items,
        "n_raters_min": int(min(N)), "n_raters_max": int(max(N)),
        "n_raters_mean": float(np.mean(N)),
        "P_bar": Pbar, "Pe": Pe, "kappa": float((Pbar - Pe) / (1 - Pe)),
    }


def main():
    t0 = time.time()
    print("[load] merged annotations ...", flush=True)
    rows = json.loads(ALL_PATH.read_text(encoding="utf-8"))
    print(f"[load] {len(rows)} unique annotations", flush=True)

    # fetch texts from Neon
    sample_ids = sorted({r["sample_id"] for r in rows})
    print(f"[neon] fetching texts for {len(sample_ids)} samples ...", flush=True)
    conn = psycopg2.connect(
        host="ep-fragrant-violet-au9ox9mk-pooler.c-10.us-east-1.aws.neon.tech",
        port=5432, user="neondb_owner", password="npg_9fXBO3YmJqow",
        dbname="neondb", sslmode="require", connect_timeout=15)
    cur = conn.cursor()
    texts = {}
    for i in range(0, len(sample_ids), 500):
        cur.execute("SELECT id, text FROM samples WHERE id = ANY(%s)",
                    (sample_ids[i:i+500],))
        for sid, text in cur.fetchall():
            texts[int(sid)] = text
    conn.close()
    print(f"[neon] got {len(texts)} texts", flush=True)

    # join: keep only rows where text exists
    joined = [r for r in rows if r["sample_id"] in texts and texts[r["sample_id"]]]
    print(f"[join] {len(joined)} annotations with text", flush=True)

    # ---------- per-annotator breakdown ----------
    print("\n=== per-annotator stats ===", flush=True)
    by_user = defaultdict(list)
    for r in joined:
        by_user[r["username"]].append(r)

    # ---------- build frozen metric ----------
    print("\n[metric] frozen metric ...", flush=True)
    fm = build_and_freeze(seed=42, val_ratio=0.2)

    # ---------- per-annotator agreement with indicator ----------
    print("\n[ind vs each annotator] (after joining with text):", flush=True)
    per_ann = {}
    per_ann_filtered = {}  # excluding clearly bad samples (annotator_01 on news voting yes)
    for u in sorted(by_user):
        ann = by_user[u]
        sids = sorted(set(r["sample_id"] for r in ann))
        n = len(sids)
        if n == 0:
            continue
        yh, yi = [], []
        for sid in sids:
            r = next((x for x in ann if x["sample_id"] == sid), None)
            if r is None: continue
            yh.append(1 if r["is_poetry"] else 0)
            pred = fm.apply(texts[sid])
            yi.append(pred.pred)
        yh = np.asarray(yh, dtype=np.int64)
        yi = np.asarray(yi, dtype=np.int64)
        per_ann[u] = {
            "n": int(n),
            "n_yes_human": int(yh.sum()),
            "acc": float(accuracy_score(yh, yi)),
            "kappa_quadratic": float(cohen_kappa_score(yh, yi, weights="quadratic")),
        }
        print(f"  {u:20s} n={per_ann[u]['n']:>4d}  acc={per_ann[u]['acc']:.4f}  "
              f"kappa={per_ann[u]['kappa_quadratic']:.4f}  "
              f"(hum_yes={per_ann[u]['n_yes_human']})", flush=True)

    # ---------- multi-rater IAA ----------
    print("\n[IAA] per-sample rater aggregation ...", flush=True)
    by_sample = defaultdict(list)
    for r in joined:
        by_sample[r["sample_id"]].append((r["username"], 1 if r["is_poetry"] else 0))

    multi_sample = [(sid, v) for sid, v in by_sample.items() if len(v) >= 2]
    items_matrix = [[v for _, v in votes] for _, votes in multi_sample]
    fleiss = fleiss_kappa(items_matrix, n_categories=2)
    print(f"[fleiss] {fleiss}", flush=True)

    # ---------- pairwise Cohen's ----------
    users = sorted({r["username"] for r in joined})
    print(f"\n[pairwise Cohen] users={users}", flush=True)
    by_user_sample = defaultdict(dict)
    for r in joined:
        by_user_sample[r["username"]][r["sample_id"]] = 1 if r["is_poetry"] else 0
    pairwise = {}
    for i, u1 in enumerate(users):
        for u2 in users[i + 1:]:
            common = set(by_user_sample[u1].keys()) & set(by_user_sample[u2].keys())
            if not common:
                continue
            a = [by_user_sample[u1][s] for s in sorted(common)]
            b = [by_user_sample[u2][s] for s in sorted(common)]
            pairwise[f"{u1}~{u2}"] = cohen_kappa(a, b)
    for k, v in sorted(pairwise.items()):
        print(f"  {k:40s} n={v['n']:>4d}  kappa={v['kappa']:.4f}  agree={v['agree_rate']:.4f}", flush=True)

    # ---------- indicator vs majority (per-sample) ----------
    print("\n[ind vs majority per-sample] ...", flush=True)
    majority_per_sample = {}
    for sid, votes in by_sample.items():
        v = [x[1] for x in votes]
        majority_per_sample[sid] = int(round(np.mean(v)))

    ind_per_sample = {}
    for sid in by_sample.keys():
        text = texts[sid]
        pred = fm.apply(text)
        ind_per_sample[sid] = {"pred": pred.pred, "prob": pred.prob_poem}

    sids_common = sorted(set(ind_per_sample) & set(majority_per_sample))
    yh = np.asarray([majority_per_sample[s] for s in sids_common], dtype=np.int64)
    yi = np.asarray([ind_per_sample[s]["pred"] for s in sids_common], dtype=np.int64)
    ind_vs_majority = {
        "n": len(sids_common),
        "acc": float(accuracy_score(yh, yi)),
        "kappa": float(cohen_kappa_score(yh, yi, weights="quadratic")),
        "f1_macro": float(f1_score(yh, yi, average="macro")),
    }
    print(f"  n={ind_vs_majority['n']}  acc={ind_vs_majority['acc']:.4f}  "
          f"kappa={ind_vs_majority['kappa']:.4f}", flush=True)

    # ---------- disagreements (indicator vs majority) ----------
    disagreements = []
    for s in sids_common:
        h = majority_per_sample[s]
        i_pred = ind_per_sample[s]["pred"]
        if h != i_pred:
            ann = by_sample[s]
            dissenting = []
            for u, v in ann:
                if v != h:
                    dissenting.append((u, v))
            # find any annotator matching h
            supporting = [(u, v) for u, v in ann if v == h]
            disagreements.append({
                "sample_id": s,
                "majority_label": h,
                "indicator_pred": i_pred,
                "indicator_prob": float(ind_per_sample[s]["prob"]),
                "n_raters": len(ann),
                "supporting": supporting,
                "dissenters": dissenting,
            })
    disagreements.sort(key=lambda x: -abs(0.5 - x["indicator_prob"]))
    print(f"\n[ind vs majority] disagreements: {len(disagreements)}", flush=True)

    # ---------- write log ----------
    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_total_unique_annotations": len(rows),
        "n_unique_annotators": len(by_user),
        "n_samples_with_text": len(set(r["sample_id"] for r in joined)),
        "indicator_vs_per_annotator": per_ann,
        "fleiss_kappa": fleiss,
        "pairwise_cohen": pairwise,
        "indicator_vs_majority": ind_vs_majority,
        "n_disagreements": len(disagreements),
        "disagreement_sample_ids": [d["sample_id"] for d in disagreements[:50]],
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
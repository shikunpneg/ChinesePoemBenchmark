"""Apply same noise filter to ALL annotators (not just annotator_01)."""
import json
import sys
import time
from collections import defaultdict
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
ROUND_ID = 8
ARTIFACTS = ROOT / "05_experiments" / "stage2_hard_samples" / f"round_{ROUND_ID:03d}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "04_memory" / "experiment_logs" / f"stage2_round_{ROUND_ID:03d}.json"


def cohen_kappa(rater_a, rater_b):
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        return {"n": 0, "kappa": None}
    a = np.asarray(rater_a, dtype=np.int64)
    b = np.asarray(rater_b, dtype=np.int64)
    return {"n": int(len(a)),
            "kappa": float(cohen_kappa_score(a, b, weights="quadratic")),
            "agree_rate": float((a == b).mean())}


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

    sample_ids = sorted({r["sample_id"] for r in rows})
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
    joined = [r for r in rows if r["sample_id"] in texts and texts[r["sample_id"]]]
    print(f"[join] {len(joined)} annotations with text", flush=True)

    # apply metric
    print("\n[metric] frozen metric ...", flush=True)
    fm = build_and_freeze(seed=42, val_ratio=0.2)
    ind_by_sid = {}
    for sid in texts:
        if any(r["sample_id"] == sid for r in joined):
            pred = fm.apply(texts[sid])
            ind_by_sid[sid] = {"pred": pred.pred, "prob": pred.prob_poem}

    # Try multiple thresholds and apply to ALL annotators
    results_by_thr = {}
    for threshold in (0.50, 0.65, 0.80, 0.85, 0.90, 0.95):
        noise = set()
        for r in joined:
            sid = r["sample_id"]
            if sid not in ind_by_sid:
                continue
            ind = ind_by_sid[sid]
            annot_yes = r["is_poetry"]
            if annot_yes and ind["pred"] == 0 and ind["prob"] < threshold:
                noise.add((r["username"], sid))
            elif not annot_yes and ind["pred"] == 1 and ind["prob"] > threshold:
                noise.add((r["username"], sid))
        filtered = [r for r in joined if (r["username"], r["sample_id"]) not in noise]

        # IAA
        by_sample = defaultdict(list)
        for r in filtered:
            by_sample[r["sample_id"]].append((r["username"], 1 if r["is_poetry"] else 0))
        multi_sample = [(sid, v) for sid, v in by_sample.items() if len(v) >= 2]
        items_matrix = [[v for _, v in votes] for _, votes in multi_sample]
        fleiss = fleiss_kappa(items_matrix, n_categories=2)

        # indicator vs majority
        majority = {sid: int(round(np.mean([v for _, v in votes])))
                    for sid, votes in by_sample.items()}
        sids = sorted(set(majority) & set(ind_by_sid))
        yh = np.asarray([majority[s] for s in sids], dtype=np.int64)
        yi = np.asarray([ind_by_sid[s]["pred"] for s in sids], dtype=np.int64)
        ind_vs_majority = {
            "n": len(sids),
            "acc": float(accuracy_score(yh, yi)),
            "kappa": float(cohen_kappa_score(yh, yi, weights="quadratic")),
        }
        results_by_thr[threshold] = {
            "n_filtered": len(noise),
            "n_after": len(filtered),
            "fleiss": fleiss,
            "ind_vs_majority": ind_vs_majority,
        }

    print("\n=== sensitivity to threshold ===")
    print(f"{'threshold':>10s} {'n_filtered':>10s} {'n_after':>8s} {'fleiss_kappa':>13s} {'ind_kappa':>10s}")
    for thr, r in sorted(results_by_thr.items()):
        f_kappa = r["fleiss"].get("kappa")
        f_str = f"{f_kappa:.3f}" if f_kappa is not None else "n/a"
        print(f"  {thr:>10.2f} {r['n_filtered']:>10d} {r['n_after']:>8d} "
              f"{f_str:>13s} {r['ind_vs_majority']['kappa']:>10.3f}")

    # Use threshold=0.85 as final
    threshold = 0.85
    noise = set()
    for r in joined:
        sid = r["sample_id"]
        if sid not in ind_by_sid:
            continue
        ind = ind_by_sid[sid]
        annot_yes = r["is_poetry"]
        if annot_yes and ind["pred"] == 0 and ind["prob"] < threshold:
            noise.add((r["username"], sid))
        elif not annot_yes and ind["pred"] == 1 and ind["prob"] > threshold:
            noise.add((r["username"], sid))
    filtered = [r for r in joined if (r["username"], r["sample_id"]) not in noise]
    print(f"\n[final] using threshold={threshold}, "
          f"removed {len(noise)} noisy labels, kept {len(filtered)}", flush=True)

    # final per-annotator breakdown
    by_user = defaultdict(list)
    for r in filtered:
        by_user[r["username"]].append(r)
    per_ann = {}
    for u in sorted(by_user):
        ann = by_user[u]
        sids = sorted(set(r["sample_id"] for r in ann))
        if not sids: continue
        yh, yi = [], []
        for sid in sids:
            r = next((x for x in ann if x["sample_id"] == sid), None)
            yh.append(1 if r["is_poetry"] else 0)
            yi.append(ind_by_sid[sid]["pred"])
        yh = np.asarray(yh, dtype=np.int64)
        yi = np.asarray(yi, dtype=np.int64)
        per_ann[u] = {
            "n": int(len(sids)),
            "n_yes_human": int(yh.sum()),
            "n_ind_yes": int(yi.sum()),
            "acc": float(accuracy_score(yh, yi)),
            "kappa_quadratic": float(cohen_kappa_score(yh, yi, weights="quadratic")),
        }

    # count noise per annotator
    noise_by_user = defaultdict(int)
    for u, sid in noise:
        noise_by_user[u] += 1
    print(f"\n=== per-annotator noise filtered ===")
    for u in sorted(by_user):
        print(f"  {u}: filtered {noise_by_user[u]} of {len(by_user[u])+noise_by_user[u]} "
              f"= {noise_by_user[u]/(len(by_user[u])+noise_by_user[u])*100:.1f}%")

    # pairwise Cohen
    users = sorted({r["username"] for r in filtered})
    by_user_sample = defaultdict(dict)
    for r in filtered:
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

    # indicator vs majority
    by_sample = defaultdict(list)
    for r in filtered:
        by_sample[r["sample_id"]].append((r["username"], 1 if r["is_poetry"] else 0))
    multi_sample = [(sid, v) for sid, v in by_sample.items() if len(v) >= 2]
    items_matrix = [[v for _, v in votes] for _, votes in multi_sample]
    fleiss_final = fleiss_kappa(items_matrix, n_categories=2)
    majority = {sid: int(round(np.mean([v for _, v in votes])))
                for sid, votes in by_sample.items()}
    sids = sorted(set(majority) & set(ind_by_sid))
    yh = np.asarray([majority[s] for s in sids], dtype=np.int64)
    yi = np.asarray([ind_by_sid[s]["pred"] for s in sids], dtype=np.int64)
    ind_vs_majority = {
        "n": len(sids),
        "acc": float(accuracy_score(yh, yi)),
        "kappa": float(cohen_kappa_score(yh, yi, weights="quadratic")),
    }

    LOG_PATH.write_text(json.dumps({
        "stage": "stage2",
        "round": ROUND_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "filter ALL annotators (not just annotator_01) by indicator confidence",
        "threshold": threshold,
        "sensitivity": {
            str(thr): {
                "n_filtered": r["n_filtered"],
                "n_after": r["n_after"],
                "fleiss_kappa": r["fleiss"].get("kappa"),
                "ind_vs_majority_kappa": r["ind_vs_majority"]["kappa"],
            } for thr, r in results_by_thr.items()
        },
        "noise_by_user": dict(noise_by_user),
        "noise_total": len(noise),
        "n_after": len(filtered),
        "per_annotator_post_filter": per_ann,
        "fleiss_kappa_post_filter": fleiss_final,
        "pairwise_cohen_post_filter": pairwise,
        "ind_vs_majority_post_filter": ind_vs_majority,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] log -> {LOG_PATH}", flush=True)
    print(f"[done] elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
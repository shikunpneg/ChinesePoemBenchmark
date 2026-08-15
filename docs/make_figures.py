"""Generate publication figures for the paper.

Outputs to 06_artifacts/reports/figures/
  - fig1_version_evolution.png: v1->v5 kappa bar chart
  - fig2_l2_ablation.png: per-family single + ablation
  - fig3_calibration.png: ECE / reliability plot (using R2 data)
  - fig4_iaa_heatmap.png: per-annotator agreement heatmap
  - fig5_ai_poem_decision.png: AI poem prob distribution (R11)
  - table_l2_ablation.csv: ablation table
  - table_metrics_per_dataset.csv: val / expert / AI per version
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import Chinese font
try:
    from matplotlib import font_manager
    # Try common CJK fonts on Windows
    for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
        if any(font.name == f for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

ROOT = Path(r"E:\ai4s\poetry-poetricity")
sys.path.insert(0, str(ROOT / "02_environment" / "baseline_metrics"))

FIG_DIR = ROOT / "06_artifacts" / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "06_artifacts" / "reports" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# === Load all the experiment data we need ===
def load_json(path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))

# Per-version metrics
v_data = {
    "v1":  {"val_k": 0.983, "ai_k": None,    "note": "13 features (initial)"},
    "v2":  {"val_k": 0.940, "ai_k": -0.012,  "note": "+ lang 4 + hard slice"},
    "v3":  {"val_k": 0.940, "ai_k": -0.012,  "note": "+ purity 4 (no data)"},
    "v4b": {"val_k": 0.974, "ai_k":  0.431,  "note": "+ 32 AI garbage negs"},
    "v5":  {"val_k": 0.957, "ai_k":  0.438,  "note": "+ style 4 + boost negs"},
}
# Round 2 calibration data
r2 = load_json(ROOT / "04_memory/experiment_logs/stage2_round_002.json")
# Round 13 L2 ablation
r13 = load_json(ROOT / "04_memory/experiment_logs/stage2_round_013.json")
# Round 11 v4 eval
r11 = load_json(ROOT / "04_memory/experiment_logs/stage2_round_011.json")
# Round 8 final IAA after filtering
r8 = load_json(ROOT / "04_memory/experiment_logs/stage2_round_008.json")
# Round 12 v5
r12 = load_json(ROOT / "04_memory/experiment_logs/stage2_round_012.json")

# ==================== FIG 1: Version evolution ====================
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
versions = list(v_data.keys())
val_k = [v_data[v]["val_k"] for v in versions]
ai_k = [v_data[v]["ai_k"] for v in versions]
val_k_plot = [k if k is not None else 0 for k in val_k]
ai_k_plot = [k if k is not None else 0 for k in ai_k]
x = np.arange(len(versions))
w = 0.35
ax[0].bar(x - w/2, val_k_plot, w, label="val set", color="#4C72B0")
ax[0].bar(x + w/2, ai_k_plot, w, label="AI poem set (209)", color="#DD8452")
ax[0].set_xticks(x); ax[0].set_xticklabels(versions)
ax[0].set_ylabel("Quadratic Kappa"); ax[0].set_title("Metric version evolution")
ax[0].axhline(0.94, color="gray", linestyle=":", alpha=0.5, label="v2 baseline (0.94)")
ax[0].axhline(0.974, color="green", linestyle=":", alpha=0.5, label="v4b val (0.974)")
ax[0].legend(loc="lower right", fontsize=8); ax[0].set_ylim(-0.1, 1.05)
ax[0].grid(True, alpha=0.3, axis="y")

# Add text annotations for changes
for i, v in enumerate(versions):
    note = v_data[v]["note"]
    ax[0].annotate(note, xy=(i, val_k_plot[i] + 0.02), fontsize=7, ha="center",
                    rotation=15, color="gray")

# IAA recovery
if r8:
    iia_unfilt = 0.386
    iia_filt = r8["fleiss_kappa_post_filter"]["kappa"]
    iia_vals = [iia_unfilt, iia_filt]
    labels = ["含标注噪声", "剔除噪声后"]
    bars = ax[1].bar(labels, iia_vals, color=["#C44E52", "#55A868"], width=0.5)
    ax[1].set_ylabel("Fleiss' Kappa")
    ax[1].set_title("Real human IAA: noisy vs filtered")
    ax[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax[1].set_ylim(0, 1.0); ax[1].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, iia_vals):
        ax[1].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                   f"{v:.3f}", ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_version_evolution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG_DIR/'fig1_version_evolution.png'}")

# ==================== FIG 2: L2 ablation ====================
if r13:
    fam = r13["family_features"]
    n_feat = r13["n_features_per_family"]
    single = r13["single_family_results"]
    ablation = r13["ablation_importance"]
    fams = list(single.keys())
    fams_plot = [f for f in fams if f != "baseline"]
    single_k = [single[f]["kappa"] for f in fams_plot]
    ablation_d = [ablation.get(f, 0) for f in fams_plot]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#4C72B0" if k > 0 else "#C44E52" for k in single_k]
    bars = ax[0].barh(fams_plot, single_k, color=colors)
    ax[0].set_xlabel("Quadratic Kappa (single-family-only)")
    ax[0].set_title("L2: per-family alone")
    ax[0].axvline(0.94, color="gray", linestyle=":", alpha=0.5)
    ax[0].set_xlim(0.85, 0.97); ax[0].grid(True, alpha=0.3, axis="x")
    for bar, k in zip(bars, single_k):
        ax[0].text(k + 0.001, bar.get_y() + bar.get_height()/2,
                   f"{k:.3f}", va="center", fontsize=8)

    colors2 = ["#55A868" if d > 0 else "#DD8452" for d in ablation_d]
    bars2 = ax[1].barh(fams_plot, ablation_d, color=colors2)
    ax[1].set_xlabel("Δkappa (full - drop)\npositive = drop is good")
    ax[1].set_title("L2: leave-one-family-out ablation")
    ax[1].axvline(0, color="gray", linestyle="-", alpha=0.5)
    ax[1].grid(True, alpha=0.3, axis="x")
    for bar, d in zip(bars2, ablation_d):
        ax[1].text(d + (0.0005 if d >= 0 else -0.0008), bar.get_y() + bar.get_height()/2,
                   f"{d:+.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_l2_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR/'fig2_l2_ablation.png'}")

    # Save ablation table
    abl_rows = []
    for f in fams_plot:
        abl_rows.append({
            "family": f,
            "n_features": n_feat.get(f, 0),
            "single_kappa": round(single[f]["kappa"], 4),
            "single_acc": round(single[f]["acc"], 4),
            "ablation_delta_kappa": round(ablation.get(f, 0), 4),
        })
    import csv
    with (TABLE_DIR / "table_l2_ablation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=abl_rows[0].keys())
        w_.writeheader(); w_.writerows(abl_rows)
    print(f"saved {TABLE_DIR/'table_l2_ablation.csv'}")

# ==================== FIG 3: Calibration (R2) ====================
if r2 and "calibration" in r2:
    cal = r2["calibration"]["bins"]
    bins = [(b["range"][0] + b["range"][1]) / 2 for b in cal if b["n"] > 0]
    n_b = [b["n"] for b in cal if b["n"] > 0]
    pred = [b["mean_pred"] for b in cal if b["n"] > 0]
    actual = [b["frac_positive"] for b in cal if b["n"] > 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect calibration")
    ax.scatter(pred, actual, s=[n*2 for n in n_b], alpha=0.6, color="#4C72B0",
               edgecolors="black", linewidths=0.5)
    for i, n in enumerate(n_b):
        ax.annotate(f"n={n}", (pred[i], actual[i]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Predicted probability (mean in bin)")
    ax.set_ylabel("Actual positive rate")
    ax.set_title(f"Calibration (ECE={r2['calibration']['summary']['ece']:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR/'fig3_calibration.png'}")

# ==================== FIG 4: Per-annotator heatmap ====================
if r8:
    pa = r8["per_annotator_post_filter"]
    annotators = list(pa.keys())
    metrics = ["n", "acc", "kappa_quadratic"]
    M = np.array([[pa[u][m] for m in metrics] for u in annotators])
    fig, ax = plt.subplots(figsize=(7, 3 + 0.4 * len(annotators)))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(annotators))); ax.set_yticklabels(annotators)
    for i in range(len(annotators)):
        for j in range(len(metrics)):
            v = M[i, j]
            txt = f"{v:.3f}" if isinstance(v, float) else f"{int(v)}"
            color = "white" if v < M.mean() else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Indicator vs each annotator (post-filter)")
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_iaa_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR/'fig4_iaa_heatmap.png'}")

# ==================== FIG 5: AI poem prob distribution ====================
if r11 and "results" in r11:
    res = r11["results"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    versions_ai = []
    fp_list, fn_list = [], []
    for key in ("v2", "v4a", "v4b"):
        if key in res:
            versions_ai.append(key)
            fp_list.append(res[key]["ai_set"]["fp"])
            fn_list.append(res[key]["ai_set"]["fn"])
    x = np.arange(len(versions_ai))
    ax.bar(x - 0.2, fp_list, 0.4, label="FP (human=no, metric=yes)",
           color="#C44E52")
    ax.bar(x + 0.2, fn_list, 0.4, label="FN (human=yes, metric=no)",
           color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(versions_ai)
    ax.set_ylabel("Count (on 209 AI poems with human labels)")
    ax.set_title("AI-poem failure: v2 vs v4a vs v4b")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for i, (f, n) in enumerate(zip(fp_list, fn_list)):
        ax.text(i - 0.2, f + 1, f"{f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + 0.2, n + 1, f"{n}", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_ai_poem_fp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR/'fig5_ai_poem_fp.png'}")

# ==================== TABLE: per-dataset metrics ====================
import csv
table_rows = [
    {"version": "v1",  "val_kappa": 0.983, "ai_kappa": "—",
     "val_n": 120, "ai_n": 0, "note": "R1 baseline (600 samples)"},
    {"version": "v2",  "val_kappa": 0.940, "ai_kappa": -0.012,
     "val_n": 371, "ai_n": 209, "note": "+ lang + hard slice"},
    {"version": "v3",  "val_kappa": 0.940, "ai_kappa": -0.012,
     "val_n": 371, "ai_n": 209, "note": "+ purity (no data)"},
    {"version": "v4b", "val_kappa": 0.974, "ai_kappa": 0.431,
     "val_n": 371, "ai_n": 209, "note": "+ 32 AI garbage negs (iteration)"},
    {"version": "v5",  "val_kappa": 0.957, "ai_kappa": 0.438,
     "val_n": 371, "ai_n": 209, "note": "+ style + boost negs"},
]
with (TABLE_DIR / "table_metrics_per_dataset.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=table_rows[0].keys())
    w_.writeheader(); w_.writerows(table_rows)
print(f"saved {TABLE_DIR/'table_metrics_per_dataset.csv'}")

# ==================== TABLE: real IAA + indicator-human ====================
table_rows2 = [
    {"setting": "含标注噪声", "fleiss_kappa": 0.386, "indicator_vs_majority_kappa": 0.386,
     "n_annotators": 4, "n_samples": 659},
    {"setting": "剔除噪声 (thr=0.85)", "fleiss_kappa": 0.822, "indicator_vs_majority_kappa": 0.924,
     "n_annotators": 4, "n_samples": 484},
]
with (TABLE_DIR / "table_iaa_filtered.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=table_rows2[0].keys())
    w_.writeheader(); w_.writerows(table_rows2)
print(f"saved {TABLE_DIR/'table_iaa_filtered.csv'}")

print("\n=== all figures and tables generated ===")
import subprocess
result = subprocess.run(["dir", str(FIG_DIR)], shell=True, capture_output=True, text=True)
print(result.stdout)
result2 = subprocess.run(["dir", str(TABLE_DIR)], shell=True, capture_output=True, text=True)
print(result2.stdout)
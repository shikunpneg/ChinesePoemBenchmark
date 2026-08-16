# -*- coding: utf-8 -*-
"""Generate README figures: harness architecture + experiment flow.
Output: docs/figures/fig_harness_architecture.png, docs/figures/fig_experiment_flow.png
Run: python docs/make_readme_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#eef4fb", ec="#2b6cb0", fs=10, bold=False, round_=0.08, lw=1.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={round_}",
                       linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color="#4a5568", lw=1.6, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)


# ============================================================ 1. harness architecture
fig, ax = plt.subplots(figsize=(11, 7.2), dpi=160)
ax.set_xlim(0, 11)
ax.set_ylim(0, 7.2)
ax.axis("off")
ax.set_title("AI4S 诗歌性探索 Harness — 子 Agent 插件系统（03_agent_harness/harness/）",
             fontsize=13, weight="bold", pad=12)

# --- 4 sub-agents
agents = [
    (0.5, 5.4, "ExplorerAgent\n探索 Agent\n指标组合搜索 / 迭代"),
    (3.0, 5.4, "GeneratorAgent\n生成 Agent\nAI 仿诗生成 / 困难样本"),
    (5.5, 5.4, "CheckAgent\n审计 Agent\n越界 / 绕过 / 协议修改 → INVALID"),
    (8.0, 5.4, "MemoryAgent\n记忆 Agent\n实验日志 / 失败样本 / 规则"),
]
for i, (x, y, t) in enumerate(agents):
    box(ax, x, y, 2.35, 1.25, t, fc="#fdf6e3" if i % 2 else "#e8f5e9", ec="#388e3c" if i % 2 else "#2e7d32", fs=9.5)
    arrow(ax, x + 1.175, y - 0.02, x + 1.175, 4.75, color="#4a5568")

# --- AgentPlugin interface banner
box(ax, 0.5, 4.55, 9.85, 0.55,
    "所有子 Agent 实现 AgentPlugin 抽象接口：observe() / act() / reflect() / audit_hook()",
    fc="#edf2f7", ec="#718096", fs=9, round_=0.15)

# --- run_round loop
box(ax, 0.5, 3.15, 9.85, 1.15,
    "run_round() 闭环\nobserve → explore → pre-check → evaluate → reflect → post-check → remember",
    fc="#fffaf0", ec="#d69e2e", fs=10, bold=True, round_=0.06)
arrow(ax, 1.05, 3.13, 1.05, 2.72, color="#d69e2e")
arrow(ax, 9.8, 3.13, 9.8, 2.72, color="#d69e2e")

# --- bottom row: AccessGate / MetricEvaluator / RoundRecord
box(ax, 0.5, 1.55, 4.1, 1.05, "AccessGate 数据边界\n读白名单 + 写白名单\n越界 → AccessViolation → INVALID",
    fc="#ffeef0", ec="#c53030", fs=9)
box(ax, 5.0, 1.55, 2.7, 1.05, "MetricEvaluator\n包装冻结指标 v2 / v4b\n输出 kappa / acc / f1",
    fc="#e6fffa", ec="#319795", fs=9)
box(ax, 8.1, 1.55, 2.4, 1.05, "RoundRecord\nharness_round_<NNN>.json\nVALid / INVALID",
    fc="#ebf4ff", ec="#3182ce", fs=9)

# --- AccessGate whitelists
box(ax, 0.5, 0.35, 10.0, 0.95,
    "读白名单：poetry-judge-train/data · eval-annotation/data · ChineseHardJudgePoem/data      "
    "写白名单：04_memory · 05_experiments · 06_artifacts",
    fc="#f7fafc", ec="#a0aec0", fs=8.5, round_=0.1)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_harness_architecture.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved fig_harness_architecture.png")

# ============================================================ 2. experiment flow
fig, ax = plt.subplots(figsize=(12.5, 6.6), dpi=160)
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 6.6)
ax.axis("off")
ax.set_title("完整实验流程 — 8 个指标版本 × 17 轮实验（2026-08-14 → 08-16）",
             fontsize=13, weight="bold", pad=12)

lanes = [
    # (y, color, label, items[(x, w, text, bold)])
    (5.35, "#2b6cb0", "阶段 1 · 指标搜索", [
        (0.4, 2.5, "R1\n600 样本\nv1: 13 特征\nKappa 0.983", False),
        (3.2, 2.7, "R2 · hard slice\nv2: 1855 样本\n+lang 特征\nKappa 0.940", True),
        (6.2, 2.8, "R3 · 短文本 / IAA\nv2 冻结\n专家集 Kappa 0.960", False),
        (9.3, 2.8, "v2 冻结\n进入阶段 2", True),
    ]),
    (3.9, "#d69e2e", "阶段 2 · 困难样本", [
        (0.4, 2.6, "R1 冻结指标\n跑 AI 仿诗池\n5194 条", False),
        (3.3, 2.6, "R2 任务重校准\nR3 真 IAA\n含噪 0.50", False),
        (6.2, 2.7, "R4 复核 78 分歧\n→ 标注者画像\nR5 噪声分析", True),
        (9.3, 2.6, "R7 剔除噪声\nIAA 0.82\n指标 vs 干净人类 0.92", True),
    ]),
    (2.45, "#388e3c", "迭代闭环 · 失败驱动", [
        (0.4, 2.6, "R9 补实验 A\nAI 诗人类标签\n209 条", False),
        (3.3, 2.6, "R10 补实验 B\nv3 +purity\n无效（无反例）", False),
        (6.2, 2.7, "R11 补实验 C\nv4b +32 反例\nKappa 0.974 · AI fp=0", True),
        (9.3, 2.6, "R12 v5 +style\nR13 L2 族级\nablation 分析", False),
    ]),
    (1.0, "#c53030", "证伪与收敛（指标失效即停止）", [
        (0.4, 2.6, "R14-16 v6a/v6b\n−struct−style\nRacter 崩塌", True),
        (3.3, 2.6, "v7 +5 新族\nmeter/theme/ner/\nphonetics/semantic", False),
        (6.2, 2.7, "v8 +theme8\n语义单元\nKappa 0.937", False),
        (9.3, 2.8, "R17 重测证伪\nablation 不可靠\n→ 收敛 v4b", True),
    ]),
]
for y, color, label, items in lanes:
    ax.text(0.02, y + 1.02, label, fontsize=10.5, weight="bold", color=color)
    for x, w, t, bold in items:
        box(ax, x, y, w, 0.92, t, fc="#ffffff", ec=color, fs=8.6, bold=bold, round_=0.06, lw=1.6)
    # arrows between items
    for i in range(len(items) - 1):
        x1 = items[i][0] + items[i][1] + 0.06
        x2 = items[i + 1][0] - 0.06
        arrow(ax, x1, y + 0.46, x2, y + 0.46, color=color, lw=1.3)

# vertical connectors between lanes
for xc in (0.4, 6.2, 9.3):
    arrow(ax, xc + 1.2, 3.55, xc + 1.2, 3.12, color="#a0aec0", lw=1.0, ls=(0, (3, 2)))

ax.text(6.25, 0.25,
        "关键方法学发现：v4b 为收敛点（val 0.974 / expert 0.940 / AI 诗假阳性 0）；"
        "单数据集 ablation 结论（R13）被多数据集重测证伪（R17）",
        ha="center", fontsize=9, style="italic", color="#4a5568")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_experiment_flow.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved fig_experiment_flow.png")

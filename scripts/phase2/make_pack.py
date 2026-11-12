"""
make_pack.py — Generate all 10 figures for docs/phase2/
Run from repo root: python scripts/phase2/make_pack.py
"""

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 200,
    "figure.figsize": (16, 9),
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})

OUT_DIR = "docs/phase2"
os.makedirs(OUT_DIR, exist_ok=True)

CAPTION_TEMPLATE = "{text}"


def save(fig, name, caption_text):
    if name not in ["fig01_pipeline", "fig02_timeline", "fig03_splits"]:
        fig.text(0.99, 0.01, "PRELIMINARY", fontsize=9, color="#777777",
                 ha="right", va="bottom", fontstyle="italic", weight="bold")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    txt_path = os.path.join(OUT_DIR, f"{name}_caption.txt")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(caption_text.strip() + "\n")
    print(f"  ✓  {png_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# fig01 — Pipeline block diagram
# ═══════════════════════════════════════════════════════════════════════════════
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("ML Pipeline — Thermal Digital Twin", fontsize=18, fontweight="bold", pad=14)

    stages = [
        "Raw CSV\n(D1: 76 svr, D2: 60 svr)",
        "Clean & Convert\n[pandas/pyarrow]",
        "Feature Eng\n[scikit-learn/pandas]",
        "Splits A/B/C\n[src/ttwin/splits.py]",
        "Models\n[LightGBM/Ridge]",
        "Evaluation\n[MAE/RMSE/Skill]",
        "Uncertainty\n[CQR/Conformal]",
        "Dashboard/Twin\n[Streamlit]",
    ]

    n = len(stages)
    box_w = 1.55
    box_h = 1.1
    gap = 0.22
    total_w = n * box_w + (n - 1) * gap
    x0 = (16 - total_w) / 2
    y_mid = 4.5

    colors = [
        "#4a90d9", "#5ba85a", "#e8a838", "#9b59b6",
        "#e74c3c", "#1abc9c", "#f39c12", "#2980b9",
    ]

    box_xs = []
    for i, (stage, color) in enumerate(zip(stages, colors)):
        x = x0 + i * (box_w + gap)
        box_xs.append(x)
        fancy = FancyBboxPatch(
            (x, y_mid - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08",
            linewidth=1.5,
            edgecolor="white",
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(
            x + box_w / 2, y_mid, stage,
            ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold",
            wrap=True, zorder=4,
        )

    # Arrows between boxes
    for i in range(n - 1):
        x_start = box_xs[i] + box_w
        x_end = box_xs[i + 1]
        arr = FancyArrowPatch(
            (x_start, y_mid), (x_end, y_mid),
            arrowstyle="->,head_length=0.18,head_width=0.12",
            linewidth=2.0,
            color="#555555",
            zorder=5,
        )
        ax.add_patch(arr)

    # Legend bar
    ax.text(8, 1.5, "Data flow: Raw sensor CSVs → feature engineering → model training →"
            " evaluation → uncertainty quantification → Streamlit dashboard",
            ha="center", va="center", fontsize=11, color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef2ff", edgecolor="#aab4d4"))

    save(fig, "fig01_pipeline",
         "Machine learning pipeline architecture from raw CSV telemetry to digital twin dashboard.\n"
         "exp_id: pipeline_architecture; source: requirements.txt and src/ttwin/")


# ═══════════════════════════════════════════════════════════════════════════════
# fig02 — Timeline
# ═══════════════════════════════════════════════════════════════════════════════
def fig02_timeline():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("Project Timeline (git tags as milestones)", fontsize=18, fontweight="bold")

    # Dates as ordinal numbers (approximate Oct date as Oct-15)
    import datetime
    dates = {
        "v0.1-audit\n(04628a0)": datetime.date(2026, 10, 15),
        "v0.9-pre-fix\n(d8c2cbc)": datetime.date(2026, 11, 8),
        "v1.0-corrected\n(aa0e974)": datetime.date(2026, 11, 9),
        "Snapshot\n(a6e99ee)": datetime.date(2026, 11, 12),
    }

    phase_spans = [
        ("Phase 0-1\n(repo+audit)", datetime.date(2026, 10, 1), datetime.date(2026, 10, 31), "#bde0ff"),
        ("Phase 2\n(modules+tests)", datetime.date(2026, 11, 1), datetime.date(2026, 11, 7), "#d5f5d5"),
        ("Phase 2\n(experiments)", datetime.date(2026, 11, 7), datetime.date(2026, 11, 10), "#ffd6a5"),
        ("Phase 2\n(snapshot)", datetime.date(2026, 11, 10), datetime.date(2026, 11, 13), "#ffd6d6"),
    ]

    ref = datetime.date(2026, 10, 1)

    def to_x(d):
        return (d - ref).days

    # Phase bands
    for label, d_start, d_end, color in phase_spans:
        x0_ = to_x(d_start)
        x1_ = to_x(d_end)
        ax.axvspan(x0_, x1_, alpha=0.35, color=color, zorder=1)
        ax.text((x0_ + x1_) / 2, 1.3, label, ha="center", va="bottom",
                fontsize=11, color="#333333", fontweight="bold")

    # Main timeline line
    x_min = to_x(datetime.date(2026, 9, 25))
    x_max = to_x(datetime.date(2026, 11, 18))
    ax.hlines(0, x_min, x_max, linewidth=3, color="#333333", zorder=2)

    # Milestones
    colors_m = ["#4a90d9", "#e8a838", "#e74c3c", "#1abc9c"]
    y_offsets = [0.6, -0.6, 0.6, -0.6]
    for (label, date), color, yo in zip(dates.items(), colors_m, y_offsets):
        xp = to_x(date)
        ax.scatter([xp], [0], s=160, color=color, zorder=5, edgecolors="white", linewidths=1.5)
        ax.vlines(xp, 0, yo, linewidth=1.2, color=color, linestyle="--", zorder=3)
        ax.text(xp, yo + (0.12 if yo > 0 else -0.12),
                f"{date.strftime('%Y-%m-%d')}\n{label}",
                ha="center", va="bottom" if yo > 0 else "top",
                fontsize=11, color=color, fontweight="bold")

    # X-axis ticks as month labels
    import datetime as dt
    tick_dates = [dt.date(2026, m, 1) for m in [10, 11]] + [dt.date(2026, 11, 8), dt.date(2026, 11, 12)]
    tick_xs = [to_x(d) for d in tick_dates]
    tick_labels = [d.strftime("%b %d") for d in tick_dates]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels, fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.4, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=14)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    save(fig, "fig02_timeline",
         "Project development timeline with phases, git commit history, and milestone tags.\n"
         "exp_id: project_milestones; source: git log (tags v0.1-audit, v0.9-pre-fix, v1.0-corrected)")


# ═══════════════════════════════════════════════════════════════════════════════
# fig03 — Splits schematic
# ═══════════════════════════════════════════════════════════════════════════════
def fig03_splits():
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.suptitle("Evaluation Splits — Temporal, Spatial, and Strict Holdout",
                 fontsize=18, fontweight="bold", y=0.97)

    # ── Panel 1: Split A chronological ──
    ax = axes[0]
    ax.set_title("Split A — Chronological", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")

    segments = [
        ("Train\n70% · 177.76d\n1,830,561 rows", 0.00, 0.70, "#4a90d9"),
        ("Val\n10% · 25.39d\n277,932 rows",        0.70, 0.80, "#5ba85a"),
        ("Embargo\n6h",                              0.80, 0.82, "#aaaaaa"),
        ("Test\n20% · 50.53d\n553,128 rows",        0.82, 1.00, "#e74c3c"),
    ]
    bar_y = 1.5
    bar_h = 0.6
    for label, x0_, x1_, color in segments:
        ax.barh(bar_y, x1_ - x0_, left=x0_, height=bar_h, color=color,
                edgecolor="white", linewidth=1.2)
        ax.text((x0_ + x1_) / 2, bar_y, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")

    ax.text(0.35, 2.4, "train end: 2019-04-05", ha="center", fontsize=10, color="#4a90d9")
    ax.text(0.91, 2.4, "test start: 2019-04-30", ha="center", fontsize=10, color="#e74c3c")
    ax.text(0.5, 0.5, "D1: 76 servers", ha="center", fontsize=11, color="#555555")

    # ── Panel 2: Split B GroupKFold ──
    ax = axes[1]
    ax.set_title("Split B — GroupKFold (5-fold)", fontsize=14, fontweight="bold")
    fold_counts = [16, 15, 15, 15, 15]
    fold_colors = ["#4a90d9", "#5ba85a", "#e8a838", "#9b59b6", "#e74c3c"]
    fold_labels = [f"Fold {i+1}: {n} servers" for i, n in enumerate(fold_counts)]
    bottoms = np.zeros(1)
    for i, (n, color, lbl) in enumerate(zip(fold_counts, fold_colors, fold_labels)):
        ax.bar([0], [n], bottom=bottoms, color=color, edgecolor="white",
               linewidth=1.2, label=lbl, width=0.4)
        ax.text(0, bottoms[0] + n / 2, lbl, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        bottoms += n
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 82)
    ax.set_xticks([])
    ax.set_ylabel("Number of Servers", fontsize=14)
    ax.set_xlabel("All 76 servers", fontsize=12)
    ax.text(0, 78, "Total: 76 servers", ha="center", fontsize=12, fontweight="bold", color="#333333")
    ax.set_facecolor("#f8f9fa")

    # ── Panel 3: Split C Strict Spatio-temporal ──
    ax = axes[2]
    ax.set_title("Split C — Strict Spatio-temporal", fontsize=14, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    seen_color = "#4a90d9"
    unseen_color = "#e74c3c"
    ax.barh([1], [61], color=seen_color, edgecolor="white", height=0.5, label="Seen servers (61)")
    ax.barh([0], [15], color=unseen_color, edgecolor="white", height=0.5, label="Unseen servers (15)")
    ax.text(30, 1, "Train/Val — 61 servers", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")
    ax.text(7.5, 0, "Test — 15 servers\n109,170 rows", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    ax.set_xlim(0, 76)
    ax.set_ylim(-0.5, 1.8)
    ax.set_xlabel("Server count", fontsize=14)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Unseen\n(test)", "Seen\n(train/val)"], fontsize=11)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig03_splits",
         "Partitioning schemas for temporal Split A, spatial GroupKFold Split B, and strict Split C.\n"
         "exp_id: split_protocols; source: docs/snapshot/04_protocol_and_splits.md")


# ═══════════════════════════════════════════════════════════════════════════════
# fig04 — Bug fixes before/after (4 panels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig04_bugs():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Four pipeline bugs found by the audit and fixed", fontsize=18, fontweight="bold")

    # Panel 1: S1 LightGBM vs persistence MAE at 10 min and 1 h
    ax1 = axes[0, 0]
    df_before = pd.read_csv("experiments/diag_unified_ablation/results.csv")
    df_after = pd.read_csv("experiments/v2_ablation/results_split_a.csv")

    b_s0_10 = df_before[(df_before["horizon"] == "10min") & (df_before["feature_set"] == "S0")]["mae"].values[0]
    b_s1_10 = df_before[(df_before["horizon"] == "10min") & (df_before["feature_set"] == "S1") & (df_before["model"] == "lightgbm")]["mae"].values[0]
    b_s0_1h = df_before[(df_before["horizon"] == "1h") & (df_before["feature_set"] == "S0")]["mae"].values[0]
    b_s1_1h = df_before[(df_before["horizon"] == "1h") & (df_before["feature_set"] == "S1") & (df_before["model"] == "lightgbm")]["mae"].values[0]

    a_s0_10 = df_after[(df_after["dataset"] == "dataset1") & (df_after["split"] == "Split A") & (df_after["horizon"] == "10min") & (df_after["feature_set"] == "S0")]["mae"].values[0]
    a_s1_10 = df_after[(df_after["dataset"] == "dataset1") & (df_after["split"] == "Split A") & (df_after["horizon"] == "10min") & (df_after["feature_set"] == "S1") & (df_after["model"] == "lightgbm")]["mae"].values[0]
    a_s0_1h = df_after[(df_after["dataset"] == "dataset1") & (df_after["split"] == "Split A") & (df_after["horizon"] == "1h") & (df_after["feature_set"] == "S0")]["mae"].values[0]
    a_s1_1h = df_after[(df_after["dataset"] == "dataset1") & (df_after["split"] == "Split A") & (df_after["horizon"] == "1h") & (df_after["feature_set"] == "S1") & (df_after["model"] == "lightgbm")]["mae"].values[0]

    x = np.arange(2)
    w = 0.2
    ax1.bar(x - 1.5*w, [b_s0_10, b_s0_1h], width=w, label="Before: Persistence S0", color="#95a5a6", edgecolor="white")
    ax1.bar(x - 0.5*w, [b_s1_10, b_s1_1h], width=w, label="Before: LightGBM S1 (worse)", color="#e74c3c", edgecolor="white")
    ax1.bar(x + 0.5*w, [a_s0_10, a_s0_1h], width=w, label="After: Persistence S0", color="#7f8c8d", edgecolor="white")
    ax1.bar(x + 1.5*w, [a_s1_10, a_s1_1h], width=w, label="After: LightGBM S1 (delta target)", color="#2ecc71", edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["10 min", "1 h"], fontsize=12)
    ax1.set_ylabel("MAE (°C)", fontsize=13)
    ax1.set_title("1. Target Formulation: Absolute vs Delta T", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.5)

    # Panel 2: Split A test share 31.7% -> 20.0%
    ax2 = axes[0, 1]
    shares = [31.7, 20.0]
    labels_share = ["Before Fix\n(Overlapping Val)", "After Fix\n(70/10/20 + 6h Embargo)"]
    bars_s = ax2.bar([0, 1], shares, width=0.4, color=["#e74c3c", "#2ecc71"], edgecolor="white")
    for bar, val in zip(bars_s, shares):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labels_share, fontsize=11)
    ax2.set_ylabel("Test Partition Share of Calendar Time (%)", fontsize=12)
    ax2.set_ylim(0, 40)
    ax2.set_title("2. Split A Boundary Overlap Fix", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.5)

    # Panel 3: Persistence MAE of 5 seeds vs 5 GroupKFold folds
    ax3 = axes[1, 0]
    df_seeds = pd.read_csv("experiments/diag_split_b_5seeds/results.csv")
    df_gk = pd.read_csv("experiments/v2_ablation/results_split_b_groupkfold.csv")
    df_gk_d1 = df_gk[(df_gk["dataset"] == "dataset1") & (df_gk["feature_set"] == "S0") & (df_gk["model"] == "persistence")].sort_values("fold")

    seeds_mae = df_seeds["mae_persistence"].values
    gk_mae = df_gk_d1["mae"].values

    idx = np.arange(1, 6)
    ax3.plot(idx, seeds_mae, "o--", color="#e74c3c", linewidth=2, markersize=8, label="Before: 5 Seeds (identical 0.43°C)")
    ax3.plot(idx, gk_mae, "s-", color="#2ecc71", linewidth=2, markersize=8, label="After: 5 GroupKFold Folds (real variance)")
    ax3.set_xticks(idx)
    ax3.set_xticklabels([f"Fold {i}" for i in idx], fontsize=11)
    ax3.set_xlabel("Partition / Seed Index", fontsize=12)
    ax3.set_ylabel("Persistence MAE (°C)", fontsize=13)
    ax3.set_title("3. Stratified Sampling Collapse in Split B", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9, loc="center right")
    ax3.grid(True, alpha=0.5)

    # Panel 4: Cluster-2 servers kept by data gate (2 -> 59)
    ax4 = axes[1, 1]
    servers_kept = [2, 59]
    labels_gate = ["Before Audit\n(All 18 Cols Evaluated)", "After Audit\n(Used Telemetry Only)"]
    bars_g = ax4.bar([0, 1], servers_kept, width=0.4, color=["#e74c3c", "#2ecc71"], edgecolor="white")
    for bar, val in zip(bars_g, servers_kept):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f"{val} servers", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(labels_gate, fontsize=11)
    ax4.set_ylabel("Qualifying Physical Servers (Count)", fontsize=12)
    ax4.set_ylim(0, 70)
    ax4.set_title("4. Cluster-2 Missingness Gate Filtering", fontsize=13, fontweight="bold")
    ax4.grid(axis="y", alpha=0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save(fig, "fig04_bugs_before_after",
         "Four pipeline audit bugs and their before-and-after resolution metrics.\n"
         "exp_id: audit_diagnostics; source: experiments/diag_unified_ablation/results.csv, results_split_a.csv, diag_split_b_5seeds/results.csv, results_split_b_groupkfold.csv, FIX_REPORT.md")


# ═══════════════════════════════════════════════════════════════════════════════
# fig05 — Skill vs persistence
# ═══════════════════════════════════════════════════════════════════════════════
def fig05_skill_vs_persistence():
    df = pd.read_csv("experiments/v2_ablation/results_split_a.csv")

    horizon_order = ["10min", "1h", "6h"]
    horizon_minutes = [10, 60, 360]

    # S0 persistence
    s0 = df[(df["feature_set"] == "S0") & (df["model"] == "persistence") &
            (df["dataset"] == "dataset1") & (df["split"] == "Split A")]
    s0 = s0.set_index("horizon").reindex(horizon_order)

    # S1 LightGBM
    s1 = df[(df["feature_set"] == "S1") & (df["model"] == "lightgbm") &
            (df["dataset"] == "dataset1") & (df["split"] == "Split A")]
    s1 = s1.set_index("horizon").reindex(horizon_order)

    # S4 LightGBM
    s4 = df[(df["feature_set"] == "S4") & (df["model"] == "lightgbm") &
            (df["dataset"] == "dataset1") & (df["split"] == "Split A")]
    s4 = s4.set_index("horizon").reindex(horizon_order)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("Monitoring Skill vs Persistence (Split A, Dataset 1)", fontsize=18, fontweight="bold")

    ax.plot(horizon_minutes, s0["mae"].values, "k--o", linewidth=2.2, markersize=9,
            label="S0 Persistence (reference)", zorder=5)
    ax.plot(horizon_minutes, s1["mae"].values, "b-o", linewidth=2.2, markersize=9,
            label="S1 LightGBM", zorder=4)
    ax.plot(horizon_minutes, s4["mae"].values, "g-s", linewidth=2.2, markersize=9,
            label="S4 LightGBM", zorder=4)

    ax.set_xlabel("Forecast Horizon (minutes)", fontsize=14)
    ax.set_ylabel("MAE (°C)", fontsize=14)
    ax.set_xticks(horizon_minutes)
    ax.set_xticklabels(["10 min", "1 h", "6 h"], fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)

    ax.text(0.98, 0.04,
            "Note: skill close to persistence baseline — no CI computed",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#f0ad4e"))

    fig.tight_layout()
    save(fig, "fig05_skill_vs_persistence",
         "Monitoring regime MAE across forecast horizons for persistence vs LightGBM models on Dataset 1 (Split A).\n"
         "exp_id: v2_ablation; source: experiments/v2_ablation/results_split_a.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# fig06 — Planning baselines
# ═══════════════════════════════════════════════════════════════════════════════
def fig06_planning_baselines():
    q1 = pd.read_csv("experiments/diag_part_a_q1_baselines.csv")
    q2 = pd.read_csv("experiments/diag_part_a_q2_contemporaneous.csv")

    means = {
        "Cluster Mean":           q1["cluster_mean_mae"].mean(),
        "Load-Cond. Mean":        q1["load_conditional_mean_mae"].mean(),
        "Ridge (CPU)":            q1["ridge_cpu_mae"].mean(),
        "Zero-Shot LightGBM":     q1["zero_shot_mae"].mean(),
        "Zero-Shot (contemp.)":   q2["zero_shot_contemp"].mean(),
    }

    labels = list(means.keys())
    values = list(means.values())
    colors = ["#9b59b6", "#e8a838", "#3498db", "#e74c3c", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("Planning Baselines — Mean Server MAE (°C)", fontsize=18, fontweight="bold")

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{val:.2f}°C", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Mean MAE (°C)", fontsize=14)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)

    ax.text(0.98, 0.97, "Diagnostic subset · 16 servers · 1 h horizon",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef2ff", edgecolor="#aab4d4"))

    fig.tight_layout()
    save(fig, "fig06_planning_baselines",
         "Planning regime temperature forecasting baselines evaluated on the 16-server diagnostic subset.\n"
         "exp_id: diag_part_a; source: experiments/diag_part_a_q1_baselines.csv and diag_part_a_q2_contemporaneous.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# fig07 — Adaptation curve
# ═══════════════════════════════════════════════════════════════════════════════
def fig07_adaptation_curve():
    df = pd.read_csv("experiments/v2_adaptation/summary_table.csv")
    plan = df[df["regime"] == "planning"].copy()

    zero_shot = plan[plan["strategy"] == "zero_shot"]
    cb = plan[plan["strategy"] == "continued_boosting"].sort_values("budget_hours")
    ro = plan[plan["strategy"] == "residual_offset"].sort_values("budget_hours")

    cluster_mean_val = plan[plan["strategy"] == "cluster_mean"]["mean_mae"].iloc[0]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("Cold-Start Adaptation — Planning MAE vs Local Data Budget (°C)",
                 fontsize=18, fontweight="bold")

    # Zero-shot single point
    zs_x = zero_shot["budget_hours"].values
    zs_y = zero_shot["mean_mae"].values
    zs_lo = zero_shot["ci_95_low"].values
    zs_hi = zero_shot["ci_95_high"].values
    ax.errorbar(zs_x, zs_y, yerr=[zs_y - zs_lo, zs_hi - zs_y],
                fmt="ko", markersize=10, capsize=6, linewidth=2, label="Zero-Shot (0 h)")

    # Continued boosting
    cb_x = cb["budget_hours"].values
    cb_y = cb["mean_mae"].values
    cb_lo = cb["ci_95_low"].values
    cb_hi = cb["ci_95_high"].values
    ax.plot(cb_x, cb_y, "b-o", linewidth=2.2, markersize=8, label="Continued Boosting")
    ax.fill_between(cb_x, cb_lo, cb_hi, alpha=0.18, color="blue", label="_nolegend_")

    # Residual offset
    ro_x = ro["budget_hours"].values
    ro_y = ro["mean_mae"].values
    ro_lo = ro["ci_95_low"].values
    ro_hi = ro["ci_95_high"].values
    ax.plot(ro_x, ro_y, "g-s", linewidth=2.2, markersize=8, label="Residual Offset")
    ax.fill_between(ro_x, ro_lo, ro_hi, alpha=0.18, color="green", label="_nolegend_")

    # Cluster mean reference
    ax.axhline(cluster_mean_val, color="#e74c3c", linestyle="--", linewidth=2.0,
               label=f"Cluster Mean baseline ({cluster_mean_val:.3f}°C)")

    ax.set_xlabel("Local Data Budget (hours)", fontsize=14)
    ax.set_ylabel("Mean MAE (°C)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    ax.set_xlim(-5, 180)

    fig.tight_layout()
    save(fig, "fig07_adaptation_curve",
         "Cold-start planning adaptation curves showing target server MAE across local training budgets with 95% confidence intervals.\n"
         "exp_id: v2_adaptation; source: experiments/v2_adaptation/summary_table.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# fig08 — Coverage by regime (2-panel)
# ═══════════════════════════════════════════════════════════════════════════════
def fig08_coverage_by_regime():
    cov = pd.read_csv("experiments/v2_uncertainty/coverage_by_regime_and_bin.csv")
    comp = pd.read_csv("experiments/v2_uncertainty/conformal_methods_comparison.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("Conformal Coverage by Regime and Method (90% Target)",
                 fontsize=18, fontweight="bold", y=0.98)

    # ── Panel 1: grouped bar by group ──
    groups = cov["group"].tolist()
    short_groups = [g.split("(")[0].strip() for g in groups]
    x = np.arange(len(groups))
    w = 0.25
    ax1.bar(x - w, cov["cov_const"].values, width=w, color="#4a90d9", label="Constant-Width", edgecolor="white")
    ax1.bar(x,     cov["cov_norm"].values,  width=w, color="#5ba85a", label="Normalized",     edgecolor="white")
    ax1.bar(x + w, cov["cov_cqr"].values,   width=w, color="#e8a838", label="CQR",            edgecolor="white")
    ax1.axhline(0.90, color="#e74c3c", linestyle="--", linewidth=2.0, label="90% target")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_groups, fontsize=11, rotation=20, ha="right")
    ax1.set_ylabel("Empirical Coverage", fontsize=14)
    ax1.set_ylim(0.75, 1.05)
    ax1.set_title("Coverage by Regime / Bin", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.5)
    ax1.set_facecolor("#f8f9fa")

    # ── Panel 2: horizontal bar — mean width ──
    methods_short = ["Constant-Width", "Normalized", "CQR"]
    widths = comp["mean_width_C"].values
    colors2 = ["#4a90d9", "#5ba85a", "#e8a838"]
    y2 = np.arange(len(methods_short))
    ax2.barh(y2, widths, color=colors2, edgecolor="white", height=0.4)
    for i, (w_, meth) in enumerate(zip(widths, methods_short)):
        ax2.text(w_ + 0.03, i, f"{w_:.2f}°C", va="center", fontsize=12, fontweight="bold")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(methods_short, fontsize=12)
    ax2.set_xlabel("Mean Interval Width (°C)", fontsize=14)
    ax2.set_title("Mean Width (°C)", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, max(widths) * 1.3)
    ax2.grid(axis="x", alpha=0.5)
    ax2.set_facecolor("#f8f9fa")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig08_coverage_by_regime",
         "Conformal uncertainty prediction interval empirical coverage across regimes and mean interval widths on Split C.\n"
         "exp_id: v2_uncertainty; source: experiments/v2_uncertainty/coverage_by_regime_and_bin.csv and conformal_methods_comparison.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# fig09 — Cross-cluster planning transfer
# ═══════════════════════════════════════════════════════════════════════════════
def fig09_cross_cluster():
    df = pd.read_csv("experiments/v2_transfer/cross_cluster_planning_results.csv")

    # Map to display labels in fixed order
    label_map = {
        "Cluster 2 Mean Baseline": "Cluster 2 Mean",
        "Native Cluster 2 Planning": "Native C2",
        "Transferred Planning": "Transferred C1→C2",
    }
    df["label"] = df["model"].map(label_map)
    df = df.set_index("label").reindex(["Cluster 2 Mean", "Native C2", "Transferred C1→C2"])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("Cross-Cluster Planning Transfer (C1→C2) — MAE (°C)",
                 fontsize=18, fontweight="bold")

    colors = ["#9b59b6", "#2ecc71", "#e8a838"]
    bars = ax.bar(df.index, df["mae"].values, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.4)

    for bar, val in zip(bars, df["mae"].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{val:.2f}°C", ha="center", va="bottom", fontsize=13, fontweight="bold")

    # Annotate transfer penalty: gap between Native C2 and Transferred
    native_mae = df.loc["Native C2", "mae"]
    trans_mae  = df.loc["Transferred C1→C2", "mae"]
    gap = trans_mae - native_mae
    x_native = 1   # index of Native C2
    x_trans  = 2   # index of Transferred
    y_ann = max(native_mae, trans_mae) + 0.5
    ax.annotate("", xy=(x_trans, trans_mae + 0.1), xytext=(x_native, native_mae + 0.1),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=2))
    ax.text((x_native + x_trans) / 2, y_ann,
            f"+{gap:.2f}°C transfer gap",
            ha="center", va="bottom", fontsize=12, color="#e74c3c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaea", edgecolor="#e74c3c"))

    ax.set_ylabel("Mean MAE (°C)", fontsize=14)
    ax.set_ylim(0, df["mae"].max() * 1.4)
    ax.grid(axis="y", alpha=0.5)
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, fontsize=12)

    fig.tight_layout()
    save(fig, "fig09_cross_cluster",
         "Cross-cluster planning performance transferring models from Cluster 1 to Cluster 2.\n"
         "exp_id: v2_transfer; source: experiments/v2_transfer/cross_cluster_planning_results.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# fig10 — SHAP feature importance
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_shap():
    df = pd.read_csv("experiments/v2_ablation/shap_summary.csv")
    top10 = df.nlargest(10, "mean_abs_shap").sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("SHAP Feature Importance — Top 10 (S4 LightGBM, 1h, 1000 samples)",
                 fontsize=18, fontweight="bold")

    cmap = matplotlib.colormaps.get_cmap("viridis").resampled(len(top10))
    colors = [cmap(i) for i in range(len(top10))]
    bars = ax.barh(top10["feature"], top10["mean_abs_shap"], color=colors,
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, top10["mean_abs_shap"].values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Mean |SHAP| value", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    ax.grid(axis="x", alpha=0.5)
    ax.set_facecolor("#f8f9fa")

    fig.tight_layout()
    save(fig, "fig10_shap",
         "Top 10 features ranked by mean absolute SHAP value for the S4 LightGBM model at 1-hour horizon.\n"
         "exp_id: v2_ablation; source: experiments/v2_ablation/shap_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating figures → {OUT_DIR}/\n")
    errors = {}

    funcs = [
        ("fig01_pipeline",           fig01_pipeline),
        ("fig02_timeline",           fig02_timeline),
        ("fig03_splits",             fig03_splits),
        ("fig04_bugs_before_after",  fig04_bugs),
        ("fig05_skill_vs_persistence", fig05_skill_vs_persistence),
        ("fig06_planning_baselines", fig06_planning_baselines),
        ("fig07_adaptation_curve",   fig07_adaptation_curve),
        ("fig08_coverage_by_regime", fig08_coverage_by_regime),
        ("fig09_cross_cluster",      fig09_cross_cluster),
        ("fig10_shap",               fig10_shap),
    ]

    for name, fn in funcs:
        try:
            fn()
        except Exception as exc:
            import traceback
            errors[name] = traceback.format_exc()
            print(f"  ✗  {name}: {exc}")

    print("\n── Summary ──")
    succeeded = [n for n, _ in funcs if n not in errors]
    print(f"Succeeded ({len(succeeded)}/10): {', '.join(succeeded)}")
    if errors:
        print(f"Failed ({len(errors)}):")
        for n, tb in errors.items():
            print(f"  {n}:\n{tb}")
    else:
        print("All 10 figures generated successfully.")

"""
Model Comparison Poster Figures
================================
Generates two poster-quality figures:
  Figure 1 — Overall Accuracy bar chart + Tradeoff Matrix
  Figure 2 — Mean Wetland F1 + Per-Class F1 grouped bars
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Class names (from truth source) ──────────────────────────────────────────
CLASS_NAMES = [
    "Fen\n(Graminoid)",
    "Fen\n(Woody)",
    "Marsh",
    "Shallow\nOpen Water",
    "Swamp",
]
CLASS_COLORS = [
    "#7BC67E",   # Fen Graminoid – light green
    "#2E7D32",   # Fen Woody     – dark green
    "#29B6F6",   # Marsh         – sky blue
    "#0D47A1",   # Shallow OW    – navy
    "#8D6E63",   # Swamp         – brown
]

# ── Model colours ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "CNN\n(U-Net v12)": "#1565C0",
    "RF Wetland\nOnly": "#2E7D32",
    "SVM\n(RBF)": "#E65100",
}
MODEL_LABELS = list(MODEL_COLORS.keys())
MODEL_PALETTE = list(MODEL_COLORS.values())

# ── Raw numbers ───────────────────────────────────────────────────────────────
OVERALL_ACC = [92.53, 82.74, 56.86]   # CNN, RF Wetland-Only, SVM  (RF accuracy on classes 1-5 only)

# Per-class F1 [cls1..cls5] using class IDs as per truth source
# Class order: 1=Fen(Graminoid), 2=Fen(Woody), 3=Marsh, 4=Shallow Open Water, 5=Swamp
PER_CLASS_F1 = {
    "CNN\n(U-Net v12)": [0.0000, 0.7039, 0.5774, 0.5815, 0.3375],
    "RF Wetland\nOnly": [0.7300, 0.8242, 0.9017, 0.6884, 0.6572],  # grid search, classes 1-5 only
    "SVM\n(RBF)":        [0.4526, 0.6559, 0.6595, 0.5697, 0.4941],
}

MEAN_WETLAND_F1 = {k: np.mean(v) for k, v in PER_CLASS_F1.items()}

# ── Tradeoff matrix data ───────────────────────────────────────────────────────
TRADEOFF_ROWS = [
    ("Overall Accuracy",        "✦ CNN (92.5%)",       "RF-Only (82.7%)*",  "SVM (56.9%)"),
    ("Mean Wetland F1",         "✦ RF-Only (0.760)",   "SVM (0.566)",       "CNN (0.440)"),
    ("Training Time",           "✦ SVM (~fast)",       "RF (~med)",         "CNN (~slow)"),
    ("Inference Speed",         "✦ RF (CPU)",          "SVM (CPU/GPU)",     "CNN (GPU)"),
    ("Cloud Inference Cost",    "✦ RF (cheapest)",     "SVM (moderate)",    "CNN (highest)"),
    ("GPU Required",            "✦ RF / SVM — No",     "—",                 "CNN — Yes"),
    ("Spatial Context",         "✦ CNN (conv layers)", "—",                 "RF/SVM — No"),
    ("Interpretability",        "✦ RF (feat. import.)","SVM — moderate",    "CNN — black box"),
    ("Rare Class (Fen Gram.)",  "✦ RF (F1=0.730)",     "SVM (F1=0.453)",    "CNN (F1=0.000)"),
]
TRADEOFF_HEADERS = ["Criterion", "Best", "2nd", "Worst"]


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Accuracy Bar Chart + Tradeoff Matrix
# ═════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(16, 11))
gs1 = GridSpec(2, 1, figure=fig1, height_ratios=[1.3, 1.8], hspace=0.45)

# ── Panel A: Accuracy bars ────────────────────────────────────────────────────
ax_acc = fig1.add_subplot(gs1[0])
x = np.arange(len(MODEL_LABELS))
bars = ax_acc.bar(x, OVERALL_ACC, width=0.5, color=MODEL_PALETTE,
                  edgecolor="white", linewidth=1.2, zorder=3)

ax_acc.set_ylim(0, 115)
ax_acc.set_xticks(x)
ax_acc.set_xticklabels(MODEL_LABELS, fontsize=13)
ax_acc.set_ylabel("Overall Accuracy (%)", fontsize=12)
ax_acc.set_title("Overall Accuracy — Final Model Comparison\n* RF accuracy evaluated on wetland pixels only (classes 1–5)", fontsize=13, fontweight="bold", pad=10)
ax_acc.axhline(100, color="grey", lw=0.6, ls="--", alpha=0.4)
ax_acc.yaxis.grid(True, alpha=0.3, zorder=0)
ax_acc.set_axisbelow(True)

for bar, val in zip(bars, OVERALL_ACC):
    ax_acc.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

ax_acc.text(-0.55, 115, "(A)", fontsize=13, fontweight="bold", va="top")

# ── Panel B: Tradeoff Matrix ──────────────────────────────────────────────────
ax_tbl = fig1.add_subplot(gs1[1])
ax_tbl.axis("off")
ax_tbl.set_title("Model Tradeoff Matrix", fontsize=15, fontweight="bold", pad=6)

col_widths = [0.26, 0.28, 0.22, 0.22]
col_colours_header = ["#37474F", "#1B5E20", "#E65100", "#B71C1C"]
row_fill_colours = ["#F5F5F5", "#ECEFF1"]
best_fill   = "#E8F5E9"
second_fill = "#FFF3E0"
worst_fill  = "#FFEBEE"
cell_fills  = [best_fill, second_fill, worst_fill]

n_rows = len(TRADEOFF_ROWS)
row_h  = 1.0 / (n_rows + 1.5)
xs     = [sum(col_widths[:i]) for i in range(4)]

# Header
for j, (hdr, cw, xpos, hcol) in enumerate(
        zip(TRADEOFF_HEADERS, col_widths, xs, col_colours_header)):
    rect = plt.Rectangle((xpos, 1 - row_h), cw, row_h,
                          transform=ax_tbl.transAxes,
                          color=hcol, zorder=2, clip_on=False)
    ax_tbl.add_patch(rect)
    ax_tbl.text(xpos + cw / 2, 1 - row_h / 2, hdr,
                transform=ax_tbl.transAxes,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white", zorder=3)

# Data rows
for i, row in enumerate(TRADEOFF_ROWS):
    y_top = 1 - (i + 1) * row_h
    base_fill = row_fill_colours[i % 2]
    for j, (cell, cw, xpos) in enumerate(zip(row, col_widths, xs)):
        fill = base_fill if j == 0 else cell_fills[j - 1]
        rect = plt.Rectangle((xpos, y_top - row_h), cw, row_h,
                              transform=ax_tbl.transAxes,
                              color=fill, zorder=2, clip_on=False,
                              linewidth=0.5, edgecolor="#CFD8DC")
        ax_tbl.add_patch(rect)
        fw = "bold" if j == 0 else ("bold" if "✦" in cell else "normal")
        ax_tbl.text(xpos + cw / 2, y_top - row_h / 2, cell,
                    transform=ax_tbl.transAxes,
                    ha="center", va="center", fontsize=9.5,
                    fontweight=fw, zorder=3, wrap=True)

ax_tbl.text(-0.01, 1.05, "(B)", transform=ax_tbl.transAxes,
            fontsize=13, fontweight="bold", va="top")

fig1.suptitle("Wetland Classification — Model Performance & Tradeoffs",
              fontsize=17, fontweight="bold", y=0.98)

fig1.savefig("model_accuracy_tradeoff.png", dpi=200, bbox_inches="tight",
             facecolor="white")
print("Saved: model_accuracy_tradeoff.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Mean Wetland F1 + Per-Class F1 Grouped Bars
# ═════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle("Wetland Class F1 Score — Final Model Comparison",
              fontsize=17, fontweight="bold", y=1.01)

# ── Panel A: Mean Wetland F1 ──────────────────────────────────────────────────
ax_mf1 = axes2[0]
mean_vals = [MEAN_WETLAND_F1[m] for m in MODEL_LABELS]
bars2 = ax_mf1.bar(np.arange(len(MODEL_LABELS)), mean_vals, width=0.5,
                   color=MODEL_PALETTE, edgecolor="white", linewidth=1.2, zorder=3)

ax_mf1.set_ylim(0, 0.85)
ax_mf1.set_xticks(np.arange(len(MODEL_LABELS)))
ax_mf1.set_xticklabels(MODEL_LABELS, fontsize=12)
ax_mf1.set_ylabel("Mean Wetland F1 (Classes 1–5)", fontsize=12)
ax_mf1.set_title("Mean Wetland F1 Score", fontsize=14, fontweight="bold")
ax_mf1.yaxis.grid(True, alpha=0.3, zorder=0)
ax_mf1.set_axisbelow(True)

for bar, val in zip(bars2, mean_vals):
    ax_mf1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

ax_mf1.text(-0.12, 1.04, "(A)", transform=ax_mf1.transAxes,
            fontsize=13, fontweight="bold", va="top")

# ── Panel B: Per-Class F1 Grouped Bars ───────────────────────────────────────
ax_pcf1 = axes2[1]
n_classes = len(CLASS_NAMES)
n_models  = len(MODEL_LABELS)
group_w   = 0.7
bar_w     = group_w / n_models
x_centers = np.arange(n_classes)

for mi, (model, color) in enumerate(zip(MODEL_LABELS, MODEL_PALETTE)):
    offsets = x_centers - group_w / 2 + bar_w * mi + bar_w / 2
    f1s = PER_CLASS_F1[model]
    ax_pcf1.bar(offsets, f1s, width=bar_w * 0.92,
                color=color, edgecolor="white", linewidth=0.8,
                label=model.replace("\n", " "), zorder=3)

ax_pcf1.set_xticks(x_centers)
ax_pcf1.set_xticklabels(CLASS_NAMES, fontsize=10.5)
ax_pcf1.set_ylim(0, 1.05)
ax_pcf1.set_ylabel("F1 Score", fontsize=12)
ax_pcf1.set_title("Per-Class F1 Score by Model", fontsize=14, fontweight="bold")
ax_pcf1.yaxis.grid(True, alpha=0.3, zorder=0)
ax_pcf1.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(color=c, label=m.replace("\n", " "))
    for m, c in zip(MODEL_LABELS, MODEL_PALETTE)
]
ax_pcf1.legend(handles=legend_handles, fontsize=10, loc="upper right",
               framealpha=0.85, edgecolor="#CFD8DC")

ax_pcf1.text(-0.08, 1.04, "(B)", transform=ax_pcf1.transAxes,
             fontsize=13, fontweight="bold", va="top")

fig2.tight_layout()
fig2.savefig("model_wetland_f1_comparison.png", dpi=200, bbox_inches="tight",
             facecolor="white")
print("Saved: model_wetland_f1_comparison.png")

print("\nDone.")

"""
Final RF Model — Stats Table + Confusion Matrix Figure
=======================================================
Model: RF Standalone (200 trees, depth=25, spatial split)
Source: Statistics/RF/3 - final_rf_wetland_model_56.json
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final Stats")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Background",
    "Fen\n(Graminoid)",
    "Fen\n(Woody)",
    "Marsh",
    "Shallow\nOpen Water",
    "Swamp",
]

conf_matrix = np.array([
    [85729,  1156,  5363, 38232,  8621, 10899],
    [   10,  4780,    12,     0,     0,     4],
    [11287,  3145, 22408,     0,     0,   660],
    [38675,    40,   445, 72165, 11691,  1984],
    [ 5158,    18,   153,  4975, 26240,   956],
    [ 5845,   812,  3107,   988,   603, 13645],
])

N = conf_matrix.sum()
row_sums = conf_matrix.sum(axis=1)
col_sums = conf_matrix.sum(axis=0)
diagonal = np.diag(conf_matrix)

p_o = diagonal.sum() / N
p_e = (row_sums * col_sums).sum() / (N ** 2)
kappa = (p_o - p_e) / (1 - p_e)

accuracy        = 0.5923
f1_weighted     = 0.5924
mean_f1         = (0.5779 + 0.6478 + 0.6496 + 0.5980 + 0.6199 + 0.5135) / 6

# Row-normalised confusion matrix (% of actual class)
conf_pct = conf_matrix / row_sums[:, None] * 100

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9))
gs  = gridspec.GridSpec(
    2, 1,
    height_ratios=[1, 5],
    hspace=0.35,
    left=0.02, right=0.98, top=0.93, bottom=0.04
)

# ── Panel 1: Overall metrics table ───────────────────────────────────────────
ax_table = fig.add_subplot(gs[0])
ax_table.axis("off")

col_labels  = ["Accuracy", "Weighted F1", "Mean F1\n(all classes)", "Cohen's Kappa"]
cell_values = [[f"{accuracy:.1%}", f"{f1_weighted:.3f}", f"{mean_f1:.3f}", f"{kappa:.3f}"]]

tbl = ax_table.table(
    cellText=cell_values,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0.05, 0.0, 0.90, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)

for (row, col), cell in tbl.get_celld().items():
    cell.set_linewidth(0.8)
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.45)
    else:
        cell.set_facecolor("#f4f6f7")
        cell.set_height(0.45)

ax_table.set_title(
    "Random Forest — Final Model Performance v2 (Spatial Split, Classes 0–5)",
    fontsize=13, fontweight="bold", pad=8
)

# ── Panel 2: Confusion matrix heatmap ────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1])

cmap = LinearSegmentedColormap.from_list(
    "wetland_cm", ["#f7fbff", "#c6dbef", "#2171b5", "#08306b"]
)

im = ax_cm.imshow(conf_pct, cmap=cmap, vmin=0, vmax=100, aspect="auto")

n_classes = len(CLASS_NAMES)
for i in range(n_classes):
    for j in range(n_classes):
        pct   = conf_pct[i, j]
        count = conf_matrix[i, j]
        color = "white" if pct > 50 else "#1a1a1a"
        ax_cm.text(
            j, i,
            f"{pct:.1f}%\n({count:,})",
            ha="center", va="center",
            fontsize=7.5, color=color,
            fontweight="bold" if i == j else "normal",
        )

ax_cm.set_xticks(range(n_classes))
ax_cm.set_yticks(range(n_classes))
ax_cm.set_xticklabels(CLASS_NAMES, fontsize=9)
ax_cm.set_yticklabels(CLASS_NAMES, fontsize=9)
ax_cm.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
ax_cm.set_ylabel("Actual Class", fontsize=11, labelpad=8)
ax_cm.set_title("Confusion Matrix (row-normalised %)", fontsize=11, pad=8)

# Grid lines between cells
for x in np.arange(-0.5, n_classes, 1):
    ax_cm.axhline(x, color="white", linewidth=0.8)
    ax_cm.axvline(x, color="white", linewidth=0.8)

cbar = plt.colorbar(im, ax=ax_cm, fraction=0.03, pad=0.02)
cbar.set_label("% of actual class", fontsize=9)

out_path = os.path.join(OUTPUT_DIR, "rf_final_stats_table.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()

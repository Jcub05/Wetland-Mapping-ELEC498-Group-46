"""
RF Model Iteration Timeline
============================
Line chart showing Mean Wetland F1 across 4 RF development milestones,
with annotations for each key decision made.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final Stats")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
# Mean F1 across all 6 classes (0–5) for v1–v3.5; wetland-only (1–5) for v4
milestones = [1, 2, 3, 3.5, 4]

_v3_f1 = (0.4066 + 0.6316 + 0.7231 + 0.6776 + 0.6593 + 0.4914) / 6
_v35_f1 = 0.6011

mean_wetland_f1 = [
    (0.9431 + 0.9762 + 0.9283 + 0.9382 + 0.9092 + 0.8652) / 6,  # v1: random split (leakage)
    (0.1362 + 0.8539 + 0.2395 + 0.2231 + 0.0003 + 0.0005) / 6,  # v2: spatial split, class issue
    _v3_f1,                                                        # v3: improved dataset/params
    _v35_f1,                                                       # v3.5: adjusted weights (bg fix)
    (0.7300 + 0.8242 + 0.9017 + 0.6884 + 0.6572) / 5,           # v4: background removed (classes 1–5 only)
]

labels = [
    "v1\n(Random Split)",
    "v2\n(Spatial Split)",
    "v3\n(Tuned Params)",
    "v3.5\n(Bg Fixed)",
    "v4\n(No Background)",
]

annotations = [
    f"Mean F1: {(0.9431+0.9762+0.9283+0.9382+0.9092+0.8652)/6:.1%} — but\nspatial data leakage",
    f"Mean F1 drops to {(0.1362+0.8539+0.2395+0.2231+0.0003+0.0005)/6:.1%}\n— class imbalance exposed",
    f"Iterated dataset &\nparameters: Mean F1 {_v3_f1:.1%}",
    f"Adjusted class weights —\nbg recall improved: 0.28 → 0.57\nMean F1: {_v35_f1:.3f}",
    f"Background removed (classes 1–5 only)\nwetland-only pipeline: Mean F1 {(0.7300+0.8242+0.9017+0.6884+0.6572)/5:.1%}",
]

# Milestone types: 'bad' = inflated/misleading, 'drop' = regression, 'improve' = gain
colors = ["#e05c5c", "#c0392b", "#2980b9", "#1abc9c", "#27ae60"]
marker_styles = ["D", "v", "o", "s", "^"]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Dashed line connecting all points
ax.plot(milestones, mean_wetland_f1,
        color="#888888", linewidth=1.5, linestyle="--", zorder=1)

# Solid line for v2→v3→v4 (the "real" progress after fixing leakage)
ax.plot(milestones[1:], mean_wetland_f1[1:],
        color="#2c3e50", linewidth=2.5, zorder=2)

# Plot each milestone marker
for i, (x, y, color, marker) in enumerate(zip(milestones, mean_wetland_f1, colors, marker_styles)):
    ax.scatter(x, y, color=color, s=120, zorder=5, marker=marker)

# Annotation boxes
annotation_offsets = [
    (0.10, 0.10),    # v1 — shift right and up (above spatial leakage zone text)
    (0.10, -0.13),   # v2 — shift right and down
    (-0.38, -0.16),  # v3 — shift left and down (avoid v3.5 annotation)
    (0.10, 0.00),    # v3.5 — shift right and up
    (-0.55, 0.06),   # v4 — shift left and up
]

for i, (x, y, ann, color, (dx, dy)) in enumerate(
        zip(milestones, mean_wetland_f1, annotations, colors, annotation_offsets)):
    ax.annotate(
        ann,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=8.5,
        color="#2c3e50",
        ha="left" if dx > 0 else "right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )

# Shaded region indicating inflated zone
ax.axhspan(0.85, 1.0, alpha=0.07, color="#e05c5c",
           label="Inflated region (spatial leakage)")
ax.text(1.05, 0.955, "Spatial leakage zone", fontsize=8,
        color="#e05c5c", style="italic", va="center")

# Reference line for v1 (inflated baseline)
ax.axhline(mean_wetland_f1[0], color="#e05c5c", linewidth=0.8,
           linestyle=":", alpha=0.5)

# Axis formatting
ax.set_xticks(milestones)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Mean F1 Score\n(avg. across all classes)", fontsize=11)
ax.set_ylim(0.0, 1.05)
ax.set_xlim(0.6, 4.6)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_title("Random Forest Model Iteration Progress", fontsize=14, fontweight="bold", pad=14)

# Milestone number labels on x-axis background bars
for x in milestones:
    ax.axvline(x, color="#eeeeee", linewidth=8, zorder=0)

# Value labels on each point
for x, y in zip(milestones, mean_wetland_f1):
    ax.text(x, y - 0.035, f"{y:.3f}", ha="center", va="top", fontsize=9,
            fontweight="bold", color="#2c3e50")

# Legend for dashed vs solid line
dashed_patch = mpatches.Patch(color="#888888", label="Including inflated v1")
solid_patch   = mpatches.Patch(color="#2c3e50", label="Real progress (post-fix)")
ax.legend(handles=[dashed_patch, solid_patch], fontsize=9,
          loc="lower right", framealpha=0.9)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "rf_iteration_timeline.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()

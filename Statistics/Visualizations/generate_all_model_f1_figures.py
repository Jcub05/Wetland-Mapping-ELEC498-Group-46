"""
All-Model Wetland F1 Comparison Figures
=========================================
Generates two figures comparing all 6 models across Statistics subfolders:
  Figure 1 — Mean Wetland F1 bar chart  (one bar per model)
  Figure 2 — Per-Class F1 grouped bars  (one group per wetland class)

Class structure note
--------------------
All classes are labelled according to the truth source (class_names_truth_source.txt):
  class 1 = Fen (Graminoid)
  class 2 = Fen (Woody)
  class 3 = Marsh
  class 4 = Shallow Open Water
  class 5 = Swamp

CNN / CNN+RF models store these classes under different internal names (Bog, Fen, Swamp,
Open Water) but share the same numeric class IDs, so alignment by ID is correct.
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── File paths ────────────────────────────────────────────────────────────────
STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

FILES = {
    "CNN\n(U-Net v12)":   os.path.join(STATS_DIR, "CNN",       "Model_CNN_statistics.json"),
    "CNN+RF\nPipeline":   os.path.join(STATS_DIR, "CNN + RF",  "Model_CNN_to_RF_statistics.json"),
    "RF\n(Spatial)":      os.path.join(STATS_DIR, "RF",        "3 - final_rf_wetland_model_56.json"),
    "RF+RF\nPipeline":    os.path.join(STATS_DIR, "RF + RF",   "final_Model_RF_to_RF_statistics_20260309_235639.json"),
    "SVM\n(RBF)":         os.path.join(STATS_DIR, "SVM",       "svm_metadata_20260309_170939 (1).json"),
    "SVM+SVM\nPipeline":  os.path.join(STATS_DIR, "SVM + SVM", "svm_combo_pipeline_stats_20260310_032810.json"),
}

# Models whose per_class_metrics use class name strings (CNN class structure)
CNN_MODELS = {"CNN\n(U-Net v12)", "CNN+RF\nPipeline"}

# X-axis labels — truth source (class_names_truth_source.txt)
CLASS_NAMES = [
    "Fen\n(Graminoid)",
    "Fen\n(Woody)",
    "Marsh",
    "Shallow\nOpen Water",
    "Swamp",
]

MODEL_LABELS = list(FILES.keys())

MODEL_COLORS = {
    "CNN\n(U-Net v12)":  "#1565C0",   # deep blue
    "CNN+RF\nPipeline":  "#42A5F5",   # sky blue
    "RF\n(Spatial)":     "#2E7D32",   # dark green
    "RF+RF\nPipeline":   "#66BB6A",   # light green
    "SVM\n(RBF)":        "#E65100",   # deep orange
    "SVM+SVM\nPipeline": "#F9A825",   # amber
}
MODEL_PALETTE = [MODEL_COLORS[m] for m in MODEL_LABELS]


# ── Data extraction helpers ───────────────────────────────────────────────────

def extract_wetland_f1s(model_name: str, data: dict) -> list:
    """Return a list of 5 F1 scores for wetland classes 1–5."""
    if model_name in CNN_MODELS:
        # Keys are class name strings; class order: Bog, Fen, Marsh, Swamp, Open Water
        pcm = data["per_class_metrics"]
        return [pcm[k]["f1_score"] for k in ["Bog", "Fen", "Marsh", "Swamp", "Open Water"]]

    pcm = data.get("per_class_metrics") or data.get("per_class", {})

    if "1" in pcm:
        # Numeric string keys (RF, RF+RF, SVM)
        key = "f1_score" if "f1_score" in pcm["1"] else "f1"
        return [pcm[str(i)][key] for i in range(1, 6)]

    # Named keys (SVM+SVM)
    keys = ["Fen (Graminoid)", "Fen (Woody)", "Marsh", "Shallow OW", "Swamp"]
    return [pcm[k]["f1"] for k in keys]


def get_mean_wetland_f1(data: dict, f1s: list) -> float:
    """Return pre-computed mean if present, otherwise compute from f1s."""
    om = data.get("overall_metrics", {})
    if "mean_wetland_f1" in om:
        return om["mean_wetland_f1"]
    if "mean_wetland_f1" in data:
        return float(data["mean_wetland_f1"])
    return float(np.mean(f1s))


# ── Load all models ───────────────────────────────────────────────────────────
per_class_f1 = {}
mean_wetland_f1 = {}

for model_name, filepath in FILES.items():
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    f1s = extract_wetland_f1s(model_name, data)
    per_class_f1[model_name] = f1s
    mean_wetland_f1[model_name] = get_mean_wetland_f1(data, f1s)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Mean Wetland F1 (all models)
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(13, 6))

mean_vals = [mean_wetland_f1[m] for m in MODEL_LABELS]
x = np.arange(len(MODEL_LABELS))
bars = ax1.bar(x, mean_vals, width=0.55, color=MODEL_PALETTE,
               edgecolor="white", linewidth=1.2, zorder=3)

ax1.set_ylim(0, max(mean_vals) * 1.22)
ax1.set_xticks(x)
ax1.set_xticklabels(MODEL_LABELS, fontsize=12)
ax1.set_ylabel("Mean Wetland F1 (Classes 1–5)", fontsize=12)
ax1.set_title("Mean Wetland F1 Score — All Model Comparison",
              fontsize=14, fontweight="bold", pad=12)
ax1.yaxis.grid(True, alpha=0.3, zorder=0)
ax1.set_axisbelow(True)

for bar, val in zip(bars, mean_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
             f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax1.text(0.01, 0.01,
         "Classes 1–5: Fen (Graminoid), Fen (Woody), Marsh, Shallow Open Water, Swamp  "
         "(truth source — aligned by class ID across all models)",
         transform=ax1.transAxes, fontsize=8, color="grey", va="bottom")

fig1.tight_layout()
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "all_models_mean_wetland_f1.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Per-Class F1 grouped bars (all models)
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(18, 7))

n_classes = len(CLASS_NAMES)
n_models  = len(MODEL_LABELS)
group_w   = 0.80
bar_w     = group_w / n_models
x_centers = np.arange(n_classes)

for mi, (model, color) in enumerate(zip(MODEL_LABELS, MODEL_PALETTE)):
    offsets = x_centers - group_w / 2 + bar_w * mi + bar_w / 2
    f1s = per_class_f1[model]
    ax2.bar(offsets, f1s, width=bar_w * 0.88, color=color,
            edgecolor="white", linewidth=0.7, zorder=3)

ax2.set_xticks(x_centers)
ax2.set_xticklabels(CLASS_NAMES, fontsize=9)
ax2.set_ylim(0, 1.10)
ax2.set_ylabel("F1 Score", fontsize=12)
ax2.set_title("Per-Class F1 Score — All Model Comparison",
              fontsize=14, fontweight="bold", pad=12)
ax2.yaxis.grid(True, alpha=0.3, zorder=0)
ax2.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(color=MODEL_COLORS[m], label=m.replace("\n", " "))
    for m in MODEL_LABELS
]
ax2.legend(handles=legend_handles, fontsize=10, loc="upper right",
           framealpha=0.85, edgecolor="#CFD8DC")

fig2.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "all_models_per_class_f1.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out2}")

print("\nDone.")

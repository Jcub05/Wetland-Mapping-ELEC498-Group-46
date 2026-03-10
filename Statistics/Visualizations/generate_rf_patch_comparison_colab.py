"""
RF Patch Comparison — Colab Script
====================================
Runs the RF v2 model over the geographic test band (rows 9216–12288),
then automatically selects:

  - One WELL-classified 512×512 patch  (highest accuracy, multi-class)
  - One POORLY-classified 512×512 patch (lowest accuracy, multi-class)

Output: 4-panel figure
  Row 1:  Well-classified  |  Ground Truth   RF Prediction
  Row 2:  Poorly-classified|  Ground Truth   RF Prediction

USAGE (Colab):
  1. Mount Google Drive
  2. !pip install -q rasterio joblib
  3. Run this script

PATHS — edit these to match your Drive layout:
"""

# ── Mount Drive ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, sys
import numpy as np
import joblib
import rasterio
from rasterio.windows import Window
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  ← edit if your Drive layout differs
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_BASE    = '/content/drive/MyDrive/EarthEngine'

LABELS_PATH   = os.path.join(DRIVE_BASE, 'bow_river_wetlands_10m_final.tif')
MODEL_PATH    = os.path.join(DRIVE_BASE, 'rf_wetland_model_middle_v2_20260310_124941.pkl')
SCALER_PATH   = os.path.join(DRIVE_BASE, 'rf_scaler_middle_v2_20260310_124941.pkl')
EMBEDDINGS_DIR = DRIVE_BASE                    # folder containing *-ROW-COL.tif tiles
OUTPUT_PATH   = os.path.join('/content/drive/MyDrive', 'rf_patch_comparison.png')

# Test band from metadata
TEST_ROW_MIN  = 9216
TEST_ROW_MAX  = 12288

# Patch scanning parameters
PATCH_SIZE    = 512          # pixels (10 m/px → 5.12 km patch)
MIN_CLASSES   = 2            # minimum unique GT classes required in a patch
MIN_WETLAND_CLASSES = 1      # at least 1 wetland class (1–5) required
MIN_VALID_COVERAGE = 0.85    # fraction of patch that must have valid predictions

# ─────────────────────────────────────────────────────────────────────────────
# CLASS DEFINITIONS (truth source: class_names_truth_source.txt)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: 'Background',
    1: 'Fen (Graminoid)',
    2: 'Fen (Woody)',
    3: 'Marsh',
    4: 'Shallow Open Water',
    5: 'Swamp',
}

CLASS_COLORS = np.array([
    [0.75, 0.75, 0.75],   # 0 Background        – light grey
    [0.85, 0.75, 0.30],   # 1 Fen (Graminoid)   – straw yellow
    [0.40, 0.65, 0.25],   # 2 Fen (Woody)       – olive green
    [0.20, 0.55, 0.20],   # 3 Marsh             – mid green
    [0.20, 0.55, 0.90],   # 4 Shallow Open Water – sky blue
    [0.00, 0.30, 0.55],   # 5 Swamp             – dark blue-green
], dtype=np.float32)

NODATA = 255


def labels_to_rgb(arr):
    rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
    for cls, color in enumerate(CLASS_COLORS):
        rgb[arr == cls] = color
    rgb[arr == NODATA] = [0.12, 0.12, 0.12]   # unmapped → near-black
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load model and scaler
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

assert os.path.exists(MODEL_PATH),  f"Model not found: {MODEL_PATH}"
assert os.path.exists(SCALER_PATH), f"Scaler not found: {SCALER_PATH}"

rf_model = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
print(f"  Model:  {os.path.basename(MODEL_PATH)}  ({rf_model.n_estimators} trees)")
print(f"  Scaler: {os.path.basename(SCALER_PATH)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Read ground truth dimensions
# ─────────────────────────────────────────────────────────────────────────────
print("\nReading raster dimensions...")
with rasterio.open(LABELS_PATH) as src:
    FULL_HEIGHT = src.height
    FULL_WIDTH  = src.width
    raster_transform = src.transform
    raster_crs       = src.crs

BAND_HEIGHT = TEST_ROW_MAX - TEST_ROW_MIN
print(f"  Full raster:  {FULL_HEIGHT} × {FULL_WIDTH}")
print(f"  Test band:    rows {TEST_ROW_MIN}–{TEST_ROW_MAX}  ({BAND_HEIGHT} rows × {FULL_WIDTH} cols)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Load ground truth for test band
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading ground truth test band...")
with rasterio.open(LABELS_PATH) as src:
    gt_window    = Window(0, TEST_ROW_MIN, FULL_WIDTH, BAND_HEIGHT)
    ground_truth = src.read(1, window=gt_window).astype(np.int16)
print(f"  GT shape: {ground_truth.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Discover test tiles and run inference
# ─────────────────────────────────────────────────────────────────────────────
print("\nDiscovering test tiles...")
from pathlib import Path

all_tiles = sorted(Path(EMBEDDINGS_DIR).glob('*.tif'))
test_tiles = []
for tf in all_tiles:
    parts = tf.stem.split('-')
    if len(parts) >= 3:
        try:
            row_off = int(parts[-2])
            if TEST_ROW_MIN <= row_off <= TEST_ROW_MAX:
                test_tiles.append(tf)
        except ValueError:
            pass

print(f"  Found {len(test_tiles)} test tiles")
if not test_tiles:
    raise RuntimeError("No test tiles found. Check EMBEDDINGS_DIR and tile naming (*-ROW-COL.tif).")

print("\nRunning inference...")
predictions = np.full((BAND_HEIGHT, FULL_WIDTH), NODATA, dtype=np.uint8)

for tile_path in test_tiles:
    parts = tile_path.stem.split('-')
    try:
        tile_row_off = int(parts[-2])
        tile_col_off = int(parts[-1])
    except (ValueError, IndexError):
        continue

    with rasterio.open(tile_path) as tile_src:
        if tile_src.count != 64:
            continue
        tile_h = tile_src.height
        tile_w = tile_src.width

        abs_row_start = tile_row_off
        abs_row_end   = tile_row_off + tile_h

        clip_row_start = max(abs_row_start, TEST_ROW_MIN)
        clip_row_end   = min(abs_row_end,   TEST_ROW_MIN + BAND_HEIGHT)
        if clip_row_start >= clip_row_end:
            continue

        tile_local_row_start = clip_row_start - abs_row_start
        tile_local_row_end   = clip_row_end   - abs_row_start
        valid_h  = tile_local_row_end - tile_local_row_start
        valid_w  = min(tile_w, FULL_WIDTH - tile_col_off)

        tile_data = tile_src.read(
            window=Window(0, tile_local_row_start, valid_w, valid_h)
        )  # (64, valid_h, valid_w)

        n_pixels = valid_h * valid_w
        pixels   = tile_data.reshape(64, n_pixels).T
        nan_mask = ~np.isnan(pixels).any(axis=1)

        preds_flat = np.full(n_pixels, NODATA, dtype=np.uint8)
        if nan_mask.any():
            X_valid = scaler.transform(pixels[nan_mask])
            preds_flat[nan_mask] = rf_model.predict(X_valid).astype(np.uint8)

        pred_2d = preds_flat.reshape(valid_h, valid_w)

        out_r0 = clip_row_start - TEST_ROW_MIN
        out_r1 = out_r0 + valid_h
        col_s  = tile_col_off
        col_e  = col_s + valid_w

        predictions[out_r0:out_r1, col_s:col_e] = pred_2d

print("  Inference complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Scan patches and score
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nScanning {PATCH_SIZE}×{PATCH_SIZE} patches for good/bad examples...")

candidates = []   # (accuracy, row_start, col_start)

n_row_patches = BAND_HEIGHT  // PATCH_SIZE
n_col_patches = FULL_WIDTH   // PATCH_SIZE

for pr in range(n_row_patches):
    r0 = pr * PATCH_SIZE
    r1 = r0 + PATCH_SIZE
    for pc in range(n_col_patches):
        c0 = pc * PATCH_SIZE
        c1 = c0 + PATCH_SIZE

        gt_patch   = ground_truth[r0:r1, c0:c1]
        pred_patch = predictions [r0:r1, c0:c1]

        # Coverage — how much of patch has valid predictions
        valid_mask = pred_patch != NODATA
        coverage   = valid_mask.sum() / (PATCH_SIZE * PATCH_SIZE)
        if coverage < MIN_VALID_COVERAGE:
            continue

        # Must have ≥ MIN_CLASSES unique GT classes
        gt_valid   = gt_patch[valid_mask]
        unique_cls = np.unique(gt_valid)
        if len(unique_cls) < MIN_CLASSES:
            continue

        # Must contain at least one wetland class (1–5)
        wetland_cls = unique_cls[(unique_cls >= 1) & (unique_cls <= 5)]
        if len(wetland_cls) < MIN_WETLAND_CLASSES:
            continue

        acc = (pred_patch[valid_mask] == gt_valid).mean()
        candidates.append((acc, r0, c0))

print(f"  Valid candidate patches: {len(candidates)}")
if len(candidates) < 2:
    raise RuntimeError("Not enough valid patches found. Try reducing MIN_VALID_COVERAGE or PATCH_SIZE.")

candidates.sort(key=lambda x: x[0])
worst_acc, worst_r, worst_c = candidates[0]
best_acc,  best_r,  best_c  = candidates[-1]

print(f"  Best  patch  @ row={best_r},  col={best_c}  — accuracy={best_acc*100:.1f}%")
print(f"  Worst patch  @ row={worst_r}, col={worst_c} — accuracy={worst_acc*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Extract patches
# ─────────────────────────────────────────────────────────────────────────────
def extract_patch(arr, r0, c0):
    return arr[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]

best_gt   = extract_patch(ground_truth, best_r,  best_c)
best_pred = extract_patch(predictions,  best_r,  best_c)
worst_gt  = extract_patch(ground_truth, worst_r, worst_c)
worst_pred= extract_patch(predictions,  worst_r, worst_c)

# Absolute row in full raster (for annotation)
best_abs_row  = best_r  + TEST_ROW_MIN
worst_abs_row = worst_r + TEST_ROW_MIN

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Figure
# ─────────────────────────────────────────────────────────────────────────────
print("\nRendering figure...")

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
fig.patch.set_facecolor('white')

panel_data = [
    (axes[0, 0], labels_to_rgb(best_gt),   "Ground Truth"),
    (axes[0, 1], labels_to_rgb(best_pred), "RF Prediction"),
    (axes[1, 0], labels_to_rgb(worst_gt),  "Ground Truth"),
    (axes[1, 1], labels_to_rgb(worst_pred),"RF Prediction"),
]

for ax, rgb, title in panel_data:
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
    ax.axis('off')

# Row labels on the left side
for ax, label, acc in [
    (axes[0, 0], "Well-classified region", best_acc),
    (axes[1, 0], "Poorly-classified region", worst_acc),
]:
    ax.set_ylabel(
        f"{label}\n(patch accuracy: {acc*100:.1f}%)",
        fontsize=10, labelpad=8, color='#2c3e50',
        fontweight='bold'
    )
    ax.yaxis.set_label_position('left')
    ax.yaxis.label.set_visible(True)

# Legend
legend_patches = [
    mpatches.Patch(facecolor=CLASS_COLORS[c], edgecolor='#888888', linewidth=0.5,
                   label=f"{c}: {CLASS_NAMES[c]}")
    for c in range(6)
]
fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=3,
    fontsize=9,
    bbox_to_anchor=(0.5, 0.01),
    framealpha=0.9,
    edgecolor='#CCCCCC',
)

fig.suptitle(
    "RF v2 — Ground Truth vs Prediction: Test Band Patch Comparison\n"
    f"(Test region: rows {TEST_ROW_MIN}–{TEST_ROW_MAX}, patch size: {PATCH_SIZE}×{PATCH_SIZE} px)",
    fontsize=13, fontweight='bold', y=0.98
)

plt.tight_layout(rect=[0, 0.10, 1, 0.96])
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✅  Saved: {OUTPUT_PATH}")
print("Done.")

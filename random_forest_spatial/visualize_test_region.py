"""
Visualize Test Region: Ground Truth vs RF Predictions
======================================================
Loads the spatially-held-out test tiles, runs the RF model on them,
loads the matching ground-truth labels, and produces a 3-panel figure:

  Panel 1: Ground truth (from bow_river_wetlands_10m_final.tif)
  Panel 2: RF predictions
  Panel 3: Correct/incorrect overlay (green = correct, red = wrong)

Usage:
    python visualize_test_region.py <embeddings_dir>
    python visualize_test_region.py <embeddings_dir> --model path/to/model.pkl --output my_fig.png

Requires:
    wetland_dataset_spatial_split.npz  (for test_row_min)
    rf_wetland_model_spatial_*.pkl     (trained spatial model)
    data_preprocessing/bow_river_wetlands_10m_final.tif
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent

DEFAULT_NPZ_PATH    = REPO_ROOT / 'wetland_dataset_spatial_split.npz'
DEFAULT_LABELS_PATH = REPO_ROOT / 'data_preprocessing' / 'bow_river_wetlands_10m_final.tif'
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / 'test_region_comparison.png'

# Auto-find the newest spatial model in the random_forest folder
def _find_latest_spatial_model():
    pattern = str(SCRIPT_DIR / 'rf_wetland_model_spatial_*.pkl')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None

CLASS_NAMES = {
    0: 'Background/Upland',
    1: 'Marsh',
    2: 'Swamp',
    3: 'Shallow Water',
    4: 'Fen',
    5: 'Bog',
}

# Colormap: one distinct colour per class (Background kept neutral grey)
CLASS_COLORS = np.array([
    [0.75, 0.75, 0.75],   # 0 Background  – grey
    [0.20, 0.60, 0.20],   # 1 Marsh       – mid green
    [0.00, 0.39, 0.00],   # 2 Swamp       – dark green
    [0.25, 0.55, 0.90],   # 3 Shallow Water – sky blue
    [0.80, 0.50, 0.10],   # 4 Fen         – amber
    [0.55, 0.25, 0.65],   # 5 Bog         – purple
], dtype=np.float32)


def labels_to_rgb(label_array):
    """Convert a 2-D class array to an RGB image using CLASS_COLORS."""
    h, w = label_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in enumerate(CLASS_COLORS):
        mask = label_array == cls
        rgb[mask] = color
    return rgb


def find_tiles_in_test_region(embeddings_dir: Path, test_row_min: int):
    """Return tile paths whose row offset >= test_row_min."""
    patterns = [
        "bow_river_embeddings_2020_CORRECTED*.tif",
        "bow_river_embeddings_2020_matched*.tif",
        "bow_river_embeddings_*.tif",
        "*.tif",
    ]
    tile_files = []
    for pat in patterns:
        found = sorted(embeddings_dir.glob(pat))
        if found:
            tile_files = found
            break

    test_tiles = []
    for tf in tile_files:
        parts = tf.stem.split('-')
        if len(parts) >= 3:
            try:
                row_off = int(parts[-2])
                if row_off >= test_row_min:
                    test_tiles.append(tf)
            except ValueError:
                pass
    return test_tiles


def run(embeddings_dir, model_path, npz_path, labels_path, output_path):
    print("=" * 60)
    print("TEST REGION VISUALIZER")
    print("=" * 60)

    # ── 1. Load metadata from npz ───────────────────────────────
    print("\n1. Loading split metadata...")
    npz = np.load(npz_path)
    test_row_min = int(npz['test_row_min'])
    npz.close()
    print(f"   Test region: rows >= {test_row_min}")

    # ── 2. Load model ───────────────────────────────────────────
    print(f"\n2. Loading model: {model_path}")
    rf_model = joblib.load(model_path)
    print(f"   {rf_model.n_estimators} trees, {rf_model.n_features_in_} features")

    # ── 3. Read raster dimensions from labels ───────────────────
    print(f"\n3. Reading label raster dimensions...")
    with rasterio.open(labels_path) as src:
        full_height = src.height
        full_width  = src.width
        out_transform = src.transform
        out_crs = src.crs
    print(f"   Full raster: {full_height} x {full_width}")

    # Determine exact pixel rows for the test region
    test_height = full_height - test_row_min
    if test_height <= 0:
        print(f"   ERROR: test_row_min ({test_row_min}) >= raster height ({full_height})")
        sys.exit(1)
    print(f"   Test region: rows {test_row_min}–{full_height-1}  ({test_height} rows)")

    # ── 4. Load ground truth for test region ────────────────────
    print(f"\n4. Reading ground truth labels for test region...")
    with rasterio.open(labels_path) as src:
        gt_window = Window(0, test_row_min, full_width, test_height)
        ground_truth = src.read(1, window=gt_window).astype(np.int16)
    print(f"   Shape: {ground_truth.shape}")

    # ── 5. Find test tiles ───────────────────────────────────────
    print(f"\n5. Finding test tiles in: {embeddings_dir}")
    test_tiles = find_tiles_in_test_region(Path(embeddings_dir), test_row_min)
    print(f"   Found {len(test_tiles)} test tiles")
    if not test_tiles:
        print("   ERROR: No test tiles found. Check embeddings_dir and test_row_min.")
        sys.exit(1)

    # ── 6. Run inference tile by tile ───────────────────────────
    print(f"\n6. Running inference on test region...")
    NODATA = 255
    predictions = np.full((test_height, full_width), NODATA, dtype=np.uint8)

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

            # Row range relative to full raster
            abs_row_start = tile_row_off
            abs_row_end   = tile_row_off + tile_h

            # Clip to test region [test_row_min, full_height)
            clip_row_start = max(abs_row_start, test_row_min)
            clip_row_end   = min(abs_row_end,   full_height)
            if clip_row_start >= clip_row_end:
                continue

            # Window inside the tile
            tile_local_row_start = clip_row_start - abs_row_start
            tile_local_row_end   = clip_row_end   - abs_row_start
            valid_h = tile_local_row_end - tile_local_row_start
            valid_w = min(tile_w, full_width - tile_col_off)

            tile_data = tile_src.read(
                window=Window(0, tile_local_row_start, valid_w, valid_h)
            )  # (64, valid_h, valid_w)

            n_pixels = valid_h * valid_w
            pixels = tile_data.reshape(64, n_pixels).T  # (n_pixels, 64)
            valid_mask = ~np.isnan(pixels).any(axis=1)

            preds_flat = np.full(n_pixels, NODATA, dtype=np.uint8)
            if valid_mask.any():
                preds_flat[valid_mask] = rf_model.predict(pixels[valid_mask]).astype(np.uint8)

            pred_2d = preds_flat.reshape(valid_h, valid_w)

            # Place into output array (row coords relative to test_row_min)
            out_row_start = clip_row_start - test_row_min
            out_row_end   = out_row_start + valid_h
            col_start     = tile_col_off
            col_end       = col_start + valid_w

            predictions[out_row_start:out_row_end, col_start:col_end] = pred_2d
        print(f"   ✓ {tile_path.name}")

    # ── 7. Build accuracy overlay ────────────────────────────────
    print(f"\n7. Computing accuracy...")
    valid_mask_2d = predictions != NODATA
    n_valid = valid_mask_2d.sum()

    if n_valid == 0:
        print("   ERROR: No valid predictions found. Tiles may not cover test region.")
        sys.exit(1)

    correct = (predictions[valid_mask_2d] == ground_truth[valid_mask_2d])
    acc = correct.mean()
    print(f"   Pixels evaluated: {n_valid:,}")
    print(f"   Overall accuracy: {acc*100:.2f}%")

    # Per-class accuracy
    print("\n   Per-class accuracy:")
    classes_seen = np.unique(ground_truth[valid_mask_2d])
    for cls in classes_seen:
        cls_mask = valid_mask_2d & (ground_truth == cls)
        if cls_mask.sum() == 0:
            continue
        cls_acc = (predictions[cls_mask] == cls).mean()
        print(f"     Class {cls} ({CLASS_NAMES.get(cls, '?'):20s}): {cls_acc*100:.1f}%  (n={cls_mask.sum():,})")

    # ── 8. Build RGB images ──────────────────────────────────────
    print(f"\n8. Rendering panels...")

    gt_rgb   = labels_to_rgb(ground_truth)
    pred_rgb = labels_to_rgb(predictions)

    # Overlay: green = correct, red = wrong, grey = nodata
    overlay = np.full((*predictions.shape, 3), 0.6, dtype=np.float32)  # grey base
    correct_px = valid_mask_2d & (predictions == ground_truth)
    wrong_px   = valid_mask_2d & (predictions != ground_truth)
    overlay[correct_px] = [0.10, 0.80, 0.10]   # green
    overlay[wrong_px]   = [0.90, 0.15, 0.15]   # red

    # ── 9. Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Spatial Holdout Test Region — Overall Accuracy: {acc*100:.1f}%\n'
        f'(rows ≥ {test_row_min}, geographically unseen during training)',
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(gt_rgb)
    axes[0].set_title('Ground Truth', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(pred_rgb)
    axes[1].set_title('RF Predictions', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Correct (green) / Wrong (red)', fontsize=11)
    axes[2].axis('off')

    # Legend for class colours (shared by panels 1 & 2)
    legend_patches = [
        mpatches.Patch(facecolor=CLASS_COLORS[c], label=f'{c}: {CLASS_NAMES[c]}')
        for c in range(6)
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center', ncol=6,
        fontsize=8, bbox_to_anchor=(0.5, -0.02),
        framealpha=0.8
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   Saved: {output_path}")
    print(f"{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize test region: ground truth vs RF predictions')
    parser.add_argument('embeddings_dir', help='Directory containing embedding GeoTIFF tiles')
    parser.add_argument('--model', '-m', default=None,
                        help='Path to trained spatial RF model .pkl (auto-detected if omitted)')
    parser.add_argument('--npz', default=str(DEFAULT_NPZ_PATH),
                        help=f'Path to wetland_dataset_spatial_split.npz (default: {DEFAULT_NPZ_PATH})')
    parser.add_argument('--labels', '-l', default=str(DEFAULT_LABELS_PATH),
                        help=f'Path to labels raster (default: {DEFAULT_LABELS_PATH})')
    parser.add_argument('--output', '-o', default=str(DEFAULT_OUTPUT_PATH),
                        help=f'Output PNG path (default: {DEFAULT_OUTPUT_PATH})')
    args = parser.parse_args()

    model_path = args.model or _find_latest_spatial_model()
    if not model_path:
        print("ERROR: No spatial RF model found. Run model_rf_spatial.py first, or pass --model.")
        sys.exit(1)

    run(
        embeddings_dir=args.embeddings_dir,
        model_path=model_path,
        npz_path=args.npz,
        labels_path=args.labels,
        output_path=args.output,
    )

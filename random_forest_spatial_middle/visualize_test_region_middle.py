"""
Visualize Test Region: Ground Truth vs RF Predictions (Middle Row Band)
========================================================================
Loads the middle-band test tiles, runs the RF model on them,
loads ground-truth labels, and produces a 3-panel figure:

  Panel 1: Ground truth (from bow_river_wetlands_10m_final.tif)
  Panel 2: RF predictions
  Panel 3: Correct/incorrect overlay (green = correct, red = wrong)

Usage:
    python visualize_test_region_middle.py <embeddings_dir>
    python visualize_test_region_middle.py <embeddings_dir> --model path/to/model.pkl --output fig.png

Requires:
    wetland_dataset_middle_split.npz  (for test_row_min / test_row_max)
    rf_wetland_model_middle_*.pkl     (trained middle-band model)
    rf_scaler_middle_*.pkl            (StandardScaler from training)
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

DEFAULT_NPZ_PATH    = SCRIPT_DIR / 'wetland_dataset_middle_split.npz'
DEFAULT_LABELS_PATH = REPO_ROOT / 'data_preprocessing' / 'bow_river_wetlands_10m_final.tif'
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / 'test_region_comparison_middle.png'


def _find_latest_middle_model():
    pattern = str(SCRIPT_DIR / 'rf_wetland_model_middle_*.pkl')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _find_latest_scaler():
    pattern = str(SCRIPT_DIR / 'rf_scaler_middle_*.pkl')
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

CLASS_COLORS = np.array([
    [0.75, 0.75, 0.75],   # 0 Background  – grey
    [0.20, 0.60, 0.20],   # 1 Marsh       – mid green
    [0.00, 0.39, 0.00],   # 2 Swamp       – dark green
    [0.25, 0.55, 0.90],   # 3 Shallow Water – sky blue
    [0.80, 0.50, 0.10],   # 4 Fen         – amber
    [0.55, 0.25, 0.65],   # 5 Bog         – purple
], dtype=np.float32)


def labels_to_rgb(label_array):
    h, w = label_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in enumerate(CLASS_COLORS):
        rgb[label_array == cls] = color
    return rgb


def find_tiles_in_band(embeddings_dir: Path, test_row_min: int, test_row_max: int):
    """Return tile paths whose row offset falls in [test_row_min, test_row_max]."""
    tile_files = sorted(embeddings_dir.glob('*.tif'))
    test_tiles = []
    for tf in tile_files:
        parts = tf.stem.split('-')
        if len(parts) >= 3:
            try:
                row_off = int(parts[-2])
                if test_row_min <= row_off <= test_row_max:
                    test_tiles.append(tf)
            except ValueError:
                pass
    return test_tiles


def run(embeddings_dir, model_path, scaler_path, npz_path, labels_path, output_path):
    print("=" * 60)
    print("MIDDLE BAND TEST REGION VISUALIZER")
    print("=" * 60)

    # ── 1. Load metadata from npz ───────────────────────────────
    print("\n1. Loading split metadata...")
    npz = np.load(npz_path)
    test_row_min = int(npz['test_row_min'])
    test_row_max = int(npz['test_row_max'])
    npz.close()
    print(f"   Test region: rows {test_row_min}–{test_row_max}")

    # ── 2. Load model + scaler ──────────────────────────────────
    print(f"\n2. Loading model: {model_path}")
    rf_model = joblib.load(model_path)
    print(f"   {rf_model.n_estimators} trees, {rf_model.n_features_in_} features")

    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        print(f"   Loading scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)

    # ── 3. Read raster dimensions ───────────────────────────────
    print(f"\n3. Reading label raster dimensions...")
    with rasterio.open(labels_path) as src:
        full_height = src.height
        full_width  = src.width
    print(f"   Full raster: {full_height} x {full_width}")

    # Determine the exact row slice for the test band
    band_height = test_row_max - test_row_min
    print(f"   Test band:  rows {test_row_min}–{test_row_max}  ({band_height} rows)")

    # ── 4. Load ground truth for the test band ──────────────────
    print(f"\n4. Reading ground truth labels for test band...")
    with rasterio.open(labels_path) as src:
        gt_window = Window(0, test_row_min, full_width, band_height)
        ground_truth = src.read(1, window=gt_window).astype(np.int16)
    print(f"   Shape: {ground_truth.shape}")

    # ── 5. Find test tiles ───────────────────────────────────────
    print(f"\n5. Finding test tiles in: {embeddings_dir}")
    test_tiles = find_tiles_in_band(Path(embeddings_dir), test_row_min, test_row_max)
    print(f"   Found {len(test_tiles)} test tiles")
    if not test_tiles:
        print("   ERROR: No test tiles found. Check embeddings_dir.")
        sys.exit(1)

    # ── 6. Run inference tile by tile ───────────────────────────
    print(f"\n6. Running inference on test band...")
    NODATA = 255
    predictions = np.full((band_height, full_width), NODATA, dtype=np.uint8)

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

            # Clip to test band [test_row_min, test_row_max+tile_h)
            clip_row_start = max(abs_row_start, test_row_min)
            clip_row_end   = min(abs_row_end,   test_row_min + band_height)
            if clip_row_start >= clip_row_end:
                continue

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
                X_valid = pixels[valid_mask]
                if scaler is not None:
                    X_valid = scaler.transform(X_valid)
                preds_flat[valid_mask] = rf_model.predict(X_valid).astype(np.uint8)

            pred_2d = preds_flat.reshape(valid_h, valid_w)

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
        print("   ERROR: No valid predictions found.")
        sys.exit(1)

    correct = (predictions[valid_mask_2d] == ground_truth[valid_mask_2d])
    acc = correct.mean()
    print(f"   Pixels evaluated: {n_valid:,}")
    print(f"   Overall accuracy: {acc*100:.2f}%")

    print("\n   Per-class accuracy:")
    for cls in np.unique(ground_truth[valid_mask_2d]):
        cls_mask = valid_mask_2d & (ground_truth == cls)
        if cls_mask.sum() == 0:
            continue
        cls_acc = (predictions[cls_mask] == cls).mean()
        print(f"     Class {cls} ({CLASS_NAMES.get(cls, '?'):20s}): {cls_acc*100:.1f}%  (n={cls_mask.sum():,})")

    # ── 8. Build RGB images ──────────────────────────────────────
    print(f"\n8. Rendering panels...")

    gt_rgb   = labels_to_rgb(ground_truth)
    pred_rgb = labels_to_rgb(predictions)

    overlay = np.full((*predictions.shape, 3), 0.6, dtype=np.float32)
    overlay[valid_mask_2d & (predictions == ground_truth)] = [0.10, 0.80, 0.10]
    overlay[valid_mask_2d & (predictions != ground_truth)] = [0.90, 0.15, 0.15]

    # ── 9. Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Middle Band Spatial Test Region — Overall Accuracy: {acc*100:.1f}%\n'
        f'(rows {test_row_min}–{test_row_max}, geographically unseen during training)',
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(gt_rgb);   axes[0].set_title('Ground Truth', fontsize=11);        axes[0].axis('off')
    axes[1].imshow(pred_rgb); axes[1].set_title('RF Predictions', fontsize=11);      axes[1].axis('off')
    axes[2].imshow(overlay);  axes[2].set_title('Correct (green) / Wrong (red)', fontsize=11); axes[2].axis('off')

    legend_patches = [
        mpatches.Patch(facecolor=CLASS_COLORS[c], label=f'{c}: {CLASS_NAMES[c]}')
        for c in range(6)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.02), framealpha=0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   Saved: {output_path}")
    print(f"{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize middle band test region: ground truth vs RF predictions')
    parser.add_argument('embeddings_dir', help='Directory containing embedding GeoTIFF tiles')
    parser.add_argument('--model', '-m', default=None,
                        help='Path to trained middle RF model .pkl (auto-detected if omitted)')
    parser.add_argument('--scaler', '-s', default=None,
                        help='Path to StandardScaler .pkl (auto-detected if omitted)')
    parser.add_argument('--npz', default=str(DEFAULT_NPZ_PATH),
                        help=f'Path to wetland_dataset_middle_split.npz')
    parser.add_argument('--labels', '-l', default=str(DEFAULT_LABELS_PATH),
                        help='Path to labels raster')
    parser.add_argument('--output', '-o', default=str(DEFAULT_OUTPUT_PATH),
                        help='Output PNG path')
    args = parser.parse_args()

    model_path = args.model or _find_latest_middle_model()
    if not model_path:
        print("ERROR: No middle RF model found. Run model_rf_middle.py first, or pass --model.")
        sys.exit(1)

    scaler_path = args.scaler or _find_latest_scaler()

    run(
        embeddings_dir=args.embeddings_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        npz_path=args.npz,
        labels_path=args.labels,
        output_path=args.output,
    )

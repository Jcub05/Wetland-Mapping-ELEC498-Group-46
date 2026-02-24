"""
Generate Wetland Classification GeoTIFF Map
============================================
Applies the trained Random Forest model to Google Earth Engine satellite
embedding tiles and produces a single-band classified GeoTIFF.

Classes:
    0: Background/Upland
    1: Marsh
    2: Swamp
    3: Shallow Water
    4: Fen
    5: Bog

Usage (local):
    python generate_classification_map.py "C:/path/to/embedding_tiles"

Usage (Colab - see generate_classification_map.ipynb):
    Preferred approach when tiles are on Google Drive.
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import joblib
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ======================================
# CONFIGURATION
# ======================================
# Default paths (override via arguments or modify here)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'rf_wetland_model_v1_20260120_130828.pkl')
DEFAULT_LABELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data_preprocessing', 'bow_river_wetlands_10m_final.tif')
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'bow_river_classification_rf.tif')

CLASS_NAMES = {
    0: 'Background/Upland',
    1: 'Marsh',
    2: 'Swamp',
    3: 'Shallow Water',
    4: 'Fen',
    5: 'Bog',
}
NODATA_VALUE = 255


def find_embedding_tiles(embeddings_dir):
    """Find all embedding GeoTIFF tiles in the given directory."""
    embeddings_path = Path(embeddings_dir)
    
    # Try multiple naming patterns
    patterns = [
        "bow_river_embeddings_2020_CORRECTED*.tif",
        "bow_river_embeddings_2020_matched*.tif",
        "bow_river_embeddings_*.tif",
        "*.tif",
    ]
    
    for pattern in patterns:
        tiles = sorted(embeddings_path.glob(pattern))
        if tiles:
            print(f"  Found {len(tiles)} tiles matching '{pattern}'")
            return tiles
    
    return []


def parse_tile_offset(tile_path):
    """
    Parse row/col offset from tile filename.
    Expected format: *-RRRRRRRRRR-CCCCCCCCCC.tif
    """
    parts = tile_path.stem.split('-')
    if len(parts) >= 3:
        try:
            row_offset = int(parts[-2])
            col_offset = int(parts[-1])
            return row_offset, col_offset
        except ValueError:
            pass
    return None, None


def generate_classification_map(embeddings_dir, model_path, labels_path, output_path):
    """
    Main function: apply RF model to embedding tiles and create classification GeoTIFF.
    """
    print("=" * 60)
    print("WETLAND CLASSIFICATION MAP GENERATOR")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ------------------------------------------
    # 1. Load the trained Random Forest model
    # ------------------------------------------
    print(f"\n1. Loading RF model...")
    print(f"   Path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"   ERROR: Model file not found at {model_path}")
        sys.exit(1)
    
    rf_model = joblib.load(model_path)
    print(f"   Loaded: {rf_model.n_estimators} trees, {rf_model.n_features_in_} features")
    
    # ------------------------------------------
    # 2. Read spatial metadata from labels raster
    # ------------------------------------------
    print(f"\n2. Reading spatial reference from labels raster...")
    print(f"   Path: {labels_path}")
    
    if not os.path.exists(labels_path):
        print(f"   ERROR: Labels raster not found at {labels_path}")
        sys.exit(1)
    
    with rasterio.open(labels_path) as labels_src:
        out_height = labels_src.height
        out_width = labels_src.width
        out_crs = labels_src.crs
        out_transform = labels_src.transform
        print(f"   Dimensions: {out_height} x {out_width}")
        print(f"   CRS: {out_crs}")
        print(f"   Resolution: {out_transform[0]:.1f}m")
    
    # ------------------------------------------
    # 3. Find embedding tiles
    # ------------------------------------------
    print(f"\n3. Finding embedding tiles...")
    print(f"   Directory: {embeddings_dir}")
    
    tile_files = find_embedding_tiles(embeddings_dir)
    
    if not tile_files:
        print(f"   ERROR: No embedding tiles found in {embeddings_dir}")
        sys.exit(1)
    
    # Verify first tile has 64 bands
    with rasterio.open(tile_files[0]) as test_src:
        n_bands = test_src.count
        print(f"   Bands per tile: {n_bands}")
        if n_bands != 64:
            print(f"   WARNING: Expected 64 bands, got {n_bands}")
    
    # ------------------------------------------
    # 4. Create output GeoTIFF
    # ------------------------------------------
    print(f"\n4. Creating output GeoTIFF...")
    print(f"   Path: {output_path}")
    
    out_profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': out_width,
        'height': out_height,
        'count': 1,
        'crs': out_crs,
        'transform': out_transform,
        'nodata': NODATA_VALUE,
        'compress': 'lzw',          # Good compression for classified data
        'tiled': True,              # Tiled for efficient random access
        'blockxsize': 512,
        'blockysize': 512,
    }
    
    # Initialize output with nodata
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        # Fill with nodata value
        nodata_block = np.full((out_height, out_width), NODATA_VALUE, dtype=np.uint8)
        dst.write(nodata_block, 1)
    
    print(f"   Initialized {out_height}x{out_width} raster with nodata={NODATA_VALUE}")
    
    # ------------------------------------------
    # 5. Process each embedding tile
    # ------------------------------------------
    print(f"\n5. Running inference on {len(tile_files)} tiles...")
    print(f"   {'='*50}")
    
    total_pixels_classified = 0
    total_pixels_nodata = 0
    class_counts = np.zeros(6, dtype=np.int64)
    skipped_tiles = []
    
    with rasterio.open(output_path, 'r+') as dst:
        for tile_idx, tile_file in enumerate(tile_files):
            tile_name = tile_file.name
            progress = f"[{tile_idx + 1}/{len(tile_files)}]"
            
            # Parse tile position
            row_offset, col_offset = parse_tile_offset(tile_file)
            if row_offset is None:
                print(f"   {progress} SKIP {tile_name} (can't parse offset)")
                skipped_tiles.append(tile_name)
                continue
            
            try:
                with rasterio.open(tile_file) as tile_src:
                    # Verify band count
                    if tile_src.count != 64:
                        print(f"   {progress} SKIP {tile_name} ({tile_src.count} bands, expected 64)")
                        skipped_tiles.append(tile_name)
                        continue
                    
                    tile_h = tile_src.height
                    tile_w = tile_src.width
                    
                    # Clip to output bounds
                    valid_h = min(tile_h, out_height - row_offset)
                    valid_w = min(tile_w, out_width - col_offset)
                    
                    if valid_h <= 0 or valid_w <= 0:
                        print(f"   {progress} SKIP {tile_name} (outside bounds)")
                        skipped_tiles.append(tile_name)
                        continue
                    
                    # Read tile data: shape (64, valid_h, valid_w)
                    tile_data = tile_src.read(
                        window=Window(0, 0, valid_w, valid_h)
                    )
                    
                    # Reshape to (n_pixels, 64) for prediction
                    n_pixels = valid_h * valid_w
                    pixels = tile_data.reshape(64, n_pixels).T  # (n_pixels, 64)
                    
                    # Create mask for valid (non-NaN) pixels
                    valid_mask = ~np.isnan(pixels).any(axis=1)
                    n_valid = valid_mask.sum()
                    n_nan = n_pixels - n_valid
                    
                    # Predict on valid pixels only
                    predictions = np.full(n_pixels, NODATA_VALUE, dtype=np.uint8)
                    
                    if n_valid > 0:
                        predictions[valid_mask] = rf_model.predict(pixels[valid_mask]).astype(np.uint8)
                    
                    # Reshape back to 2D
                    pred_2d = predictions.reshape(valid_h, valid_w)
                    
                    # Write to output at correct position
                    write_window = Window(col_offset, row_offset, valid_w, valid_h)
                    dst.write(pred_2d, 1, window=write_window)
                    
                    # Update stats
                    total_pixels_classified += n_valid
                    total_pixels_nodata += n_nan
                    for cls in range(6):
                        class_counts[cls] += np.sum(predictions[valid_mask] == cls)
                    
                    print(f"   {progress} ✓ {tile_name} | {valid_h}x{valid_w} | "
                          f"{n_valid:,} classified, {n_nan:,} nodata")
            
            except Exception as e:
                print(f"   {progress} ERROR {tile_name}: {e}")
                skipped_tiles.append(tile_name)
    
    # ------------------------------------------
    # 6. Summary
    # ------------------------------------------
    print(f"\n{'='*60}")
    print("CLASSIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_path}")
    print(f"  Dimensions: {out_height} x {out_width}")
    print(f"  CRS: {out_crs}")
    print(f"  Resolution: {out_transform[0]:.1f}m")
    
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  File size: {file_size_mb:.1f} MB")
    
    print(f"\n  Pixels classified: {total_pixels_classified:,}")
    print(f"  Pixels nodata:     {total_pixels_nodata:,}")
    
    total_classified = class_counts.sum()
    if total_classified > 0:
        print(f"\n  Class distribution:")
        for cls in range(6):
            pct = 100 * class_counts[cls] / total_classified
            print(f"    Class {cls} ({CLASS_NAMES[cls]:20s}): {class_counts[cls]:>12,} ({pct:5.2f}%)")
    
    if skipped_tiles:
        print(f"\n  ⚠ Skipped {len(skipped_tiles)} tiles:")
        for t in skipped_tiles:
            print(f"    - {t}")
    
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nYou can now:")
    print(f"  1. Open '{os.path.basename(output_path)}' in QGIS to visualize")
    print(f"  2. Hand off to frontend for web map display")
    print(f"  3. Convert to Cloud Optimized GeoTIFF (COG) for web serving")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate wetland classification GeoTIFF from trained RF model'
    )
    parser.add_argument(
        'embeddings_dir',
        help='Path to directory containing embedding GeoTIFF tiles'
    )
    parser.add_argument(
        '--model', '-m',
        default=DEFAULT_MODEL_PATH,
        help=f'Path to trained RF model .pkl (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--labels', '-l',
        default=DEFAULT_LABELS_PATH,
        help=f'Path to labels raster for spatial reference (default: {DEFAULT_LABELS_PATH})'
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT_PATH,
        help=f'Path for output classification GeoTIFF (default: {DEFAULT_OUTPUT_PATH})'
    )
    
    args = parser.parse_args()
    
    generate_classification_map(
        embeddings_dir=args.embeddings_dir,
        model_path=args.model,
        labels_path=args.labels,
        output_path=args.output,
    )

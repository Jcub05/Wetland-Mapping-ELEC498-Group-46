"""
predict.py — Core inference engine for the Wetland Mapping backend.

Loads a trained sklearn model, runs prediction over all valid pixels
in the Bow River embedding tiles, and writes results to a GeoTIFF.

Usage (called automatically by app.py on first request):
    from predict import run_inference
    stats = run_inference()   # Returns dict with class_distribution etc.
"""o

import os
import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import joblib
import rasterio
from rasterio.enums import ColorInterp

import config

logger = logging.getLogger(__name__)

# Module-level cache — inference only runs once per server session
_cached_stats: dict | None = None


def run_inference(force: bool = False) -> dict:
    """
    Run inference over all valid pixels and write a GeoTIFF.
    Results are cached so this is only expensive on the first call.

    Returns
    -------
    dict with keys:
        total_samples       int   — number of valid pixels classified
        class_distribution  dict  — {class_id_str: pixel_count}
        confidence          float — overall model accuracy from metadata (or None)
        model_type          str   — model class name
        geotiff_ready       bool  — True if predictions.tif was written successfully
    """
    global _cached_stats
    if _cached_stats is not None and not force:
        logger.info("Returning cached inference results")
        return _cached_stats

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model_path = config.find_model_path()
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    model_type = type(model).__name__

    # ── 2. Load model metadata (for accuracy / confidence) ────────────────────
    confidence = None
    metadata_path = config.find_metadata_path(model_path)
    if metadata_path:
        with open(metadata_path) as f:
            meta = json.load(f)
        confidence = meta.get('overall_metrics', {}).get('accuracy')
        logger.info(f"Model accuracy from metadata: {confidence}")

    # ── 3. Open label raster → get spatial reference for output GeoTIFF ───────
    logger.info(f"Opening label raster: {config.LABEL_RASTER_PATH}")
    with rasterio.open(config.LABEL_RASTER_PATH) as label_src:
        labels_full = label_src.read(1)           # shape: (rows, cols)
        crs = label_src.crs
        transform = label_src.transform
        raster_shape = labels_full.shape           # (rows, cols)

    # ── 4. Build valid-pixel mask ──────────────────────────────────────────────
    valid_mask = (
        (labels_full >= config.VALID_CLASS_MIN) &
        (labels_full <= config.VALID_CLASS_MAX)
    )
    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)
    logger.info(f"Valid pixels to classify: {n_valid:,}")

    # Pre-allocate prediction output (nodata = NODATA_VALUE)
    predictions = np.full(raster_shape, config.NODATA_VALUE, dtype=np.uint8)

    # ── 5. Iterate over embedding tiles ───────────────────────────────────────
    tile_dir = Path(config.EMBEDDINGS_DIR)
    tile_files = sorted(tile_dir.glob('*.tif'))
    if not tile_files:
        raise FileNotFoundError(f"No embedding tiles found in: {config.EMBEDDINGS_DIR}")
    logger.info(f"Found {len(tile_files)} embedding tile(s)")

    total_classified = 0
    for tile_path in tile_files:
        with rasterio.open(tile_path) as tile_src:
            # Parse row/col offset from filename:
            # filename format: ...-RRRRRRRRRR-CCCCCCCCCC.tif
            parts = tile_path.stem.split('-')
            if len(parts) == 3:
                try:
                    tile_row_off = int(parts[1])
                    tile_col_off = int(parts[2])
                except ValueError:
                    logger.warning(f"Could not parse offsets from {tile_path.name}, skipping")
                    continue
            else:
                logger.warning(f"Unexpected tile filename format: {tile_path.name}, skipping")
                continue

            tile_h = tile_src.height
            tile_w = tile_src.width
            row_end = tile_row_off + tile_h
            col_end = tile_col_off + tile_w

            # Find valid-pixel indices that fall inside this tile
            in_tile = (
                (valid_rows >= tile_row_off) & (valid_rows < row_end) &
                (valid_cols >= tile_col_off) & (valid_cols < col_end)
            )
            if not in_tile.any():
                continue

            # Local pixel coordinates within the tile
            local_r = valid_rows[in_tile] - tile_row_off
            local_c = valid_cols[in_tile] - tile_col_off

            # Read the full tile (64 bands × H × W) — much faster than per-pixel reads
            tile_data = tile_src.read()  # shape: (64, h, w)

            # Extract one 64-d embedding per pixel
            X = tile_data[:, local_r, local_c].T  # shape: (n_pixels, 64)

            # Predict
            preds = model.predict(X).astype(np.uint8)

            # Write back into the global prediction array
            predictions[valid_rows[in_tile], valid_cols[in_tile]] = preds
            total_classified += len(preds)

        logger.info(f"  {tile_path.name}: classified {in_tile.sum():,} pixels  (total so far: {total_classified:,})")

    logger.info(f"Inference complete. Total classified: {total_classified:,} / {n_valid:,} valid pixels")

    # ── 6. Write predictions to GeoTIFF ───────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    geotiff_ready = False
    try:
        _write_geotiff(predictions, crs, transform, raster_shape)
        geotiff_ready = True
        logger.info(f"GeoTIFF written to: {config.PREDICTIONS_TIFF}")
    except Exception as e:
        logger.error(f"Failed to write GeoTIFF: {e}")

    # ── 7. Compute class distribution over classified pixels ───────────────────
    classified_pixels = predictions[predictions != config.NODATA_VALUE]
    counts = Counter(int(v) for v in classified_pixels)
    class_distribution = {str(k): counts.get(k, 0) for k in config.WETLAND_CLASSES}

    _cached_stats = {
        'total_samples': int(total_classified),
        'class_distribution': class_distribution,
        'confidence': float(confidence) if confidence is not None else None,
        'model_type': model_type,
        'geotiff_ready': geotiff_ready,
    }
    return _cached_stats


def _write_geotiff(
    predictions: np.ndarray,
    crs,
    transform,
    shape: tuple,
) -> None:
    """Write the uint8 prediction array to a single-band GeoTIFF."""
    rows, cols = shape
    with rasterio.open(
        config.PREDICTIONS_TIFF,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        nodata=config.NODATA_VALUE,
        compress='lzw',          # lossless compression — keeps file small
        tiled=True,              # tiled layout for faster partial reads
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(predictions, 1)

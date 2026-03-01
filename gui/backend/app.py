"""
app.py — Static Flask backend server for the Wetland Mapping GUI.

Endpoints:
  GET /api/health    — liveness check
  GET /api/results   — JSON stats (class distribution from the pre-computed GeoTIFF)
  GET /api/geotiff   — streams the pre-computed GeoTIFF for the Leaflet map

Run with:
  python app.py

This backend expects the model to be run offline, producing a GeoTIFF.
It serves that pre-computed GeoTIFF directly to the frontend.
"""

import logging
import os
from collections import Counter

from flask import Flask, jsonify, send_file, abort
from flask_cors import CORS
import rasterio
import numpy as np

import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow browser frontend to reach this server

# Module-level cache for stats
_cached_stats = None

def _get_stats():
    global _cached_stats
    if _cached_stats is not None:
        return _cached_stats

    logger.info(f"Computing stats from GeoTIFF: {config.GEOTIFF_PATH}")
    if not os.path.exists(config.GEOTIFF_PATH):
        raise FileNotFoundError(f"GeoTIFF not found: {config.GEOTIFF_PATH}")
        
    with rasterio.open(config.GEOTIFF_PATH) as src:
        data = src.read(1)
        
    valid_mask = (data >= config.VALID_CLASS_MIN) & (data <= config.VALID_CLASS_MAX)
    valid_pixels = data[valid_mask]
    counts = Counter(int(v) for v in valid_pixels)
    class_distribution = {str(k): counts.get(k, 0) for k in config.WETLAND_CLASSES}
    total = int(valid_pixels.size)
    
    _cached_stats = {
        'total_samples': total,
        'class_distribution': class_distribution,
        'confidence': None,     
        'model_type': 'Pre-computed GeoTIFF',
        'geotiff_ready': True,
    }
    return _cached_stats

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    """Simple liveness check."""
    return jsonify({'status': 'ok'})


@app.route('/api/results')
def results():
    """Return classification statistics as JSON."""
    try:
        stats = _get_stats()
        return jsonify(stats)
    except FileNotFoundError as e:
        logger.error(str(e))
        abort(404, description=str(e))
    except Exception as e:
        logger.exception("Failed to get stats")
        abort(500, description=str(e))


@app.route('/api/geotiff')
def geotiff():
    """Stream the pre-computed predictions GeoTIFF file."""
    if not os.path.exists(config.GEOTIFF_PATH):
        abort(404, description=f'{os.path.basename(config.GEOTIFF_PATH)} not found on disk.')

    return send_file(
        config.GEOTIFF_PATH,
        mimetype='image/tiff',
        as_attachment=False,
        download_name=os.path.basename(config.GEOTIFF_PATH),
    )


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': str(e)}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Wetland Mapping Backend (Static GeoTIFF Mode)")
    logger.info("=" * 60)
    logger.info(f"Serving GeoTIFF: {config.GEOTIFF_PATH}")
    logger.info("=" * 60)
    logger.info("Starting server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

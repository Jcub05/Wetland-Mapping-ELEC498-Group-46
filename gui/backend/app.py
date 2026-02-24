"""
app.py — Flask backend server for the Wetland Mapping GUI.

Endpoints:
  GET /api/health    — liveness check
  GET /api/results   — JSON stats (class distribution, accuracy, etc.)
  GET /api/geotiff   — streams the predictions GeoTIFF for the Leaflet map

Run with:
  python app.py

The first request to /api/results triggers inference (can take several minutes
depending on dataset size). Subsequent calls are served from an in-memory cache.
"""

import logging
import os

from flask import Flask, jsonify, send_file, abort
from flask_cors import CORS

import config
import predict

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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    """Simple liveness check."""
    return jsonify({'status': 'ok'})


@app.route('/api/results')
def results():
    """
    Return classification statistics as JSON.
    Triggers inference on the first call; subsequent calls are instant.

    Response shape:
    {
        "total_samples":      int,
        "class_distribution": {"0": int, "1": int, ..., "5": int},
        "confidence":         float | null,
        "model_type":         str,
        "geotiff_ready":      bool
    }
    """
    try:
        stats = predict.run_inference()
        return jsonify(stats)
    except FileNotFoundError as e:
        logger.error(str(e))
        abort(404, description=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        abort(500, description=str(e))


@app.route('/api/geotiff')
def geotiff():
    """
    Stream the predictions GeoTIFF file.
    The frontend fetches this as an ArrayBuffer and renders it via
    georaster-layer-for-leaflet on the Leaflet map.
    """
    # Ensure inference has run (no-op if already cached)
    try:
        stats = predict.run_inference()
    except Exception as e:
        logger.exception("Inference failed before serving GeoTIFF")
        abort(500, description=str(e))

    if not stats.get('geotiff_ready'):
        abort(503, description='GeoTIFF is not available — inference may have failed.')

    if not os.path.exists(config.PREDICTIONS_TIFF):
        abort(404, description='predictions.tif not found on disk.')

    return send_file(
        config.PREDICTIONS_TIFF,
        mimetype='image/tiff',
        as_attachment=False,
        download_name='predictions.tif',
    )


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': str(e)}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': str(e)}), 500


@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({'error': str(e)}), 503


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Wetland Mapping Backend")
    logger.info("=" * 60)
    logger.info(f"Model search paths: {config._MODEL_GLOBS}")
    logger.info(f"Label raster:       {config.LABEL_RASTER_PATH}")
    logger.info(f"Embeddings dir:     {config.EMBEDDINGS_DIR}")
    logger.info(f"Output GeoTIFF:     {config.PREDICTIONS_TIFF}")
    logger.info("=" * 60)
    logger.info("Starting server on http://localhost:5000")
    logger.info("Inference will run on the first request to /api/results")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)

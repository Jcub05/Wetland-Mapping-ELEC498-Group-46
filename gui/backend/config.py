"""
config.py — Central configuration for the Wetland Mapping backend.

Update the paths in this file to match your local directory structure.
All other backend files import from here, so nothing else needs to change.
"""

import os
import glob

# ── Directory of this file ────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))

# ── Input data ────────────────────────────────────────────────────────────────
# Path to the full-resolution label raster (used to get CRS, transform, shape)
LABEL_RASTER_PATH = os.path.join(
    _REPO_ROOT, 'bow_river_wetlands_10m_final.tif'
)

# Directory containing the 64-band GEE embedding tiles
# Each tile is named: bow_river_embeddings_2020_matched-RRRR-CCCC.tif
EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, 'Google_Dataset')

# ── Trained model ─────────────────────────────────────────────────────────────
# Auto-detect the most recent .pkl file across both model directories.
# Priority order for glob patterns — first match wins.
_MODEL_GLOBS = [
    os.path.join(_REPO_ROOT, 'random_forest', '*.pkl'),
    os.path.join(_REPO_ROOT, 'SVM', '*.pkl'),
]

def find_model_path() -> str:
    """Return the path to the most recently modified .pkl model file."""
    candidates = []
    for pattern in _MODEL_GLOBS:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No trained model (.pkl) found. Searched:\n" +
            "\n".join(_MODEL_GLOBS)
        )
    # Pick the most recently modified file
    return max(candidates, key=os.path.getmtime)


def find_metadata_path(model_path: str) -> str:
    """
    Given a model .pkl path, return the companion _metadata.json path.
    Returns None if no matching metadata file exists.
    """
    base = model_path.replace('.pkl', '_metadata.json')
    return base if os.path.exists(base) else None


# ── Output GeoTIFF ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(_HERE, 'output')
PREDICTIONS_TIFF = os.path.join(OUTPUT_DIR, 'predictions.tif')

# ── Wetland class definitions ─────────────────────────────────────────────────
# Must match CONFIG.WETLAND_CLASSES in frontend/app.js
WETLAND_CLASSES = {
    0: 'Background',
    1: 'Marsh',
    2: 'Swamp',
    3: 'Fen',
    4: 'Bog',
    5: 'Open Water',
}

# Valid class label range (pixels outside this range are treated as nodata)
VALID_CLASS_MIN = 0
VALID_CLASS_MAX = 5
NODATA_VALUE = 255  # Written to the GeoTIFF for invalid/background pixels

"""
config.py — Configuration for the Wetland Mapping backend.

Update GEOTIFF_PATH to point at your pre-computed predictions GeoTIFF.
Nothing else should need to change.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))

# ── Pre-computed GeoTIFF ──────────────────────────────────────────────────────
# Point this at whatever GeoTIFF you want to display (label raster for demo,
# model output for the final version — same backend either way).
#CHANGE THIS DEPENDING ON WHAT MACHINE WE'RE ON
GEOTIFF_PATH = r"C:\Users\gavin\ELEC498\bow_river_wetlands_10m_final.tif"

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

VALID_CLASS_MIN = 0
VALID_CLASS_MAX = 5
NODATA_VALUE = 255

"""
Bow River Basin -- Wetland Classification Poster Visualization
==============================================================
Overlays the wetland label (or model classification) GeoTIFF on top of an
ESRI World Imagery satellite basemap and exports a poster-quality figure.

Classes:
    0: Background         (rendered transparent -- satellite shows through)
    1: Fen (Graminoid)
    2: Fen (Woody)
    3: Marsh
    4: Shallow Open Water
    5: Swamp

Usage:
    # Default: visualizes the ground-truth label TIF
    python visualize_wetlands.py

    # Visualize a model classification output instead
    python visualize_wetlands.py --tif path/to/classification.tif

    # Full options
    python visualize_wetlands.py --tif path/to/file.tif --output my_map.png --dpi 300 --figsize 18 12
"""

import os
import sys
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import array_bounds
import contextily as ctx

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
CLASS_INFO = {
    0: {"name": "Background",         "color": "#8B7355", "show": False},  # transparent
    1: {"name": "Fen (Graminoid)",    "color": "#FFD600", "show": True},
    2: {"name": "Fen (Woody)",        "color": "#FF9800", "show": True},
    3: {"name": "Marsh",              "color": "#4CAF50", "show": True},
    4: {"name": "Shallow Open Water", "color": "#29B6F6", "show": True},
    5: {"name": "Swamp",              "color": "#1B5E20", "show": True},
}

NODATA_VALUE = 255
OVERLAY_ALPHA = 0.8   # opacity of wetland classes over the satellite
MAX_PIXELS    = 10000   # max raster dimension -- downsamples if larger

TARGET_CRS = "EPSG:3857"  # Web Mercator required by contextily

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_reproject(tif_path, max_pixels=MAX_PIXELS):
    """
    Load a single-band label GeoTIFF, downsample if needed, and reproject
    to EPSG:3857 (Web Mercator) for contextily compatibility.

    Returns
    -------
    data   : np.ndarray (H x W) uint8, class labels
    bounds : (west, south, east, north) in EPSG:3857
    """
    with rasterio.open(tif_path) as src:
        src_height, src_width = src.height, src.width
        src_crs   = src.crs
        src_nodata = src.nodata if src.nodata is not None else NODATA_VALUE

        # -- Compute output shape (downsample to max_pixels on longest axis) --
        scale = max(src_height, src_width) / max_pixels
        if scale > 1:
            read_height = int(src_height / scale)
            read_width  = int(src_width  / scale)
        else:
            read_height, read_width = src_height, src_width

        print(f"  Source  : {src_width} x {src_height} px  CRS={src_crs.to_string()}")
        print(f"  Reading : {read_width} x {read_height} px  (scale 1/{scale:.1f})")

        data = src.read(
            1,
            out_shape=(read_height, read_width),
            resampling=Resampling.nearest,
        )

        # Scaled affine transform
        read_transform = src.transform * src.transform.scale(
            src_width  / read_width,
            src_height / read_height,
        )

    # -- Reproject to EPSG:3857 if needed --
    target_crs_obj = CRS.from_string(TARGET_CRS)
    already_3857   = (src_crs.to_epsg() == 3857)

    if already_3857:
        bounds = array_bounds(read_height, read_width, read_transform)
        return data, bounds

    src_bounds = array_bounds(read_height, read_width, read_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs_obj,
        read_width, read_height,
        left=src_bounds[0], bottom=src_bounds[1],
        right=src_bounds[2], top=src_bounds[3],
    )

    dst_data = np.full((dst_height, dst_width), NODATA_VALUE, dtype=np.uint8)
    reproject(
        source=data,
        destination=dst_data,
        src_transform=read_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs_obj,
        resampling=Resampling.nearest,
        src_nodata=NODATA_VALUE,
        dst_nodata=NODATA_VALUE,
    )

    bounds = array_bounds(dst_height, dst_width, dst_transform)
    print(f"  Output  : {dst_width} x {dst_height} px  CRS=EPSG:3857")
    return dst_data, bounds


def build_rgba(data, alpha=OVERLAY_ALPHA):
    """
    Convert a class label array to an RGBA float image.
    Background (class 0) and nodata are fully transparent.
    """
    h, w  = data.shape
    rgba  = np.zeros((h, w, 4), dtype=np.float32)

    for cls_id, info in CLASS_INFO.items():
        if not info["show"]:
            continue
        mask  = data == cls_id
        color = mcolors.to_rgba(info["color"])
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = alpha

    rgba[data == NODATA_VALUE, 3] = 0.0
    return rgba


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize(tif_path, output_path, title, figsize, dpi, alpha, max_pixels):
    print(f"\nLoading: {tif_path}")
    data, bounds = load_and_reproject(tif_path, max_pixels=max_pixels)

    west, south, east, north = bounds
    print(f"  Bounds  : W={west:.1f}  S={south:.1f}  E={east:.1f}  N={north:.1f}")

    rgba = build_rgba(data, alpha=alpha)

    # -- Figure setup --
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")

    # Set extent so contextily knows which region to tile
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")

    # -- Satellite basemap (drawn first, behind overlay) --
    print("Fetching ESRI World Imagery basemap...")
    try:
        ctx.add_basemap(
            ax,
            crs=TARGET_CRS,
            source=ctx.providers.Esri.WorldImagery,
            zoom="auto",
            attribution=False,
            zorder=1,
        )
    except Exception as e:
        print(f"  WARNING: Could not fetch basemap ({e}). Continuing without it.")

    # -- Wetland overlay --
    ax.imshow(
        rgba,
        extent=[west, east, south, north],
        origin="upper",
        interpolation="nearest",
        zorder=2,
        aspect="auto",
    )

    # Reset limits (contextily can shift them slightly)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    # -- Study area boundary --
    from matplotlib.patches import Rectangle
    border_rect = Rectangle(
        (west, south),
        east - west,
        north - south,
        linewidth=2.0,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(border_rect)

    # -- Legend --
    legend_patches = [
        mpatches.Patch(
            facecolor=info["color"],
            edgecolor="white",
            linewidth=0.6,
            label=info["name"],
        )
        for cls_id, info in CLASS_INFO.items()
        if info["show"]
    ]
    legend = ax.legend(
        handles=legend_patches,
        loc="lower right",
        framealpha=0.82,
        facecolor="#1a1a1a",
        edgecolor="white",
        fontsize=12,
        title="Wetland Class",
        title_fontsize=13,
        labelcolor="white",
    )
    legend.get_title().set_color("white")

    # -- Title --
    ax.set_title(title, fontsize=17, fontweight="bold", color="white", pad=14)

    # -- Basemap attribution (required for ESRI imagery use) --
    ax.text(
        0.01, 0.01,
        "Basemap: Esri, DigitalGlobe, GeoEye, Earthstar Geographics",
        transform=ax.transAxes,
        fontsize=6,
        color="white",
        alpha=0.7,
        zorder=10,
    )

    ax.set_axis_off()
    plt.tight_layout(pad=0.5)

    print(f"\nSaving to: {output_path}  (dpi={dpi})")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    print("Done.")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_TIF  = os.path.join(THIS_DIR, "..", "data_preprocessing",
                                "bow_river_wetlands_10m_final.tif")
    DEFAULT_OUT  = os.path.join(THIS_DIR, "bow_river_wetlands_poster.png")

    parser = argparse.ArgumentParser(
        description="Visualize Bow River Basin wetland labels over satellite basemap"
    )
    parser.add_argument(
        "--tif", default=DEFAULT_TIF,
        help="Path to classification/label GeoTIFF (default: ground-truth labels)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUT,
        help="Output PNG path (default: bow_river_wetlands_poster.png in this folder)"
    )
    parser.add_argument(
        "--title", default="Bow River Basin — Wetland Classification",
        help="Figure title"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output resolution in DPI (default: 300)"
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[16, 10],
        metavar=("W", "H"),
        help="Figure dimensions in inches (default: 16 10)"
    )
    parser.add_argument(
        "--alpha", type=float, default=OVERLAY_ALPHA,
        help=f"Wetland overlay opacity 0-1 (default: {OVERLAY_ALPHA})"
    )
    parser.add_argument(
        "--max-pixels", type=int, default=MAX_PIXELS,
        help=f"Max raster dimension for reading (default: {MAX_PIXELS})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.tif):
        print(f"ERROR: TIF not found at:\n  {args.tif}")
        print("Make sure bow_river_wetlands_10m_final.tif is in data_preprocessing/")
        sys.exit(1)

    visualize(
        tif_path   = args.tif,
        output_path= args.output,
        title      = args.title,
        figsize    = tuple(args.figsize),
        dpi        = args.dpi,
        alpha      = args.alpha,
        max_pixels = args.max_pixels,
    )

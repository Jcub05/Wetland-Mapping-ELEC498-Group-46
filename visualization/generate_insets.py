"""
Bow River Basin -- Wetland Inset / Detail Views
================================================
Scans the label GeoTIFF to find small geographic windows that contain multiple
wetland classes, then renders each one at full resolution over a high-zoom
satellite basemap.  Each inset is saved as a separate PNG.

The insets are intended to sit alongside the basin-wide overview map on the
poster to demonstrate the classification quality at ground level.

Usage:
    # Auto-find the N most class-diverse windows (default N=4)
    python generate_insets.py

    # More control
    python generate_insets.py --tif path/to/labels.tif --n 6 --km 8 --zoom 14 --dpi 300
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
from rasterio.windows import Window
import contextily as ctx
from pyproj import Transformer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Shared class definitions (keep in sync with visualize_wetlands.py)
# ---------------------------------------------------------------------------
CLASS_INFO = {
    0: {"name": "Background",         "color": "#8B7355", "show": False},
    1: {"name": "Fen (Graminoid)",    "color": "#FFD600", "show": True},
    2: {"name": "Fen (Woody)",        "color": "#FF9800", "show": True},
    3: {"name": "Marsh",              "color": "#0D4669", "show": True},
    4: {"name": "Shallow Open Water", "color": "#29B6F6", "show": True},
    5: {"name": "Swamp",              "color": "#45D44F", "show": True},
}
WETLAND_CLASSES = [c for c in CLASS_INFO if CLASS_INFO[c]["show"]]
NODATA_VALUE    = 255
OVERLAY_ALPHA   = 0.6
TARGET_CRS      = "EPSG:3857"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_rgba(data, alpha=OVERLAY_ALPHA):
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
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


def diversity_score(window_data):
    """
    Score a data window by wetland class diversity.
    Returns (n_unique_wetland_classes, total_wetland_pixels).
    """
    classes_present = set()
    total_wetland   = 0
    for c in WETLAND_CLASSES:
        count = np.sum(window_data == c)
        if count > 0:
            classes_present.add(c)
            total_wetland += count
    return len(classes_present), total_wetland


def find_diverse_windows(tif_path, n_windows, window_km, scan_stride_km=10):
    """
    Scan the TIF at a coarse stride, score each window by wetland diversity,
    and return the top-N windows as pixel (row_off, col_off, height, width)
    quads in the source CRS pixel space.
    """
    with rasterio.open(tif_path) as src:
        pixel_size_m = abs(src.transform.a)       # metres per pixel (x)
        window_px    = int((window_km * 1000) / pixel_size_m)
        stride_px    = int((scan_stride_km * 1000) / pixel_size_m)
        H, W         = src.height, src.width

        print(f"  TIF  : {W} x {H} px  ({pixel_size_m:.1f} m/px)")
        print(f"  Window: {window_km} km = {window_px} px")
        print(f"  Stride: {scan_stride_km} km = {stride_px} px")

        candidates = []
        rows = range(0, H - window_px, stride_px)
        cols = range(0, W - window_px, stride_px)
        total = len(rows) * len(cols)
        print(f"  Scanning {total} candidate windows...")

        for ri, row in enumerate(rows):
            for col in cols:
                win  = Window(col, row, window_px, window_px)
                data = src.read(1, window=win)
                n_cls, n_wet = diversity_score(data)
                if n_cls >= 2:                   # must have at least 2 wetland types
                    candidates.append((n_cls, n_wet, row, col, window_px))

            if ri % 5 == 0:
                done = (ri + 1) * len(cols)
                print(f"    {done}/{total} scanned, {len(candidates)} candidates so far...")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print(f"  Found {len(candidates)} windows with ≥2 wetland classes; taking top {n_windows}")

    # Deduplicate: skip any window that overlaps with an already-selected one
    selected = []
    for cand in candidates:
        n_cls, n_wet, row, col, wp = cand
        overlap = False
        for _, _, sr, sc, swp in selected:
            if abs(row - sr) < wp and abs(col - sc) < wp:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
        if len(selected) == n_windows:
            break

    return selected   # list of (n_cls, n_wet, row_off, col_off, window_px)


def read_window_and_reproject(tif_path, row_off, col_off, win_px):
    """
    Read a pixel window from the TIF at full resolution and reproject to
    EPSG:3857.  Returns (data, bounds_3857).
    """
    with rasterio.open(tif_path) as src:
        window    = Window(col_off, row_off, win_px, win_px)
        data      = src.read(1, window=window)
        win_transform = src.window_transform(window)
        src_crs   = src.crs
        h, w      = data.shape

    target_crs_obj = CRS.from_string(TARGET_CRS)
    src_bounds     = array_bounds(h, w, win_transform)

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, target_crs_obj, w, h,
        left=src_bounds[0], bottom=src_bounds[1],
        right=src_bounds[2], top=src_bounds[3],
    )
    dst_data = np.full((dst_h, dst_w), NODATA_VALUE, dtype=np.uint8)
    reproject(
        source=data, destination=dst_data,
        src_transform=win_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=target_crs_obj,
        resampling=Resampling.nearest,
        src_nodata=NODATA_VALUE, dst_nodata=NODATA_VALUE,
    )
    bounds = array_bounds(dst_h, dst_w, dst_transform)
    return dst_data, bounds


def render_inset(tif_path, row_off, col_off, win_px, output_path,
                 inset_idx, zoom, dpi, figsize, alpha):
    print(f"\n  Rendering inset {inset_idx} ...")
    data, bounds = read_window_and_reproject(tif_path, row_off, col_off, win_px)
    west, south, east, north = bounds

    classes_here = sorted({c for c in WETLAND_CLASSES if np.any(data == c)})
    class_names  = [CLASS_INFO[c]["name"] for c in classes_here]
    subtitle     = "  |  ".join(class_names)

    rgba = build_rgba(data, alpha=alpha)

    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")

    try:
        ctx.add_basemap(
            ax, crs=TARGET_CRS,
            source=ctx.providers.Esri.WorldImagery,
            zoom=zoom,
            attribution=False,
            zorder=1,
        )
    except Exception as e:
        print(f"    WARNING: basemap fetch failed ({e})")

    ax.imshow(
        rgba,
        extent=[west, east, south, north],
        origin="upper",
        interpolation="nearest",
        zorder=2,
        aspect="auto",
    )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    legend_patches = [
        mpatches.Patch(
            facecolor=CLASS_INFO[c]["color"],
            edgecolor="white",
            linewidth=0.6,
            label=CLASS_INFO[c]["name"],
        )
        for c in classes_here
    ]
    legend = ax.legend(
        handles=legend_patches,
        loc="lower right",
        framealpha=0.82,
        facecolor="#1a1a1a",
        edgecolor="white",
        fontsize=10,
        labelcolor="white",
    )

    ax.set_title(
        f"Detail View {inset_idx}  —  {subtitle}",
        fontsize=13, fontweight="bold", color="white", pad=10,
    )
    ax.text(
        0.01, 0.01,
        "Basemap: Esri, DigitalGlobe, GeoEye, Earthstar Geographics",
        transform=ax.transAxes, fontsize=5, color="white", alpha=0.6, zorder=10,
    )
    ax.set_axis_off()
    plt.tight_layout(pad=0.4)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"    Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_TIF = os.path.join(THIS_DIR, "..", "data_preprocessing",
                               "bow_river_wetlands_10m_final.tif")

    parser = argparse.ArgumentParser(
        description="Auto-find diverse wetland windows and render high-res inset images"
    )
    parser.add_argument("--tif",    default=DEFAULT_TIF, help="Path to label GeoTIFF")
    parser.add_argument("--n",      type=int,   default=4,   help="Number of insets (default: 4)")
    parser.add_argument("--km",     type=float, default=8.0, help="Inset side length in km (default: 8)")
    parser.add_argument("--stride", type=float, default=10.0,help="Scan stride in km (default: 10)")
    parser.add_argument("--zoom",   type=int,   default=14,  help="Basemap zoom level (default: 14)")
    parser.add_argument("--dpi",    type=int,   default=300, help="Output DPI (default: 300)")
    parser.add_argument("--figsize",nargs=2, type=float, default=[8, 8], metavar=("W","H"),
                        help="Figure size in inches (default: 8 8)")
    parser.add_argument("--alpha",  type=float, default=OVERLAY_ALPHA,
                        help=f"Overlay opacity (default: {OVERLAY_ALPHA})")
    parser.add_argument("--outdir", default=THIS_DIR, help="Output directory (default: visualization/)")
    args = parser.parse_args()

    if not os.path.exists(args.tif):
        print(f"ERROR: TIF not found at:\n  {args.tif}")
        sys.exit(1)

    print(f"\nScanning TIF for diverse windows: {args.tif}")
    windows = find_diverse_windows(args.tif, args.n, args.km, args.stride)

    if not windows:
        print("No windows with ≥2 wetland classes found. Try reducing --km or --stride.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    for i, (n_cls, n_wet, row_off, col_off, win_px) in enumerate(windows, start=1):
        out_path = os.path.join(args.outdir, f"inset_{i:02d}.png")
        print(f"\n[{i}/{len(windows)}] classes={n_cls}, wetland_px={n_wet:,}")
        render_inset(
            tif_path   = args.tif,
            row_off    = row_off,
            col_off    = col_off,
            win_px     = win_px,
            output_path= out_path,
            inset_idx  = i,
            zoom       = args.zoom,
            dpi        = args.dpi,
            figsize    = tuple(args.figsize),
            alpha      = args.alpha,
        )

    print(f"\nAll done. {len(windows)} insets saved to: {args.outdir}")

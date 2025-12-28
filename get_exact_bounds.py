import rasterio
from rasterio.warp import transform_bounds

# Get exact bounds from labels raster
with rasterio.open('bow_river_wetlands_10m_final.tif') as src:
    bounds_utm = src.bounds
    crs = src.crs
    
    # Transform to lat/lon for Earth Engine
    bounds_latlon = transform_bounds(crs, 'EPSG:4326', *bounds_utm)
    WEST, SOUTH, EAST, NORTH = bounds_latlon
    
    print("="*60)
    print("EXACT BOW RIVER BASIN BOUNDS")
    print("="*60)
    print(f"\nPython format:")
    print(f"WEST = {WEST}")
    print(f"SOUTH = {SOUTH}")
    print(f"EAST = {EAST}")
    print(f"NORTH = {NORTH}")
    
    print(f"\n\nJavaScript format (for Google Earth Engine):")
    print(f"var bowRiver = ee.Geometry.Rectangle([")
    print(f"  {WEST}, {SOUTH},  // West, South")
    print(f"  {EAST}, {NORTH}   // East, North")
    print(f"]);")
    
    print(f"\n\nDimensions: {src.width} x {src.height}")
    print(f"CRS: {crs}")

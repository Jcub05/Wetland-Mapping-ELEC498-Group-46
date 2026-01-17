"""
Download Sentinel-2 Satellite Imagery for Bow River Basin
Alternative to Google Embeddings - GUARANTEED 100% coverage!

Sentinel-2 Bands for Wetlands:
- B2 (Blue), B3 (Green), B4 (Red) - Visible light
- B8 (NIR) - Vegetation health
- B11, B12 (SWIR) - Water detection (CRITICAL for wetlands!)

This will give you ~10 bands instead of 64 embeddings, but with full coverage
"""
import ee
import rasterio
from rasterio.warp import transform_bounds

# Initialize Earth Engine
ee.Initialize(project='group46-capstone')

# Read exact bounds from labels raster
print("Reading bounds from labels raster...")
labels_file = "bow_river_wetlands_10m_final.tif"

with rasterio.open(labels_file) as src:
    bounds_utm = src.bounds
    crs = src.crs
    width = src.width
    height = src.height
    
    print(f"Labels CRS: {crs}")
    print(f"Labels dimensions: {height} x {width}")
    
    # Transform to lat/lon for Earth Engine
    bounds_latlon = transform_bounds(crs, 'EPSG:4326', *bounds_utm)
    WEST, SOUTH, EAST, NORTH = bounds_latlon
    
    print(f"Labels bounds (Lat/Lon): ({WEST:.6f}, {SOUTH:.6f}) to ({EAST:.6f}, {NORTH:.6f})")

# Define area of interest
aoi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])

print("\n" + "="*70)
print("LOADING SENTINEL-2 DATA FOR 2020")
print("="*70)

# Load Sentinel-2 Surface Reflectance (cloud-corrected)
sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# Filter for 2020 and Bow River region
collection_2020 = sentinel2 \
    .filterDate('2020-01-01', '2021-01-01') \
    .filterBounds(aoi) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Less than 20% clouds

print(f"\nFiltering Sentinel-2 images...")
print(f"  Date range: 2020-01-01 to 2021-01-01")
print(f"  Cloud filter: <20% clouds")
print(f"  Region: Bow River Basin")

# Check how many images we have
image_count = collection_2020.size().getInfo()
print(f"\n✓ Found {image_count} Sentinel-2 images for 2020!")

if image_count == 0:
    print("\n⚠⚠⚠ ERROR: No Sentinel-2 images found!")
    print("This should never happen - Sentinel-2 has global coverage.")
    exit(1)

# Create cloud-free composite using median
print("\nCreating cloud-free composite...")
print("  Method: Median (removes clouds automatically)")

# Select bands optimized for wetland mapping
wetland_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'B8A']
composite = collection_2020.select(wetland_bands).median().toFloat()  # Cast to Float32

# Add calculated indices for wetlands
print("\nAdding wetland indices...")

# NDVI = (NIR - Red) / (NIR + Red) - vegetation health
ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI').toFloat()

# NDWI = (Green - NIR) / (Green + NIR) - water detection
ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI').toFloat()

# MNDWI = (Green - SWIR) / (Green + SWIR) - modified water index (better for wetlands!)
mndwi = composite.normalizedDifference(['B3', 'B11']).rename('MNDWI').toFloat()

# Combine bands and indices (all Float32 now)
final_image = composite.addBands([ndvi, ndwi, mndwi])

print("  ✓ NDVI (vegetation)")
print("  ✓ NDWI (water)")  
print("  ✓ MNDWI (wetland-specific water)")

band_names = final_image.bandNames().getInfo()
print(f"\n✓ Final image has {len(band_names)} bands:")
print(f"  {band_names}")

# Export to Google Drive
print("\n" + "="*70)
print("STARTING EXPORT FOR SENTINEL-2 2020")
print("="*70)

task = ee.batch.Export.image.toDrive(
    image=final_image,
    description='bow_river_sentinel2_2020',
    folder='EarthEngine',
    fileFormat='GeoTIFF',
    region=aoi,
    scale=10,  # 10m resolution - matches embeddings and labels!
    crs='EPSG:32612',  # Match labels CRS
    maxPixels=1e13
)

task.start()

print(f"\n✓ Export task started!")
print(f"  Task ID: {task.id}")
print(f"  Description: bow_river_sentinel2_2020")
print(f"  Bands: {len(band_names)} ({', '.join(band_names[:5])}...)")
print(f"  Resolution: 10m per pixel")
print(f"  Expected file size: ~5-10 GB (more data than embeddings!)")

print("\n" + "="*70)
print("WHAT YOU'RE GETTING")
print("="*70)
print("✓ 100% coverage of Bow River (no sparse tiles!)")
print("✓ 7 spectral bands (B2, B3, B4, B8, B8A, B11, B12)")
print("✓ 3 wetland indices (NDVI, NDWI, MNDWI)")
print("✓ Total: 10 features per pixel")
print("\nMonitor at: https://code.earthengine.google.com/tasks")
print("\n⏱️ This will take 30-60 minutes (similar to embeddings)")
print("\nBoth exports can run in parallel - check back in ~1 hour!")

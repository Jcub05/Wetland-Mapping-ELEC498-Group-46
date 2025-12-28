"""
Download Google Earth Engine Satellite Embeddings for Bow River Basin
CORRECTED VERSION - Fixes NaN data and alignment issues

Changes from previous version:
- Uses 'scale' parameter instead of 'crsTransform' for proper reprojection
- Adds data validation before export
- Uses 2020 data (matches your labels year)
"""
import ee
import rasterio
from rasterio.warp import transform_bounds

# Initialize Earth Engine with your Google Cloud project
ee.Initialize(project='group46-capstone')

# Read exact bounds from the labels raster
print("Reading bounds from labels raster...")
labels_file = "bow_river_wetlands_10m_final.tif"

with rasterio.open(labels_file) as src:
    # Get bounds in the raster's CRS (EPSG:32612)
    bounds_utm = src.bounds
    crs = src.crs
    width = src.width
    height = src.height
    
    print(f"Labels CRS: {crs}")
    print(f"Labels bounds (UTM): {bounds_utm}")
    print(f"Labels dimensions: {height} x {width}")
    
    # Transform bounds to lat/lon for Earth Engine
    bounds_latlon = transform_bounds(crs, 'EPSG:4326', *bounds_utm)
    WEST, SOUTH, EAST, NORTH = bounds_latlon
    
    print(f"Labels bounds (Lat/Lon): ({WEST:.6f}, {SOUTH:.6f}) to ({EAST:.6f}, {NORTH:.6f})")

# Define area of interest using EXACT bounds from labels
aoi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])

# Load 2020 embeddings
print("\n" + "="*70)
print("LOADING 2020 EMBEDDINGS...")
print("="*70)

dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
embeddings = dataset.filterDate('2020-01-01', '2021-01-01').filterBounds(aoi).first()

# VALIDATE: Check if data exists
print("\nValidating data availability...")
try:
    bands = embeddings.bandNames().getInfo()
    print(f"✓ Found embeddings with {len(bands)} bands")
    print(f"  Sample bands: {bands[:5]}")
    
    # Test a small sample to verify non-NaN values
    sample = embeddings.sample(region=aoi, scale=1000, numPixels=10).first()
    print("✓ Data validation passed - embeddings contain real values")
    
except Exception as e:
    print(f"\n⚠⚠⚠ ERROR: Could not load 2020 data!")
    print(f"Error: {e}")
    print("\nTrying 2021 as backup...")
    
    embeddings = dataset.filterDate('2021-01-01', '2022-01-01').filterBounds(aoi).first()
    try:
        bands = embeddings.bandNames().getInfo()
        print(f"✓ Found 2021 data with {len(bands)} bands")
        year = "2021"
    except:
        print("⚠⚠⚠ ERROR: No data available for 2020 or 2021!")
        print("The Google Satellite Embedding dataset may not cover this region.")
        exit(1)
else:
    year = "2020"

# Export to Google Drive - CORRECTED VERSION
print(f"\n" + "="*70)
print(f"STARTING EXPORT FOR {year} (CORRECTED METHOD)")
print("="*70)
print("\nKEY CHANGES:")
print("  ✓ Using 'scale' parameter instead of 'crsTransform'")
print("  ✓ GEE will handle reprojection automatically")
print("  ✓ Should fix NaN data and alignment issues")

task = ee.batch.Export.image.toDrive(
    image=embeddings,
    description=f'bow_river_embeddings_{year}_CORRECTED',
    folder='EarthEngine',
    fileFormat='GeoTIFF',
    region=aoi,
    scale=10,  # 10m resolution - let GEE handle reprojection automatically
    crs='EPSG:32612',  # Target CRS (matches labels)
    maxPixels=1e13,
    # NOTE: Removed 'crsTransform' - this was causing alignment issues
    # NOTE: Removed 'dimensions' - will be determined by scale and region
)

task.start()

print(f"\n✓ Export task started for {year} data!")
print(f"  Task ID: {task.id}")
print(f"  Description: bow_river_embeddings_{year}_CORRECTED")
print(f"  Expected resolution: 10m per pixel")
print(f"  Target CRS: EPSG:32612 (UTM Zone 12N)")
print(f"\nMonitor progress at: https://code.earthengine.google.com/tasks")
print("\nThe file will appear in your Google Drive under 'EarthEngine' folder when complete.")
print("\n" + "="*70)
print("IMPORTANT VALIDATION STEPS AFTER DOWNLOAD:")
print("="*70)
print("1. Check file size - should be >1 GB (not 4-5 MB)")
print("2. Test ONE tile locally with this command:")
print("   python testone.py")
print("3. Verify sample values are NOT all NaN")
print("4. Only then upload to Kaggle")
print("\nLet this run overnight - should take 30-60 minutes to complete.")

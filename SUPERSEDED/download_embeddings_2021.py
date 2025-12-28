"""
Download Google Earth Engine Satellite Embeddings for Bow River Basin
Updated version with error checking and 2021 data (2020 had no coverage)
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

# Try multiple years to find data
print("\nSearching for available data...")
dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

# Try 2021 first (most likely to have data)
print("\nTrying 2021...")
embeddings_2021 = dataset.filterDate('2021-01-01', '2022-01-01').filterBounds(aoi).first()

try:
    # Test if data exists by trying to get band names
    bands = embeddings_2021.bandNames().getInfo()
    print(f"✓ Found 2021 data!")
    print(f"  Bands: {len(bands)} bands")
    embeddings = embeddings_2021
    year = "2021"
except:
    print("✗ No 2021 data, trying 2022...")
    
    # Try 2022 as backup
    embeddings_2022 = dataset.filterDate('2022-01-01', '2023-01-01').filterBounds(aoi).first()
    try:
        bands = embeddings_2022.bandNames().getInfo()
        print(f"✓ Found 2022 data!")
        print(f"  Bands: {len(bands)} bands")
        embeddings = embeddings_2022
        year = "2022"
    except:
        print("\n⚠⚠⚠ ERROR: No data found for 2021 or 2022!")
        print("The Google Satellite Embedding dataset may not cover the Bow River region.")
        print("\nOptions:")
        print("1. Try a different region")
        print("2. Use a different satellite dataset")
        print("3. Contact Google Earth Engine support")
        exit(1)

# Export to Google Drive with EXACT specification
print(f"\nStarting export to Google Drive (using {year} data)...")
print("Using exact bounds and CRS from labels raster...")

task = ee.batch.Export.image.toDrive(
    image=embeddings,
    description=f'bow_river_embeddings_{year}_matched',
    folder='EarthEngine',
    fileFormat='GeoTIFF',  # Explicit format
    region=aoi,
    crs='EPSG:32612',  # Match the labels CRS
    crsTransform=[10, 0, bounds_utm.left, 0, -10, bounds_utm.top],
    dimensions=f'{width}x{height}',
    maxPixels=1e13
)

task.start()
print(f"\n✓ Export task started for {year} data!")
print(f"  Task ID: {task.id}")
print(f"  Expected dimensions: {height} x {width}")
print(f"  Description: bow_river_embeddings_{year}_matched")
print(f"\nMonitor progress at: https://code.earthengine.google.com/tasks")
print("The file will appear in your Google Drive under 'EarthEngine' folder when complete.")
print("\n⚠ Make sure the task completes with 100% and no errors!")
print("⚠ After download, test one tile locally before uploading to Kaggle!")

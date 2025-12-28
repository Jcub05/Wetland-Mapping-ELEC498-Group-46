"""
Download Google Earth Engine Satellite Embeddings for Bow River Basin
This version reads the exact bounds from the labels raster to ensure perfect alignment.
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

# Load embeddings for 2020
print("\nLoading embeddings from Earth Engine...")
dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
embeddings = dataset.filterDate('2020-01-01', '2021-01-01').filterBounds(aoi).first()

print("Embeddings loaded successfully!")
print("\nDataset info:")
print(f"  Bands: {embeddings.bandNames().getInfo()[:5]}... (64 total)")
print(f"  Area: Bow River Basin")

# Export to Google Drive with EXACT specification
print("\nStarting export to Google Drive...")
print("Using exact bounds and CRS from labels raster...")
task = ee.batch.Export.image.toDrive(
    image=embeddings,
    description='bow_river_embeddings_2020_matched',
    folder='EarthEngine',
    region=aoi,
    crs='EPSG:32612',  # Match the labels CRS
    crsTransform=[10, 0, bounds_utm.left, 0, -10, bounds_utm.top],  # Exact transform (includes 10m resolution)
    dimensions=f'{width}x{height}',  # Exact dimensions
    maxPixels=1e13
)
    
task.start()
print(f"\n✓ Export task started!")
print(f"  Task ID: {task.id}")
print(f"  Expected dimensions: {height} x {width}")
print(f"  This will match your labels raster exactly!")
print(f"\nMonitor progress at: https://code.earthengine.google.com/tasks")
print("The file will appear in your Google Drive under 'EarthEngine' folder when complete.")

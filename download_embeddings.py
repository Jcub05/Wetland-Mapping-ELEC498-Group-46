"""
Download Google Earth Engine Satellite Embeddings for Bow River Basin
"""
import ee
import rasterio
import numpy as np

# Initialize Earth Engine with your Google Cloud project
ee.Initialize(project='group46-capstone')

# Your raster bounds
WEST = -115.54
SOUTH = 49.83
EAST = -110.99
NORTH = 51.77

# Define area of interest
aoi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])

# Load embeddings for 2020
print("Loading embeddings from Earth Engine...")
dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
embeddings = dataset.filterDate('2020-01-01', '2021-01-01').filterBounds(aoi).first()

print("Embeddings loaded successfully!")
print("\nDataset info:")
print(f"  Bands: {embeddings.bandNames().getInfo()[:5]}... (64 total)")
print(f"  Area: Bow River Basin ({WEST}, {SOUTH}) to ({EAST}, {NORTH})")

# Export to Google Drive
print("\nStarting export to Google Drive...")
task = ee.batch.Export.image.toDrive(
    image=embeddings,
    description='bow_river_embeddings_2020',
    folder='EarthEngine',
    region=aoi,
    scale=10,  # 10-meter resolution
    crs='EPSG:32612',
    maxPixels=1e13
    )
    
task.start()
print(f"✓ Export task started!")
print(f"  Task ID: {task.id}")
print(f"\nMonitor progress at: https://code.earthengine.google.com/tasks")
print("The file will appear in your Google Drive under 'EarthEngine' folder when complete.")

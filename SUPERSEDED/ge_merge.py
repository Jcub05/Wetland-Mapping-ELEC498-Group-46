import rasterio
from rasterio.merge import merge
import glob
import os

# Find all tile files
tile_pattern = "GoogleEarth_Engine_Download/bow_river_embeddings_2020-*.tif"
tile_files = sorted(glob.glob(tile_pattern))
print(f"Found {len(tile_files)} tiles to merge")

# Check if merged file already exists
output_file = "bow_river_embeddings_2020.tif"
if os.path.exists(output_file):
    print(f"Output file already exists: {output_file}")
    response = input("Overwrite? (y/n): ")
    if response.lower() != 'y':
        print("Merge cancelled.")
        exit(0)

print("\nOpening tiles...")
src_files = [rasterio.open(fp) for fp in tile_files]

print("Merging (this will use significant RAM - ~50GB+)...")
print("If this fails with memory error, you'll need to:")
print("  1. Install GDAL, or")
print("  2. Work with tiles directly in your dataloader, or")
print("  3. Use a machine with more RAM")
print()

try:
    # Merge with less precise float32 to save memory
    mosaic, out_trans = merge(src_files, dtype='float32')
    
    # Get metadata from first tile
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "dtype": "float32",
        "compress": "lzw",
        "tiled": True
    })
    
    print(f"Writing merged raster: {mosaic.shape}")
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    print(f"✓ Success! Created: {output_file}")
    
except MemoryError as e:
    print(f"\n✗ Memory error: {e}")
    print("\nAlternatives:")
    print("1. Install GDAL: conda install gdal")
    print("2. Use VRT file (virtual raster) in dataloader instead")
    print("3. Modify dataloader to read tiles directly")
    
finally:
    for src in src_files:
        src.close()
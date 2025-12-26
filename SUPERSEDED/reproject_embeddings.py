"""
Reproject Google Earth Engine embeddings to match the labels raster
"""
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

print("Loading labels raster to get target properties...")
with rasterio.open("bow_river_wetlands_10m_final.tif") as labels_src:
    target_crs = labels_src.crs
    target_transform = labels_src.transform
    target_width = labels_src.width
    target_height = labels_src.height
    target_bounds = labels_src.bounds
    
    print(f"Target dimensions: {target_height} x {target_width}")
    print(f"Target CRS: {target_crs}")
    print(f"Target bounds: {target_bounds}")

# Open embeddings VRT
print("\nOpening embeddings VRT...")
with rasterio.open("bow_river_embeddings_2020.vrt") as src:
    print(f"Source dimensions: {src.height} x {src.width}")
    print(f"Source CRS: {src.crs}")
    print(f"Source bounds: {src.bounds}")
    
    # Prepare output metadata
    out_meta = src.meta.copy()
    out_meta.update({
        'crs': target_crs,
        'transform': target_transform,
        'width': target_width,
        'height': target_height,
        'compress': 'lzw',
        'tiled': True,
        'dtype': 'float32'  # Use float32 to save space
    })
    
    # Create output file
    output_file = "bow_river_embeddings_2020_reprojected.tif"
    print(f"\nReprojecting to {output_file}...")
    print("This will take several minutes and process bands one at a time...")
    
    with rasterio.open(output_file, 'w', **out_meta) as dst:
        for band_idx in range(1, src.count + 1):
            print(f"  Band {band_idx}/{src.count}...", end='', flush=True)
            
            # Read source band
            source_band = src.read(band_idx)
            
            # Create destination array
            destination = np.zeros((target_height, target_width), dtype=np.float32)
            
            # Reproject
            reproject(
                source=source_band,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            
            # Write to output
            dst.write(destination, band_idx)
            print(" done")
    
    print(f"\n✓ Reprojection complete: {output_file}")
    print("  This file now matches your labels raster exactly!")

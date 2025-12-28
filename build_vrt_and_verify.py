"""
Create VRT from new matched tiles and verify alignment
"""
import rasterio
from rasterio.merge import merge
import glob

# Find all new tiles
tile_pattern = "Google_Dataset/bow_river_embeddings_2020_matched-*.tif"
tile_files = sorted(glob.glob(tile_pattern))
print(f"Found {len(tile_files)} tiles")

# Create VRT using rasterio
print("Building VRT...")
from rasterio import shutil as rio_shutil

# Use gdalbuildvrt-like functionality
vrt_options = {
    'resolution': 'highest'
}

# Build VRT by opening all tiles and getting combined metadata
src_files = [rasterio.open(fp) for fp in tile_files]

# Get bounds
all_bounds = [src.bounds for src in src_files]
min_left = min(b.left for b in all_bounds)
min_bottom = min(b.bottom for b in all_bounds)
max_right = max(b.right for b in all_bounds)
max_top = max(b.top for b in all_bounds)

# Create VRT XML manually (simple approach)
import xml.etree.ElementTree as ET

first_src = src_files[0]
vrt_width = int((max_right - min_left) / first_src.res[0])
vrt_height = int((max_top - min_bottom) / first_src.res[1])

vrt_root = ET.Element('VRTDataset', {
    'rasterXSize': str(vrt_width),
    'rasterYSize': str(vrt_height)
})

# Add SRS
srs = ET.SubElement(vrt_root, 'SRS')
srs.text = first_src.crs.to_wkt()

# Add GeoTransform
geotransform = ET.SubElement(vrt_root, 'GeoTransform')
geotransform.text = f"{min_left}, {first_src.res[0]}, 0, {max_top}, 0, {-first_src.res[1]}"

# Add bands
for band_idx in range(1, first_src.count + 1):
    band = ET.SubElement(vrt_root, 'VRTRasterBand', {
        'dataType': 'Float32',
        'band': str(band_idx)
    })
    
    # Add each tile as a source
    for tile_file in tile_files:
        with rasterio.open(tile_file) as tile:
            simple_source = ET.SubElement(band, 'SimpleSource')
            
            src_filename = ET.SubElement(simple_source, 'SourceFilename', {'relativeToVRT': '1'})
            src_filename.text = tile_file
            
            src_band = ET.SubElement(simple_source, 'SourceBand')
            src_band.text = str(band_idx)

# Write VRT
from xml.dom import minidom
vrt_file = "bow_river_embeddings_2020_matched.vrt"
xml_str = minidom.parseString(ET.tostring(vrt_root)).toprettyxml(indent="  ")
with open(vrt_file, 'w') as f:
    f.write(xml_str)

# Close all source files
for src in src_files:
    src.close()

print(f"✓ Created: {vrt_file}")

# Now verify alignment
print("\n" + "="*60)
print("VERIFYING ALIGNMENT")
print("="*60)

with rasterio.open(vrt_file) as emb:
    with rasterio.open("bow_river_wetlands_10m_final.tif") as lab:
        
        print("\nEmbeddings VRT:")
        print(f"  Dimensions: {emb.height} x {emb.width}")
        print(f"  Bands: {emb.count}")
        print(f"  CRS: {emb.crs}")
        print(f"  Bounds: {emb.bounds}")
        
        print("\nLabels:")
        print(f"  Dimensions: {lab.height} x {lab.width}")
        print(f"  Bands: {lab.count}")
        print(f"  CRS: {lab.crs}")
        print(f"  Bounds: {lab.bounds}")
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        
        # Check dimensions
        dims_match = (emb.height, emb.width) == (lab.height, lab.width)
        print(f"{'✓' if dims_match else '❌'} Dimensions: {emb.height}x{emb.width} vs {lab.height}x{lab.width}")
        
        # Check CRS
        crs_match = emb.crs == lab.crs
        print(f"{'✓' if crs_match else '❌'} CRS: {emb.crs} vs {lab.crs}")
        
        # Check bounds (within 1m tolerance)
        bounds_close = all(abs(a - b) < 1.0 for a, b in zip(emb.bounds, lab.bounds))
        print(f"{'✓' if bounds_close else '❌'} Bounds match (within 1m)")
        
        if dims_match and crs_match and bounds_close:
            print("\n🎉 SUCCESS! Rasters are perfectly aligned!")
            print(f"\nUpdate your dataloader.py to use: '{vrt_file}'")
        else:
            print("\n⚠ WARNING: Rasters may not be perfectly aligned")
            print("You may need to re-download or reproject")

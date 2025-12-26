import rasterio
from rasterio.vrt import WarpedVRT
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Find all tile files
tile_pattern = "GoogleEarth_Engine_Download/bow_river_embeddings_2020-*.tif"
tile_files = sorted(glob.glob(tile_pattern))
print(f"Found {len(tile_files)} tiles")

# Open first file to get metadata
with rasterio.open(tile_files[0]) as src:
    # Get overall bounds by reading all tiles
    print("Reading tile metadata...")
    all_bounds = []
    for tile_file in tile_files:
        with rasterio.open(tile_file) as tile_src:
            all_bounds.append(tile_src.bounds)
    
    # Calculate combined bounds
    min_left = min(b.left for b in all_bounds)
    min_bottom = min(b.bottom for b in all_bounds)
    max_right = max(b.right for b in all_bounds)
    max_top = max(b.top for b in all_bounds)
    
    print(f"Combined bounds: ({min_left}, {min_bottom}, {max_right}, {max_top})")
    
    # Create VRT XML manually
    vrt_root = ET.Element('VRTDataset', {
        'rasterXSize': str(int((max_right - min_left) / src.transform.a)),
        'rasterYSize': str(int((max_top - min_bottom) / abs(src.transform.e)))
    })
    
    # Add SRS
    srs = ET.SubElement(vrt_root, 'SRS')
    srs.text = str(src.crs.to_wkt())
    
    # Add GeoTransform
    geotransform = ET.SubElement(vrt_root, 'GeoTransform')
    geotransform.text = f"{min_left}, {src.transform.a}, 0, {max_top}, 0, {src.transform.e}"
    
    # Add bands
    for band_idx in range(1, src.count + 1):
        band = ET.SubElement(vrt_root, 'VRTRasterBand', {
            'dataType': str(src.dtypes[band_idx - 1]).replace('float', 'Float'),
            'band': str(band_idx)
        })
        
        # Add simple sources for each tile
        for tile_file in tile_files:
            with rasterio.open(tile_file) as tile_src:
                src_filename = ET.SubElement(band, 'SimpleSource')
                filename = ET.SubElement(src_filename, 'SourceFilename', {'relativeToVRT': '1'})
                filename.text = tile_file
                
                source_band = ET.SubElement(src_filename, 'SourceBand')
                source_band.text = str(band_idx)

# Write VRT file
vrt_output = "bow_river_embeddings_2020.vrt"
xml_str = minidom.parseString(ET.tostring(vrt_root)).toprettyxml(indent="  ")
with open(vrt_output, 'w') as f:
    f.write(xml_str)

print(f"✓ Created VRT file: {vrt_output}")
print("\nYou can now use this VRT file in your dataloader:")
print('  with rasterio.open("bow_river_embeddings_2020.vrt") as src:')
print("      embeddings = src.read()")

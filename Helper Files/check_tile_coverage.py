"""
Check which GEE embedding tiles contain valid data vs NaN
This will help you identify which parts of Bow River have coverage
"""
import rasterio
import numpy as np
from pathlib import Path

# Directory containing tiles (update this path)
tiles_dir = Path("./EarthEngine-Download")  # Current directory or update to your tiles folder
tile_files = sorted(tiles_dir.glob("bow_river_embeddings_2020_CORRECTED*.tif"))

print(f"Found {len(tile_files)} tiles\n")
print("="*80)
print("Checking each tile for valid data...")
print("="*80)

valid_tiles = []
nan_tiles = []

for tile_file in tile_files:
    file_size_mb = tile_file.stat().st_size / (1024**2)
    
    try:
        with rasterio.open(tile_file) as src:
            # Sample first band, small window
            data = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
            
            has_data = not np.all(np.isnan(data))
            percent_nan = np.isnan(data).sum() / data.size * 100
            
            status = "✓ VALID" if has_data else "✗ NaN"
            
            if has_data:
                valid_tiles.append(tile_file.name)
            else:
                nan_tiles.append(tile_file.name)
            
            print(f"{status:10} | {file_size_mb:6.1f} MB | {percent_nan:5.1f}% NaN | {tile_file.name}")
    
    except Exception as e:
        print(f"ERROR     | {file_size_mb:6.1f} MB | Error: {e} | {tile_file.name}")

print("\n" + "="*80)
print(f"SUMMARY")
print("="*80)
print(f"Valid tiles (with data):  {len(valid_tiles)}")
print(f"NaN tiles (empty):        {len(nan_tiles)}")
print(f"Coverage:                 {len(valid_tiles)/(len(valid_tiles)+len(nan_tiles))*100:.1f}%")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if len(valid_tiles) > 0:
    print(f"✓ You have {len(valid_tiles)} tiles with valid data!")
    print(f"✓ Train your model using ONLY these tiles")
    print(f"✓ Your model will work for ~{len(valid_tiles)/(len(valid_tiles)+len(nan_tiles))*100:.0f}% of the Bow River region")
    
    if len(nan_tiles) > len(valid_tiles):
        print(f"\n⚠ WARNING: Most of your region ({len(nan_tiles)} tiles) has no embedding coverage")
        print(f"⚠ Consider supplementing with Sentinel-2 for full coverage")
else:
    print("✗ No valid tiles found - all are NaN")
    print("✗ You MUST switch to Sentinel-2 or another dataset")

# Save valid tile list
if len(valid_tiles) > 0:
    with open("valid_tiles_list.txt", "w") as f:
        for tile in valid_tiles:
            f.write(f"{tile}\n")
    print(f"\n✓ Valid tile list saved to: valid_tiles_list.txt")

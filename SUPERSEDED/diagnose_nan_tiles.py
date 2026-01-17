"""
Diagnose NaN contamination in embedding tiles
Checks which tiles have NaN values and their extent
"""
import rasterio
import numpy as np
from pathlib import Path

print("="*70)
print("EMBEDDING TILES NaN DIAGNOSTIC")
print("="*70)

embeddings_dir = Path("EarthEngine-Download")
tile_files = sorted(embeddings_dir.glob("*.tif"))

print(f"\nFound {len(tile_files)} embedding tiles")
print(f"\nScanning for NaN values...\n")

contaminated_tiles = []
clean_tiles = []

for i, tile_file in enumerate(tile_files):
    with rasterio.open(tile_file) as src:
        # Read first band as a sample
        data = src.read(1)  # Shape: (height, width)
        
        has_nan = np.isnan(data).any()
        nan_percentage = 100 * np.isnan(data).sum() / data.size
        
        if has_nan:
            contaminated_tiles.append({
                'file': tile_file.name,
                'nan_pct': nan_percentage,
                'size': data.size,
                'shape': data.shape
            })
        else:
            clean_tiles.append(tile_file.name)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Scanned {i + 1}/{len(tile_files)} tiles...")

print(f"\n{'='*70}")
print("RESULTS")
print("="*70)

print(f"\nClean tiles: {len(clean_tiles)} ({100*len(clean_tiles)/len(tile_files):.1f}%)")
print(f"Contaminated tiles: {len(contaminated_tiles)} ({100*len(contaminated_tiles)/len(tile_files):.1f}%)")

if contaminated_tiles:
    print(f"\n⚠️  CONTAMINATED TILES:")
    for tile in contaminated_tiles[:10]:  # Show first 10
        print(f"   {tile['file']}: {tile['nan_pct']:.2f}% NaN values")
    
    if len(contaminated_tiles) > 10:
        print(f"   ... and {len(contaminated_tiles) - 10} more tiles")

# Save full report
with open('nan_diagnostic_report.txt', 'w') as f:
    f.write("EMBEDDING TILES NaN DIAGNOSTIC REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total tiles: {len(tile_files)}\n")
    f.write(f"Clean tiles: {len(clean_tiles)}\n")
    f.write(f"Contaminated tiles: {len(contaminated_tiles)}\n\n")
    
    if contaminated_tiles:
        f.write("CONTAMINATED TILES:\n")
        for tile in contaminated_tiles:
            f.write(f"  {tile['file']}: {tile['nan_pct']:.2f}% NaN\n")
    
    f.write("\nCLEAN TILES:\n")
    for tile_name in clean_tiles:
        f.write(f"  {tile_name}\n")

print(f"\n✓ Full report saved to nan_diagnostic_report.txt")

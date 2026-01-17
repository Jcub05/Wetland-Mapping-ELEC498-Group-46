import rasterio
import numpy as np
from pathlib import Path

print("Checking for NaN values in local embedding tiles...")

embeddings_dir = Path("EarthEngine-Download")
tile_files = sorted(embeddings_dir.glob("*.tif"))

print(f"Found {len(tile_files)} tiles\n")

# Check first tile as sample
sample_tile = tile_files[0]
print(f"Sampling: {sample_tile.name}")

with rasterio.open(sample_tile) as src:
    # Read first band
    data = src.read(1)
    has_nan = np.isnan(data).any()
    
    if has_nan:
        print(f"❌ CONTAINS NaN VALUES")
    else:
        print(f"✅ NO NaN VALUES - Clean!")
        print(f"   Shape: {data.shape}")
        print(f"   Min: {data.min():.4f}, Max: {data.max():.4f}")

import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Load the Google Earth embeddings from VRT
embeddings_file = "bow_river_embeddings_2020_matched.vrt"
labels_file = "bow_river_wetlands_10m_final.tif"

print("Loading labels...")
with rasterio.open(labels_file) as labels_src:
    labels_full = labels_src.read(1)
    print(f"Labels (original): {labels_full.shape}")

print(f"Opening embeddings VRT: {embeddings_file}")
embeddings_src = rasterio.open(embeddings_file)
print(f"Embeddings: {embeddings_src.count} bands x {embeddings_src.height} x {embeddings_src.width}")

# Crop labels to match embeddings (remove last row - 31,428 pixels)
labels = labels_full[:embeddings_src.height, :embeddings_src.width]
print(f"Labels (cropped): {labels.shape}")

# Verify dimensions now match
assert (embeddings_src.height, embeddings_src.width) == labels.shape, \
    f"Rasters not aligned! Embeddings: {embeddings_src.height}x{embeddings_src.width}, Labels: {labels.shape}"
print("✓ Dimensions match!")

# Find valid pixels (where labels exist)
valid_mask = (labels > 0) & (labels <= 5)
valid_count = valid_mask.sum()
print(f"\nValid pixels: {valid_count:,} out of {labels.size:,} ({100*valid_count/labels.size:.2f}%)")

# Extract embeddings ONLY for valid pixels (memory efficient)
# Read in row chunks for much better performance
print("\nReading embeddings for valid pixels...")
print(f"This will process {valid_count:,} pixels in chunks...")

# Pre-allocate arrays
n_samples = valid_count
X = np.zeros((n_samples, embeddings_src.count), dtype=np.float32)
y = labels[valid_mask]

# Read embeddings in row chunks (much faster than pixel-by-pixel)
chunk_size = 100  # Process 100 rows at a time
sample_idx = 0

for row_start in range(0, embeddings_src.height, chunk_size):
    row_end = min(row_start + chunk_size, embeddings_src.height)
    
    # Read this chunk of rows (all bands, all columns)
    chunk_data = embeddings_src.read(window=((row_start, row_end), (0, embeddings_src.width)))
    # Shape: (64, chunk_rows, width)
    
    # Extract valid pixels from this chunk
    chunk_mask = valid_mask[row_start:row_end, :]
    if chunk_mask.any():
        # Transpose to (chunk_rows, width, 64) for easier indexing
        chunk_transposed = np.transpose(chunk_data, (1, 2, 0))
        
        # Extract valid pixels
        valid_embeddings = chunk_transposed[chunk_mask]
        n_valid = len(valid_embeddings)
        
        # Store in output array
        X[sample_idx:sample_idx + n_valid, :] = valid_embeddings
        sample_idx += n_valid
    
    # Progress update every 1000 rows
    if (row_start + chunk_size) % 1000 == 0:
        progress = min(row_start + chunk_size, embeddings_src.height) / embeddings_src.height * 100
        print(f"  Progress: {progress:.1f}% ({sample_idx:,} / {n_samples:,} pixels)")

embeddings_src.close()

print(f"\n✓ Data loaded successfully!")
print(f"  X shape: {X.shape} ({X.nbytes / (1024**3):.2f} GB in memory)")
print(f"  y shape: {y.shape}")
print(f"  Class distribution: {np.unique(y, return_counts=True)}")

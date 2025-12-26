import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset


# Load the Google Earth embeddings (new download should be perfectly aligned!)
# TODO: Update filename once download completes from Google Drive
embeddings_file = "bow_river_embeddings_2020_matched.tif"  # Update this path
labels_file = "bow_river_wetlands_10m_final.tif"

print("Loading labels...")
with rasterio.open(labels_file) as labels_src:
    labels = labels_src.read(1)
    print(f"Labels: {labels.shape}")

print(f"Opening embeddings: {embeddings_file}")
embeddings_src = rasterio.open(embeddings_file)
print(f"Embeddings: {embeddings_src.count} bands x {embeddings_src.height} x {embeddings_src.width}")

# Check dimensions match
assert (embeddings_src.height, embeddings_src.width) == labels.shape, \
    f"Rasters not aligned! Embeddings: {embeddings_src.height}x{embeddings_src.width}, Labels: {labels.shape}"

# Find valid pixels (where labels exist)
valid_mask = (labels > 0) & (labels <= 5)
valid_count = valid_mask.sum()
print(f"\nValid pixels: {valid_count:,} out of {labels.size:,} ({100*valid_count/labels.size:.2f}%)")

# Extract embeddings ONLY for valid pixels (memory efficient)
# This avoids loading the full 352GB into memory
print("\nReading embeddings for valid pixels only...")
y_indices, x_indices = np.where(valid_mask)

# Read embeddings in batches
batch_size = 10000
n_samples = len(y_indices)
X = np.zeros((n_samples, embeddings_src.count), dtype=np.float32)

for i in range(0, n_samples, batch_size):
    end_i = min(i + batch_size, n_samples)
    batch_y = y_indices[i:end_i]
    batch_x = x_indices[i:end_i]
    
    # Read pixel-by-pixel for this batch
    for j, (y, x) in enumerate(zip(batch_y, batch_x)):
        X[i + j, :] = embeddings_src.read(window=((y, y+1), (x, x+1)))[:, 0, 0]
    
    if (i + batch_size) % 100000 == 0:
        print(f"  Processed {min(i + batch_size, n_samples):,} / {n_samples:,} pixels")

y = labels[valid_mask]
embeddings_src.close()

print(f"\n✓ Data loaded successfully!")
print(f"  X shape: {X.shape} ({X.nbytes / (1024**2):.1f} MB)")
print(f"  y shape: {y.shape}")
print(f"  Class distribution: {np.unique(y, return_counts=True)}")

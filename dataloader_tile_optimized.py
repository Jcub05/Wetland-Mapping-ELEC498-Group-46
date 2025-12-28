import rasterio
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# File paths
labels_file = "/kaggle/input/bo-river-and-google-earth/bow_river_wetlands_10m_final.tif"
embeddings_dir = Path("/kaggle/input/bo-river-and-google-earth/Google_Dataset")

print("="*60)
print("TILE-OPTIMIZED DATALOADER - Fast Version")
print("="*60)

# Load labels
print("\n1. Loading labels...")
with rasterio.open(labels_file) as labels_src:
    labels_full = labels_src.read(1)
    print(f"   Labels shape: {labels_full.shape}")

# Get list of all embedding tiles
tile_files = sorted(embeddings_dir.glob("*.tif"))
print(f"\n2. Found {len(tile_files)} embedding tiles")

# Balanced sampling strategy
samples_per_class = {
    0: 600_000,
    1: 19_225,
    2: 150_000,
    3: 500_000,
    4: 150_000,
    5: 100_000,
}
total_target = sum(samples_per_class.values())
print(f"\n3. Balanced sampling target: {total_target:,} samples")

# Analyze class distribution
valid_mask = (labels_full >= 0) & (labels_full <= 5)
unique_classes, class_counts = np.unique(labels_full[valid_mask], return_counts=True)
print("\n   Class distribution:")
for cls, count in zip(unique_classes, class_counts):
    print(f"     Class {cls}: {count:,} pixels ({100*count/valid_mask.sum():.2f}%)")

# Sample pixel coordinates
print("\n4. Sampling pixel coordinates...")
sampled_indices_y = []
sampled_indices_x = []
sampled_labels = []

for cls in unique_classes:
    class_mask = (labels_full == cls)
    y_idx, x_idx = np.where(class_mask)
    
    n_available = len(y_idx)
    n_target = samples_per_class[cls]
    n_sample = min(n_target, n_available)
    
    if n_available > n_target:
        sample_idx = np.random.choice(n_available, n_target, replace=False)
    else:
        sample_idx = np.arange(n_available)
    
    sampled_indices_y.append(y_idx[sample_idx])
    sampled_indices_x.append(x_idx[sample_idx])
    sampled_labels.append(np.full(n_sample, cls))
    
    print(f"   Class {cls}: sampled {n_sample:,} / {n_available:,}")

# Combine and shuffle
y_indices = np.concatenate(sampled_indices_y)
x_indices = np.concatenate(sampled_indices_x)
y = np.concatenate(sampled_labels)

np.random.seed(42)
shuffle_idx = np.random.permutation(len(y_indices))
y_indices = y_indices[shuffle_idx]
x_indices = x_indices[shuffle_idx]
y = y[shuffle_idx]

print(f"\n   Total samples: {len(y):,}")

# Calculate class weights
unique_sampled, sampled_counts = np.unique(y, return_counts=True)
class_weights = torch.zeros(6)
for cls, count in zip(unique_sampled, sampled_counts):
    class_weights[cls] = 1.0 / count
class_weights = class_weights / class_weights.sum() * 6

print("\n5. Class weights for training:")
for cls in range(6):
    print(f"   Class {cls}: {class_weights[cls]:.4f}")

# OPTIMIZATION: Read tile-by-tile instead of row-by-row
print("\n6. Extracting embeddings (TILE-BY-TILE - FAST!)...")
print(f"   Will process {len(tile_files)} tiles once each\n")

# Pre-allocate output
n_samples = len(y_indices)
X = np.zeros((n_samples, 64), dtype=np.float32)  # 64 bands
found_samples = np.zeros(n_samples, dtype=bool)

# Process each tile
with tqdm(total=len(tile_files), desc="Processing tiles", unit=" tiles") as pbar:
    for tile_file in tile_files:
        # Open tile
        with rasterio.open(tile_file) as tile_src:
            # Get tile bounds in global coordinates
            tile_bounds = tile_src.bounds
            tile_transform = tile_src.transform
            
            # Read entire tile into memory (much faster than per-pixel)
            tile_data = tile_src.read()  # Shape: (64, height, width)
            
            # Get tile position in global raster
            tile_row_offset = int(round((tile_bounds.top - tile_src.bounds.top) / abs(tile_transform[4])))
            tile_col_offset = int(round((tile_bounds.left - tile_src.bounds.left) / tile_transform[0]))
            
            # Actually, let's use the filename to determine position
            # Filename format: bow_river_embeddings_2020_matched-RRRRRRRRRR-CCCCCCCCCC.tif
            parts = tile_file.stem.split('-')
            if len(parts) == 3:
                tile_row_offset = int(parts[1])
                tile_col_offset = int(parts[2])
            else:
                # Fallback to transform if filename parsing fails
                tile_row_offset = 0
                tile_col_offset = 0
            
            # Find which samples fall within this tile
            tile_height, tile_width = tile_src.height, tile_src.width
            
            # Check which sample coordinates are in this tile's bounds
            in_tile_y = (y_indices >= tile_row_offset) & (y_indices < tile_row_offset + tile_height)
            in_tile_x = (x_indices >= tile_col_offset) & (x_indices < tile_col_offset + tile_width)
            in_tile_mask = in_tile_y & in_tile_x
            
            if in_tile_mask.any():
                # Get local coordinates within this tile
                local_y = y_indices[in_tile_mask] - tile_row_offset
                local_x = x_indices[in_tile_mask] - tile_col_offset
                
                # Extract embeddings for these samples
                for i, (ly, lx) in enumerate(zip(local_y, local_x)):
                    global_idx = np.where(in_tile_mask)[0][i]
                    X[global_idx, :] = tile_data[:, ly, lx]
                    found_samples[global_idx] = True
        
        pbar.update(1)
        pbar.set_postfix({"found": f"{found_samples.sum():,}/{n_samples:,}"})

print(f"\n✓ Extracted {found_samples.sum():,} / {n_samples:,} samples")

if not found_samples.all():
    print(f"   ⚠ Warning: {(~found_samples).sum():,} samples not found in tiles")

# Save dataset
output_file = 'wetland_dataset_1.5M.npz'
print(f"\n7. Saving dataset to {output_file}...")
np.savez_compressed(
    output_file,
    X=X[found_samples],
    y=y[found_samples],
    class_weights=class_weights.numpy(),
)

print(f"\n{'='*60}")
print(f"✓ COMPLETE!")
print(f"{'='*60}")
print(f"  Dataset: {output_file}")
print(f"  Samples: {found_samples.sum():,}")
print(f"  Size: {X[found_samples].nbytes / (1024**3):.2f} GB in memory")
print(f"\nUse: nn.CrossEntropyLoss(weight=class_weights)")

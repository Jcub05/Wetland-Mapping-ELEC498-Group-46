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

# Find all labeled pixels (including background class 0)
valid_mask = (labels >= 0) & (labels <= 5)
valid_count = valid_mask.sum()
print(f"\nTotal labeled pixels: {valid_count:,} out of {labels.size:,} ({100*valid_count/labels.size:.2f}%)")

# Analyze class distribution
unique_classes, class_counts = np.unique(labels[valid_mask], return_counts=True)
print("\nClass distribution:")
for cls, count in zip(unique_classes, class_counts):
    print(f"  Class {cls}: {count:,} pixels ({100*count/valid_count:.2f}%)")

# Balanced sampling strategy: ~1.5M samples total
# Allocate based on class size and importance
samples_per_class = {
    0: 600_000,   # Background: plenty available, need good "not wetland" examples
    1: 19_225,    # Class 1: USE ALL (smallest class - constraint)
    2: 150_000,   # Class 2: moderate wetland type
    3: 500_000,   # Class 3: largest wetland class, get lots of variety
    4: 150_000,   # Class 4: moderate wetland type
    5: 100_000,   # Class 5: moderate wetland type
}
total_target = sum(samples_per_class.values())
print(f"\nBalanced sampling strategy (target: {total_target:,} samples)")

sampled_indices_y = []
sampled_indices_x = []
sampled_labels = []

for cls in unique_classes:
    class_mask = (labels == cls)
    y_idx, x_idx = np.where(class_mask)
    
    n_available = len(y_idx)
    n_target = samples_per_class[cls]
    n_sample = min(n_target, n_available)
    
    # Sample from this class
    if n_available > n_target:
        sample_idx = np.random.choice(n_available, n_target, replace=False)
    else:
        sample_idx = np.arange(n_available)
        print(f"  ⚠ Class {cls}: only {n_available:,} available (target: {n_target:,})")
    
    sampled_indices_y.append(y_idx[sample_idx])
    sampled_indices_x.append(x_idx[sample_idx])
    sampled_labels.append(np.full(n_sample, cls))
    
    print(f"  Class {cls}: sampled {n_sample:,} / {n_available:,} pixels")

# Combine all sampled indices
y_indices = np.concatenate(sampled_indices_y)
x_indices = np.concatenate(sampled_indices_x)
y = np.concatenate(sampled_labels)

# Shuffle the samples
np.random.seed(42)  # Reproducibility
shuffle_idx = np.random.permutation(len(y_indices))
y_indices = y_indices[shuffle_idx]
x_indices = x_indices[shuffle_idx]
y = y[shuffle_idx]

print(f"\nTotal balanced samples: {len(y):,}")
unique_sampled, sampled_counts = np.unique(y, return_counts=True)
print("Sampled class distribution:")
for cls, count in zip(unique_sampled, sampled_counts):
    print(f"  Class {cls}: {count:,} samples ({100*count/len(y):.2f}%)")

# Calculate class weights for loss function (inverse frequency)
class_weights = torch.zeros(6)
for cls, count in zip(unique_sampled, sampled_counts):
    class_weights[cls] = 1.0 / count
class_weights = class_weights / class_weights.sum() * 6  # Normalize
print("\nClass weights for loss function:")
for cls in range(6):
    print(f"  Class {cls}: {class_weights[cls]:.4f}")
print("\n💡 Use these weights in your loss: nn.CrossEntropyLoss(weight=class_weights)")


# Extract embeddings for sampled pixels (OPTIMIZED: batch by row)
print("\nReading embeddings for sampled pixels (optimized batching)...")
n_samples = len(y_indices)
X = np.zeros((n_samples, embeddings_src.count), dtype=np.float32)

# Group samples by row for efficient batch reading
from collections import defaultdict
row_to_samples = defaultdict(list)
for idx, (y_coord, x_coord) in enumerate(zip(y_indices, x_indices)):
    row_to_samples[y_coord].append((idx, x_coord))

print(f"Grouped {n_samples:,} samples into {len(row_to_samples):,} unique rows")

# Read row by row
sample_count = 0
with tqdm(total=len(row_to_samples), desc="Reading rows", unit=" rows", ncols=100) as pbar:
    for row_idx in sorted(row_to_samples.keys()):
        # Read entire row at once
        row_data = embeddings_src.read(window=((row_idx, row_idx+1), (0, embeddings_src.width)))
        # Shape: (64 bands, 1 row, width) -> (64, width)
        row_data = row_data[:, 0, :]
        
        # Extract samples from this row
        for sample_idx, col_idx in row_to_samples[row_idx]:
            X[sample_idx, :] = row_data[:, col_idx]
            sample_count += 1
        
        pbar.update(1)
        pbar.set_postfix({"samples": f"{sample_count:,}/{n_samples:,}"})

embeddings_src.close()
print(f"✓ Loaded {sample_count:,} samples")

print(f"\n✓ Data loaded successfully!")
print(f"  X shape: {X.shape} ({X.nbytes / (1024**3):.2f} GB)")
print(f"  y shape: {y.shape}")
print(f"  Class distribution: {np.unique(y, return_counts=True)}")

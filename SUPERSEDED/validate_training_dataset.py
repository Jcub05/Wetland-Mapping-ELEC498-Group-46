"""
Validate and analyze wetland_dataset_1.5M_4Training.npz
"""
import numpy as np
import os

output_lines = []

def log(msg):
    print(msg)
    output_lines.append(msg)

log("="*70)
log("TRAINING DATASET VALIDATION REPORT")
log("="*70)
log(f"File: wetland_dataset_1.5M_4Training.npz")
log("")

# Check if file exists
npz_file = 'wetland_dataset_1.5M_4Training.npz'
if not os.path.exists(npz_file):
    log(f"❌ ERROR: File not found: {npz_file}")
    exit(1)

# Get file size
file_size_mb = os.path.getsize(npz_file) / (1024**2)
log(f"File size: {file_size_mb:.2f} MB")
log("")

# Load the dataset
data = np.load(npz_file)

log(f"Arrays in file: {list(data.keys())}")
log("")

# Analyze each array
for key in data.keys():
    arr = data[key]
    log(f"{key}:")
    log(f"  Shape: {arr.shape}")
    log(f"  Type: {arr.dtype}")
    log(f"  Size: {arr.nbytes / (1024**2):.2f} MB")
    
    if key == 'X':
        log(f"  Has NaN: {np.isnan(arr).any()}")
        log(f"  Has Inf: {np.isinf(arr).any()}")
        
        if not np.isnan(arr).any():
            log(f"  Min: {arr.min():.4f}")
            log(f"  Max: {arr.max():.4f}")
            log(f"  Mean: {arr.mean():.4f}")
            log(f"  Std: {arr.std():.4f}")
        else:
            log(f"  Min: nan")
            log(f"  Max: nan")
            log(f"  Mean: nan")
        
        zero_samples = (arr == 0).all(axis=1).sum()
        log(f"  All-zero samples: {zero_samples:,} ({100*zero_samples/arr.shape[0]:.2f}%)")
        log("")
        
    elif key == 'y':
        unique, counts = np.unique(arr, return_counts=True)
        log(f"  Unique classes: {unique}")
        log("")
        log(f"  Class Distribution:")
        for cls, count in zip(unique, counts):
            pct = 100 * count / arr.shape[0]
            log(f"    Class {cls}: {count:,} samples ({pct:.2f}%)")
        log("")
        
    elif key == 'class_weights':
        log(f"  Class Weights:")
        for i, weight in enumerate(arr):
            log(f"    Class {i}: {weight:.4f}")
        log("")

data.close()

# Validation checks
log("="*70)
log("VALIDATION CHECKS")
log("="*70)

data = np.load(npz_file)
X = data['X']
y = data['y']

checks_passed = 0
checks_total = 0

# Check 1: No NaN values
checks_total += 1
if not np.isnan(X).any():
    log("✅ PASS: No NaN values in embeddings")
    checks_passed += 1
else:
    log("❌ FAIL: Embeddings contain NaN values")

# Check 2: No Inf values
checks_total += 1
if not np.isinf(X).any():
    log("✅ PASS: No Inf values in embeddings")
    checks_passed += 1
else:
    log("❌ FAIL: Embeddings contain Inf values")

# Check 3: Shape alignment
checks_total += 1
if X.shape[0] == y.shape[0]:
    log(f"✅ PASS: X and y aligned ({X.shape[0]:,} samples)")
    checks_passed += 1
else:
    log(f"❌ FAIL: X has {X.shape[0]:,} samples but y has {y.shape[0]:,}")

# Check 4: Correct number of features
checks_total += 1
if X.shape[1] == 64:
    log("✅ PASS: Correct number of features (64)")
    checks_passed += 1
else:
    log(f"❌ FAIL: Expected 64 features, got {X.shape[1]}")

# Check 5: Classes present
checks_total += 1
unique_classes = np.unique(y)
if len(unique_classes) == 6 and set(unique_classes) == set(range(6)):
    log("✅ PASS: All 6 classes (0-5) present")
    checks_passed += 1
elif len(unique_classes) == 6:
    log(f"⚠️  WARNING: 6 classes present but not 0-5: {unique_classes}")
    checks_passed += 1
else:
    log(f"❌ FAIL: Expected 6 classes, found {len(unique_classes)}: {unique_classes}")

# Check 6: Class weights present
checks_total += 1
if 'class_weights' in data.keys():
    log("✅ PASS: Class weights included")
    checks_passed += 1
else:
    log("❌ FAIL: Class weights missing")

# Check 7: Reasonable value ranges
checks_total += 1
if not np.isnan(X).any():
    if X.min() >= -1.0 and X.max() <= 1.0:
        log(f"✅ PASS: Values in reasonable range [{X.min():.4f}, {X.max():.4f}]")
        checks_passed += 1
    else:
        log(f"⚠️  WARNING: Unusual value range [{X.min():.4f}, {X.max():.4f}]")
        checks_passed += 0.5

data.close()

log("")
log("="*70)
log(f"FINAL SCORE: {checks_passed}/{checks_total} checks passed")
log("="*70)

if checks_passed == checks_total:
    log("")
    log("🎉 DATASET IS READY FOR TRAINING!")
    log("")
    log("Next steps:")
    log("  1. Split into train/val/test sets")
    log("  2. Use class_weights in loss function:")
    log("     nn.CrossEntropyLoss(weight=class_weights)")
    log("  3. Start training your wetland classification model!")
elif checks_passed >= checks_total * 0.8:
    log("")
    log("⚠️  DATASET IS USABLE BUT HAS MINOR ISSUES")
    log("Review the warnings above before training.")
else:
    log("")
    log("❌ DATASET HAS CRITICAL ISSUES")
    log("Fix the failed checks before training.")

# Save report
with open('data_preprocessing/training_dataset_validation.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\n✓ Report saved to: data_preprocessing/training_dataset_validation.txt")

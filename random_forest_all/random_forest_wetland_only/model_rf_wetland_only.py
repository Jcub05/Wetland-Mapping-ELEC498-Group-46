from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# LOAD THE MIDDLE-SPLIT DATASET
# ======================================
# This script is Stage 2 of a two-stage inference pipeline:
#
#   Stage 1: CNN classifies background (Class 0) vs. not-background
#   Stage 2: THIS MODEL classifies only wetland pixels (Classes 1–5)
#
# We reuse the geographic middle-row-band split from the middle model:
#   - Test region: central latitude band (rows 9216–12288)
#   - Train: north + south tiles
#
# KEY CHANGE vs. model_rf_middle.py:
#   Class 0 (background) is FILTERED OUT from both train and test sets.
#   Class weights are recalculated purely over Classes 1–5.
#   The RF only ever sees and predicts wetland classes.
#
# DO NOT call train_test_split — that would re-introduce spatial leakage.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The middle-split .npz lives one folder up in random_forest_spatial_middle
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'random_forest_spatial_middle', 'wetland_dataset_middle_split.npz')

data = np.load(DATA_PATH)
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']
y_test  = data['y_test']
test_row_min = int(data['test_row_min'])
test_row_max = int(data['test_row_max'])
data.close()

print(f"Loaded dataset from: {DATA_PATH}")
print(f"Before filtering — Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ======================================
# FILTER OUT CLASS 0 (BACKGROUND)
# ======================================
# The CNN handles Class 0 upstream. This model only needs to distinguish
# wetland types (1–5) on the pixels the CNN says are NOT background.

train_mask = y_train != 0
test_mask  = y_test  != 0

X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test  = X_test[test_mask]
y_test  = y_test[test_mask]

print(f"\nAfter filtering Class 0:")
print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Test:  {X_test.shape[0]:,} samples")
print(f"  Train classes present: {sorted(np.unique(y_train).tolist())}")
print(f"  Test  classes present: {sorted(np.unique(y_test).tolist())}")

# ======================================
# RECALCULATE CLASS WEIGHTS (Classes 1–5 only)
# ======================================
# Recompute inverse-frequency weights over the filtered training set.
# This ensures Class 1 (rarest) still gets boosted appropriately.

unique_classes, class_counts = np.unique(y_train, return_counts=True)
n_total = len(y_train)
n_classes = len(unique_classes)

# Balanced weighting: weight_i = n_total / (n_classes * count_i)
class_weight_dict = {
    int(cls): float(n_total / (n_classes * count))
    for cls, count in zip(unique_classes, class_counts)
}

# Class 1 is very rare so its auto-calculated weight is very high,
# causing 100% recall but poor precision (trigger-happy predictions).
# Dampen it slightly to trade a little recall for better precision.
CLASS1_DAMPEN = 0.5
class_weight_dict[1] = class_weight_dict[1] * CLASS1_DAMPEN

print(f"\nRecalculated class weights (Classes 1–5, Class 1 dampened x{CLASS1_DAMPEN}):")
for cls, w in class_weight_dict.items():
    count = class_counts[list(unique_classes).index(cls)]
    print(f"  Class {cls}: weight={w:.4f}  (n={count:,})")

# ======================================
# FEATURE NORMALIZATION
# ======================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("\nFeatures normalized with StandardScaler.")

# ======================================
# TRAIN THE MODEL
# ======================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_leaf=20,
    random_state=42,
    class_weight=class_weight_dict,
    verbose=2,
    n_jobs=-1,
)

print(f"\n{'='*60}")
print("TRAINING (wetland-only, middle row band split, Classes 1–5)")
print(f"{'='*60}")
rf_model.fit(X_train, y_train)

# ======================================
# EVALUATE ON HELD-OUT GEOGRAPHIC REGION
# ======================================
y_pred = rf_model.predict(X_test)

# Use labels= to ensure metrics are computed for exactly classes 1–5
labels = sorted(class_weight_dict.keys())
target_names = [f'Class {i}' for i in labels]

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print("MODEL EVALUATION RESULTS  (wetland-only, middle band test set)")
print(f"{'='*60}")
print(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (weighted):  {precision_avg:.4f}")
print(f"Recall (weighted):     {recall_avg:.4f}")
print(f"F1-Score (weighted):   {f1_avg:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
print("\nConfusion Matrix (rows=actual, cols=predicted, classes 1–5):")
print(conf_matrix)

# ======================================
# SAVE MODEL + METADATA
# ======================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename    = f'rf_wetland_only_{timestamp}.pkl'
scaler_filename   = f'rf_wetland_only_scaler_{timestamp}.pkl'
metadata_filename = f'rf_wetland_only_{timestamp}_metadata.json'

joblib.dump(rf_model, os.path.join(SCRIPT_DIR, model_filename))
joblib.dump(scaler,   os.path.join(SCRIPT_DIR, scaler_filename))

metadata = {
    'timestamp': timestamp,
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline_stage': 'Stage 2 — wetland classification (post CNN background removal)',
    'split_method': 'middle_row_band',
    'test_row_min': test_row_min,
    'test_row_max': test_row_max,
    'classes': labels,
    'note': (
        'Class 0 (background) filtered from train and test before training. '
        'Intended for use after Stage 1 CNN background detection. '
        f'Test region: middle latitude band rows {test_row_min}–{test_row_max}. '
        'Class weights recalculated from filtered training set (Classes 1–5 only).'
    ),
    'overall_metrics': {
        'accuracy':            float(accuracy),
        'precision_weighted':  float(precision_avg),
        'recall_weighted':     float(recall_avg),
        'f1_weighted':         float(f1_avg),
    },
    'per_class_metrics': {
        str(labels[i]): {
            'precision': float(precision[i]),
            'recall':    float(recall[i]),
            'f1_score':  float(f1[i]),
            'support':   int(support[i]),
        }
        for i in range(len(labels))
    },
    'confusion_matrix': conf_matrix.tolist(),
    'confusion_matrix_labels': labels,
    'hyperparameters': {
        'n_estimators':     200,
        'max_depth':        25,
        'min_samples_leaf': 20,
        'feature_scaling':  'StandardScaler',
        'class_weight':     'recalculated_over_classes_1_to_5',
        'class1_dampen_factor': CLASS1_DAMPEN,
        'n_jobs':           -1,
        'random_state':     42,
    },
    'dataset': {
        'source': '../random_forest_spatial_middle/wetland_dataset_middle_split.npz',
        'class_0_filtered': True,
        'n_train': int(X_train.shape[0]),
        'n_test':  int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
    },
    'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()},
}

with open(os.path.join(SCRIPT_DIR, metadata_filename), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print("MODEL SAVED")
print(f"{'='*60}")
print(f"Model:    {model_filename}")
print(f"Scaler:   {scaler_filename}")
print(f"Metadata: {metadata_filename}")
print(f"\nReminder: At inference time, only pass pixels where the CNN")
print(f"predicted NOT background (i.e., CNN class != 0) into this model.")

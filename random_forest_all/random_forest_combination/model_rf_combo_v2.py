from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# RF Combination (Stage 1 + 2) v2 BENCHMARKING SCRIPT
# 
# Tunes: Aligned with standalone RF model parameters.
# Stage 1: n_estimators=200, max_depth=25, min_samples_leaf=20
# Stage 2: n_estimators=200, max_depth=25, min_samples_leaf=20
# Weights: Directly derived from .npz with NO dynamic boosting/dampening.
# ======================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'random_forest_spatial_middle', 'wetland_dataset_middle_split.npz')

# ---- Aligned Parameters ----
N_ESTIMATORS = 200
MAX_DEPTH = 25
MIN_SAMPLES_LEAF = 20

# ======================================
# LOAD DATA
# ======================================
data = np.load(DATA_PATH)
X_train = data['X_train']
y_train_raw = data['y_train']
X_test  = data['X_test']
y_test_raw  = data['y_test']
test_row_min = int(data['test_row_min'])
test_row_max = int(data['test_row_max'])
class_weights = data['class_weights']
data.close()

print(f"Loaded: {DATA_PATH}")
print(f"Total samples — Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}\n")

# ======================================
# FEATURE NORMALIZATION
# ======================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("Features normalized with StandardScaler.\n")

# Convert class weights to dict
original_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

# ======================================
# PREPARE STAGE 1 (Background) DATA & WEIGHTS
# ======================================
y_train_s1 = (y_train_raw != 0).astype(int)

# Use balanced weights for binary classification Stage 1
s1_unique_classes, s1_counts = np.unique(y_train_s1, return_counts=True)
s1_weight_dict = {
    int(cls): float(len(y_train_s1) / (len(s1_unique_classes) * count))
    for cls, count in zip(s1_unique_classes, s1_counts)
}

print(f"--- Stage 1 Preparation ---")
print(f"  Classes present: {s1_unique_classes.tolist()}")
print("  Class weights (Binary Balanced):")
for cls, w in s1_weight_dict.items():
    print(f"    Class {cls}: weight={w:.4f}")
print()

# ======================================
# PREPARE STAGE 2 (Wetland) DATA & WEIGHTS
# ======================================
s2_train_mask = y_train_raw != 0
X_train_s2 = X_train[s2_train_mask]
y_train_s2 = y_train_raw[s2_train_mask]

# Use exact .npz weights for classes 1-5 to perfectly align penalization with standalone RF
s2_weight_dict = {cls: original_weight_dict[cls] for cls in range(1, 6)}

s2_unique_classes = np.unique(y_train_s2)

print(f"--- Stage 2 Preparation ---")
print(f"  Data shape after filtering Background: {X_train_s2.shape[0]:,}")
print(f"  Classes present: {s2_unique_classes.tolist()}")
print("  Class weights (Direct from .npz):")
for cls, w in s2_weight_dict.items():
    print(f"    Class {cls}: weight={w:.4f}")
print()


# ======================================
# TRAIN STAGE 1 MODEL
# ======================================
print(f"{'='*60}")
print("TRAINING STAGE 1 MODEL")
print(f"n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
print(f"{'='*60}")

t_start_s1 = datetime.now()
rf_stage1 = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42,
    class_weight=s1_weight_dict,
    verbose=2,
    n_jobs=-1,
)
rf_stage1.fit(X_train, y_train_s1)
t_end_s1 = datetime.now()
train_secs_s1 = (t_end_s1 - t_start_s1).total_seconds()
print(f"\nStage 1 Training Time: {train_secs_s1:.1f}s\n")

# Run Stage 1 inference
print("Running Stage 1 Inference on full test set...")
preds_stage1_full = rf_stage1.predict(X_test)
wetland_mask_full = (preds_stage1_full == 1)
print(f"  -> Identified {np.sum(wetland_mask_full):,} valid wetland pixels out of {len(X_test):,}\n")

# Filter Stage 2 test data
X_test_s2_masked = X_test[wetland_mask_full]


# ======================================
# TRAIN STAGE 2 MODEL
# ======================================
print(f"{'='*60}")
print("TRAINING STAGE 2 MODEL")
print(f"n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
print(f"{'='*60}")

t_start_s2 = datetime.now()
rf_stage2 = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42,
    class_weight=s2_weight_dict,
    verbose=2,
    n_jobs=-1,
)
rf_stage2.fit(X_train_s2, y_train_s2)
t_end_s2 = datetime.now()
train_secs_s2 = (t_end_s2 - t_start_s2).total_seconds()
print(f"\nStage 2 Training Time: {train_secs_s2:.1f}s\n")

# ======================================
# INFERENCE AND EVALUATION
# ======================================
print("Running final combination inference...")
final_predictions = np.zeros(X_test.shape[0], dtype=np.int32)
if np.sum(wetland_mask_full) > 0:
    preds_stage2_masked = rf_stage2.predict(X_test_s2_masked)
    final_predictions[wetland_mask_full] = preds_stage2_masked

labels_full = [0, 1, 2, 3, 4, 5]

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_raw, final_predictions, labels=labels_full, average=None, zero_division=0
)
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
    y_test_raw, final_predictions, labels=labels_full, average='weighted', zero_division=0
)
conf_matrix = confusion_matrix(y_test_raw, final_predictions, labels=labels_full)
accuracy = accuracy_score(y_test_raw, final_predictions)

# Calculate mean wetland F1
f1_wetlands_only = f1[1:]
mean_wetland_f1 = float(np.mean(f1_wetlands_only))

print(f"\n{'='*60}")
print("MODEL EVALUATION RESULTS (Combo v2)")
print(f"{'='*60}")
print(f"Accuracy:              {accuracy:.4f}")
print(f"Precision (weighted):  {precision_avg:.4f}")
print(f"Recall (weighted):     {recall_avg:.4f}")
print(f"F1-Score (weighted):   {f1_avg:.4f}")
print(f"Mean Wetland F1:       {mean_wetland_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_raw, final_predictions, target_names=[f'Class {i}' for i in range(6)], zero_division=0))
print("\nConfusion Matrix:")
print(conf_matrix)

# ======================================
# SAVE MODEL + METADATA
# ======================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_stage1_filename = f'rf_combo_v2_stage1_{timestamp}.pkl'
model_stage2_filename = f'rf_combo_v2_stage2_{timestamp}.pkl'
scaler_filename       = f'rf_combo_v2_scaler_{timestamp}.pkl'
metadata_filename     = f'rf_combo_v2_{timestamp}_metadata.json'

joblib.dump(rf_stage1, os.path.join(SCRIPT_DIR, model_stage1_filename))
joblib.dump(rf_stage2, os.path.join(SCRIPT_DIR, model_stage2_filename))
joblib.dump(scaler,    os.path.join(SCRIPT_DIR, scaler_filename))

metadata = {
    'timestamp': timestamp,
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline_stage': 'Two-Stage Combination Pipeline Benchmark (v2)',
    'split_method': 'middle_row_band',
    'test_row_min': test_row_min,
    'test_row_max': test_row_max,
    'note': 'Benchmarking script designed to perfectly align parameters with the standalone RF middle split model.',
    'overall_metrics': {
        'accuracy':           float(accuracy),
        'precision_weighted': float(precision_avg),
        'recall_weighted':    float(recall_avg),
        'f1_weighted':        float(f1_avg),
        'mean_wetland_f1':    mean_wetland_f1,
    },
    'per_class_metrics': {
        str(i): {
            'precision': float(precision[i]),
            'recall':    float(recall[i]),
            'f1_score':  float(f1[i]),
            'support':   int(support[i]),
        }
        for i in range(len(labels_full))
    },
    'confusion_matrix': conf_matrix.tolist(),
    'hyperparameters': {
        'n_estimators':     N_ESTIMATORS,
        'max_depth':        MAX_DEPTH,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'feature_scaling':  'StandardScaler',
        'n_jobs':           -1,
        'random_state':     42,
    },
    'dataset': {
        'source':     '../random_forest_spatial_middle/wetland_dataset_middle_split.npz',
        'n_train':    int(X_train.shape[0]),
        'n_test':     int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
    },
    'stage1_train_seconds': train_secs_s1,
    'stage2_train_seconds': train_secs_s2,
    'stage1_weights': s1_weight_dict,
    'stage2_weights_from_npz': s2_weight_dict,
}

with open(os.path.join(SCRIPT_DIR, metadata_filename), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print("MODEL SAVED")
print(f"{'='*60}")
print(f"Stage 1 Model: {model_stage1_filename}")
print(f"Stage 2 Model: {model_stage2_filename}")
print(f"Scaler:        {scaler_filename}")
print(f"Metadata:      {metadata_filename}")

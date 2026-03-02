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
# This dataset was created by create_training_dataset_MIDDLE_SPLIT.ipynb.
# Strategy: hold out a horizontal strip from the MIDDLE of the map (~rows 40-60%)
# as the test region. Train on north + south (both sides surround the test strip).
#
# All 6 classes are split geographically — no within-zone random fallback needed.
# Class 1 and 2 row ranges overlap the middle band, giving them genuine geographic splits.
# DO NOT call train_test_split — that would re-introduce spatial leakage.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(SCRIPT_DIR, 'wetland_dataset_middle_split.npz'))
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']
y_test  = data['y_test']
class_weights = data['class_weights']
test_row_min  = int(data['test_row_min'])
test_row_max  = int(data['test_row_max'])
data.close()

print(f"Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")
print(f"Test region: rows {test_row_min}–{test_row_max} (middle latitude band)")
print(f"Features: {X_train.shape[1]}")
print(f"Train classes: {sorted(set(y_train.tolist()))}")
print(f"Test  classes: {sorted(set(y_test.tolist()))}")

# Convert class weights to dict
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
print(f"\nClass weights (from .npz): {class_weight_dict}")

# ======================================
# FEATURE NORMALIZATION
# ======================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("Features normalized with StandardScaler.")

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
print("TRAINING (middle row band spatial split)")
print(f"{'='*60}")
rf_model.fit(X_train, y_train)

# ======================================
# EVALUATE ON HELD-OUT GEOGRAPHIC REGION
# ======================================
y_pred = rf_model.predict(X_test)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print("MODEL EVALUATION RESULTS  (middle band test set)")
print(f"{'='*60}")
print(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (weighted):  {precision_avg:.4f}")
print(f"Recall (weighted):     {recall_avg:.4f}")
print(f"F1-Score (weighted):   {f1_avg:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(6)]))
print("\nConfusion Matrix:")
print(conf_matrix)

# ======================================
# SAVE MODEL + METADATA
# ======================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename    = f'rf_wetland_model_middle_{timestamp}.pkl'
scaler_filename   = f'rf_scaler_middle_{timestamp}.pkl'
metadata_filename = f'rf_wetland_model_middle_{timestamp}_metadata.json'

joblib.dump(rf_model, os.path.join(SCRIPT_DIR, model_filename))
joblib.dump(scaler,   os.path.join(SCRIPT_DIR, scaler_filename))

metadata = {
    'timestamp': timestamp,
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'split_method': 'middle_row_band',
    'test_row_min': test_row_min,
    'test_row_max': test_row_max,
    'note': (
        'Middle row band split: test pixels are from tiles in the central latitude '
        f'(rows {test_row_min}–{test_row_max}). Training uses north + south tiles. '
        'All 6 classes are geographically separated — no within-zone random splits needed.'
    ),
    'overall_metrics': {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision_avg),
        'recall_weighted': float(recall_avg),
        'f1_weighted': float(f1_avg),
    },
    'per_class_metrics': {
        str(i): {
            'precision': float(precision[i]),
            'recall':    float(recall[i]),
            'f1_score':  float(f1[i]),
            'support':   int(support[i]),
        }
        for i in range(len(precision))
    },
    'confusion_matrix': conf_matrix.tolist(),
    'hyperparameters': {
        'n_estimators': 200,
        'max_depth': 25,
        'min_samples_leaf': 20,
        'feature_scaling': 'StandardScaler',
        'class_weight': 'from_npz',
        'n_jobs': -1,
        'random_state': 42,
    },
    'dataset': {
        'source': 'random_forest_spatial_middle/wetland_dataset_middle_split.npz',
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

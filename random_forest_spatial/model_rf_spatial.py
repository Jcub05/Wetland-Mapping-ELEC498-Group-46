from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# LOAD THE SPATIALLY-SPLIT DATASET
# ======================================
# This dataset was created by create_training_dataset_SPATIAL_SPLIT.ipynb
# Train and test have ALREADY been split geographically (by tile region).
# DO NOT call train_test_split — that would re-introduce spatial leakage.

data = np.load('../wetland_dataset_spatial_split.npz')
X_train = data['X_train']   # pixels from TRAINING tiles only
y_train = data['y_train']
X_test  = data['X_test']    # pixels from TEST tiles only (geographically separate)
y_test  = data['y_test']
class_weights = data['class_weights']
test_row_min  = int(data['test_row_min'])  # raster row where test region starts
data.close()

print(f"Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")
print(f"Test region starts at row offset: {test_row_min}")
print(f"Features: {X_train.shape[1]}")

# Convert class weights to dict
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

# ======================================
# TRAIN THE MODEL
# ======================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=class_weight_dict,
    verbose=2,
    n_jobs=-1,
)

print(f"\n{'='*60}")
print("TRAINING (spatial split — geographically honest)")
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
print("MODEL EVALUATION RESULTS  (spatial holdout test set)")
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
model_filename   = f'rf_wetland_model_spatial_{timestamp}.pkl'
metadata_filename = f'rf_wetland_model_spatial_{timestamp}_metadata.json'

joblib.dump(rf_model, model_filename)

metadata = {
    'timestamp': timestamp,
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'split_method': 'spatial_tile_holdout',
    'test_row_min': test_row_min,
    'note': (
        'Train/test split is geographic: test pixels come exclusively from '
        'tiles with row_offset >= test_row_min. No spatial leakage.'
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
        'n_estimators': 100,
        'class_weight': 'custom',
        'n_jobs': -1,
        'random_state': 42,
    },
    'dataset': {
        'source': '../wetland_dataset_spatial_split.npz',
        'n_train': int(X_train.shape[0]),
        'n_test':  int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
    },
    'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()},
}

with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print("MODEL SAVED")
print(f"{'='*60}")
print(f"Model:    {model_filename}")
print(f"Metadata: {metadata_filename}")
print(f"{'='*60}")
print(f"\nNext: run  python visualize_test_region.py <embeddings_dir>  to")
print(f"      generate the side-by-side ground-truth vs prediction map.")

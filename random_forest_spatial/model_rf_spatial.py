from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# LOAD THE SMART-SPLIT DATASET
# ======================================
# This dataset was created by create_training_dataset_SMART_SPLIT.ipynb.
# Uses a MIXED split strategy:
#   - Classes 0, 3, 4, 5: geographic column split (eastern tiles = train, western = test)
#   - Classes 1 & 2: random 75/25 within their western zone (geographically confined)
# DO NOT call train_test_split — that would re-introduce spatial leakage.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(SCRIPT_DIR, 'wetland_dataset_smart_split.npz'))
X_train = data['X_train']   # pixels from training region
y_train = data['y_train']
X_test  = data['X_test']    # pixels from test region (geographically separate for cls 0,3,4,5)
y_test  = data['y_test']
class_weights = data['class_weights']
test_col_max  = int(data['test_col_max'])  # column threshold: tiles with col < this are test tiles
data.close()

print(f"Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")
print(f"Test region: tiles with col_offset < {test_col_max} (western ~22% of map)")
print(f"Features: {X_train.shape[1]}")
print(f"Train classes: {sorted(set(y_train.tolist()))}")
print(f"Test  classes: {sorted(set(y_test.tolist()))}")

# Convert class weights to dict
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

# Dampen Class 2 weight — it's geographically confined to the same western region
# as the test set, causing the model to massively over-predict it.
# Reducing its weight forces the model to compete with other classes.
class_weight_dict[2] = class_weight_dict[2] * 0.3
print(f"Adjusted class weights: {class_weight_dict}")

# ======================================
# FEATURE NORMALIZATION
# ======================================
# Standardize embedding dimensions to help the model generalize
# across the geographic domain shift between eastern and western tiles.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("Features normalized with StandardScaler.")

# ======================================
# TRAIN THE MODEL
# ======================================
rf_model = RandomForestClassifier(
    n_estimators=200,       # more trees for stable predictions
    max_depth=25,           # limit depth to prevent overfitting to eastern landscape
    min_samples_leaf=20,    # smoother decision boundaries -> better geographic generalization
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

joblib.dump(rf_model, os.path.join(SCRIPT_DIR, model_filename))
joblib.dump(scaler,   os.path.join(SCRIPT_DIR, f'rf_scaler_{timestamp}.pkl'))

metadata = {
    'timestamp': timestamp,
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'split_method': 'mixed_spatial_split',
    'test_col_max': test_col_max,
    'note': (
        'Mixed split: Classes 0,3,4,5 use geographic column tile split '
        '(test = tiles with col_offset < test_col_max). '
        'Classes 1 and 2 use random 75/25 within-zone split (geographically confined). '
        'Classes 1 and 2 splits are NOT geographically independent — documented limitation.'
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
        'class_weight': 'custom_with_class2_dampened_0.3x',
        'feature_scaling': 'StandardScaler',
        'n_jobs': -1,
        'random_state': 42,
    },
    'dataset': {
        'source': 'random_forest_spatial/wetland_dataset_smart_split.npz',
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

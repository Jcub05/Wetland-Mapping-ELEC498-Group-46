from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# WEIGHT ADJUSTMENT FACTORS
# ======================================
# v1 (original) had background recall of only 27.8% — the model was
# predicting Marsh (class 3) far too aggressively due to prior mismatch
# (training set ~40% background, real map ~97% background).
#
# Conservative corrections applied to the npz base weights:
#   Class 0 (Background):    x1.8  — boost to counter prior mismatch
#   Class 3 (Marsh):         x0.7  — dampen over-prediction
#   Class 5 (Swamp):         x0.85 — minor dampening, low precision in v1
CLASS0_BOOST  = 1.8
CLASS3_DAMPEN = 0.7
CLASS5_DAMPEN = 0.85

# ======================================
# LOAD THE MIDDLE-SPLIT DATASET
# ======================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
data = np.load(os.path.join(REPO_ROOT, 'wetland_dataset_middle_split.npz'))
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

# Convert base weights and apply adjustments
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
class_weight_dict[0] = class_weight_dict[0] * CLASS0_BOOST
class_weight_dict[3] = class_weight_dict[3] * CLASS3_DAMPEN
class_weight_dict[5] = class_weight_dict[5] * CLASS5_DAMPEN

print(f"\nAdjusted class weights:")
for cls, w in class_weight_dict.items():
    print(f"  Class {cls}: {w:.4f}")

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
    verbose=0,
    n_jobs=-1,
)

print(f"\n{'='*60}")
print("TRAINING v2 (adjusted class weights)")
print(f"{'='*60}")
rf_model.fit(X_train, y_train)

# ======================================
# EVALUATE
# ======================================
y_pred = rf_model.predict(X_test)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
mean_f1 = float(np.mean(f1))

print(f"\n{'='*60}")
print("MODEL EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (weighted):  {precision_avg:.4f}")
print(f"Recall (weighted):     {recall_avg:.4f}")
print(f"F1-Score (weighted):   {f1_avg:.4f}")
print(f"Mean F1 (all classes): {mean_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(6)]))
print("\nConfusion Matrix:")
print(conf_matrix)

# Compare background recall specifically
bg_idx = 0
print(f"\nBackground recall: {recall[bg_idx]:.4f}  (was 0.2779 in v1)")
print(f"Marsh recall:      {recall[3]:.4f}  (was 0.8919 in v1)")

# ======================================
# SAVE MODEL + METADATA
# ======================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename    = f'rf_wetland_model_middle_v2_{timestamp}.pkl'
scaler_filename   = f'rf_scaler_middle_v2_{timestamp}.pkl'
metadata_filename = f'rf_wetland_model_middle_v2_{timestamp}_metadata.json'

joblib.dump(rf_model, os.path.join(SCRIPT_DIR, model_filename))
joblib.dump(scaler,   os.path.join(SCRIPT_DIR, scaler_filename))

metadata = {
    'timestamp': timestamp,
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'split_method': 'middle_row_band',
    'test_row_min': test_row_min,
    'test_row_max': test_row_max,
    'version': 'v2 — adjusted class weights to fix background under-recall',
    'weight_adjustments': {
        'class0_boost':  CLASS0_BOOST,
        'class3_dampen': CLASS3_DAMPEN,
        'class5_dampen': CLASS5_DAMPEN,
    },
    'overall_metrics': {
        'accuracy':           float(accuracy),
        'precision_weighted': float(precision_avg),
        'recall_weighted':    float(recall_avg),
        'f1_weighted':        float(f1_avg),
        'mean_f1_all_classes': mean_f1,
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
        'n_estimators':     200,
        'max_depth':        25,
        'min_samples_leaf': 20,
        'feature_scaling':  'StandardScaler',
        'class_weight':     'from_npz_with_adjustments',
        'n_jobs':           -1,
        'random_state':     42,
    },
    'dataset': {
        'source':     'wetland_dataset_middle_split.npz',
        'n_train':    int(X_train.shape[0]),
        'n_test':     int(X_test.shape[0]),
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

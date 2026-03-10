"""
SVM RBF Two-Stage Combination Pipeline
========================================
Stage 1: Binary SVM (Background vs Wetland)
Stage 2: Multi-class SVM (Classes 1–5 Wetland types)

Run this AFTER both grid searches to evaluate the full end-to-end pipeline
using the best models from each stage.

Truth-source class mapping:
  0 = Background
  1 = Fen (Graminoid)
  2 = Fen (Woody)
  3 = Marsh
  4 = Shallow Open Water
  5 = Swamp

Usage:
  1. Run model_svm_rbf_background_grid_search.py  → note best Stage 1 model
  2. Run model_svm_rbf_wetland_grid_search.py     → note best Stage 2 model
  3. Update STAGE1_MODEL_PATH and STAGE2_MODEL_PATH below
  4. Run this script
"""

import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── !! UPDATE THESE PATHS AFTER GRID SEARCHES !! ─────────────────────────────
STAGE1_MODEL_PATH  = os.path.join(SCRIPT_DIR, 'svm_rbf_background',
                                  'BEST_STAGE1_MODEL.pkl')   # <- update
STAGE2_MODEL_PATH  = os.path.join(SCRIPT_DIR, 'svm_rbf_wetland_only',
                                  'BEST_STAGE2_MODEL.pkl')   # <- update
SCALER_PATH        = os.path.join(SCRIPT_DIR, 'svm_rbf_background',
                                  'svm_rbf_bg_scaler.pkl')

DATA_PATH = os.path.join(
    SCRIPT_DIR, '..', 'random_forest_all',
    'random_forest_spatial_middle', 'wetland_dataset_middle_split.npz'
)
# ─────────────────────────────────────────────────────────────────────────────

print("Loading models and scaler...")
stage1_model = joblib.load(STAGE1_MODEL_PATH)
stage2_model = joblib.load(STAGE2_MODEL_PATH)
scaler       = joblib.load(SCALER_PATH)
print(f"  Stage 1: {os.path.basename(STAGE1_MODEL_PATH)}")
print(f"  Stage 2: {os.path.basename(STAGE2_MODEL_PATH)}")
print(f"  Scaler:  {os.path.basename(SCALER_PATH)}\n")

# ── Load test data ─────────────────────────────────────────────────────────────
data = np.load(DATA_PATH)
X_test_raw   = data['X_test']
y_test_raw   = data['y_test']
test_row_min = int(data['test_row_min'])
test_row_max = int(data['test_row_max'])
data.close()

print(f"Test samples: {X_test_raw.shape[0]:,}")

X_test = scaler.transform(X_test_raw.astype(np.float32))

# ── Stage 1: Predict background vs wetland ────────────────────────────────────
print("\nRunning Stage 1 (background vs wetland)...")
t_s1 = datetime.now()
s1_preds = np.array(stage1_model.predict(X_test))
t_s1_end = datetime.now()
inf_s1 = (t_s1_end - t_s1).total_seconds()

wetland_mask = (s1_preds == 1)
n_wetland    = int(np.sum(wetland_mask))
print(f"  Stage 1 identified {n_wetland:,} wetland pixels "
      f"({n_wetland/len(X_test)*100:.1f}% of test set)")

# ── Stage 2: Predict wetland class for masked pixels ──────────────────────────
print("\nRunning Stage 2 (wetland multi-class)...")
final_predictions = np.zeros(X_test.shape[0], dtype=np.int32)

t_s2 = datetime.now()
if n_wetland > 0:
    s2_preds = np.array(stage2_model.predict(X_test[wetland_mask]))
    final_predictions[wetland_mask] = s2_preds
t_s2_end = datetime.now()
inf_s2 = (t_s2_end - t_s2).total_seconds()

# ── Evaluate full pipeline against multi-class ground truth ───────────────────
labels_full = [0, 1, 2, 3, 4, 5]
class_names = {
    0: "Background",
    1: "Fen (Graminoid)",
    2: "Fen (Woody)",
    3: "Marsh",
    4: "Shallow Open Water",
    5: "Swamp",
}

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_raw, final_predictions, labels=labels_full, average=None, zero_division=0
)
prec_avg, rec_avg, f1_avg, _ = precision_recall_fscore_support(
    y_test_raw, final_predictions, labels=labels_full, average='weighted', zero_division=0
)
conf_matrix     = confusion_matrix(y_test_raw, final_predictions, labels=labels_full)
accuracy        = accuracy_score(y_test_raw, final_predictions)
mean_wetland_f1 = float(np.mean(f1[1:]))   # classes 1–5 only

print(f"\n{'='*55}")
print("FULL PIPELINE RESULTS")
print(f"{'='*55}")
print(f"  Accuracy:         {accuracy:.4f}")
print(f"  Weighted F1:      {f1_avg:.4f}")
print(f"  Mean Wetland F1:  {mean_wetland_f1:.4f}")
print(f"\n  Per-class F1:")
for i, cls in enumerate(labels_full):
    print(f"    {cls} ({class_names[cls]}):  "
          f"P={precision[i]:.3f}  R={recall[i]:.3f}  F1={f1[i]:.3f}  n={support[i]:,}")
print(f"\n  Stage 1 inference: {inf_s1:.2f}s")
print(f"  Stage 2 inference: {inf_s2:.2f}s")

# ── Save stats JSON ───────────────────────────────────────────────────────────
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
metadata = {
    'timestamp': timestamp,
    'evaluation_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline_stage': 'Two-Stage SVM RBF Pipeline (Stage 1 Binary + Stage 2 Wetland)',
    'split_method': 'middle_row_band',
    'test_row_min': test_row_min,
    'test_row_max': test_row_max,
    'models_used': {
        'stage1_binary':  os.path.basename(STAGE1_MODEL_PATH),
        'stage2_wetland': os.path.basename(STAGE2_MODEL_PATH),
        'scaler':         os.path.basename(SCALER_PATH),
    },
    'classes': labels_full,
    'overall_metrics': {
        'accuracy':           float(accuracy),
        'precision_weighted': float(prec_avg),
        'recall_weighted':    float(rec_avg),
        'f1_weighted':        float(f1_avg),
        'mean_wetland_f1':    mean_wetland_f1,
    },
    'per_class_metrics': {
        str(labels_full[i]): {
            'class_name': class_names[labels_full[i]],
            'precision':  float(precision[i]),
            'recall':     float(recall[i]),
            'f1_score':   float(f1[i]),
            'support':    int(support[i]),
        }
        for i in range(len(labels_full))
    },
    'confusion_matrix':        conf_matrix.tolist(),
    'confusion_matrix_labels': labels_full,
    'dataset': {
        'source':     DATA_PATH,
        'n_test':     int(X_test.shape[0]),
        'n_features': int(X_test.shape[1]),
    },
    'stage1_wetland_pixels_found': n_wetland,
    'timing': {
        'stage1_inference_seconds': inf_s1,
        'stage2_inference_seconds': inf_s2,
        'total_inference_seconds':  inf_s1 + inf_s2,
    },
}

stats_dir  = os.path.join(
    SCRIPT_DIR, '..', 'Statistics', 'SVM'
)
os.makedirs(stats_dir, exist_ok=True)
stats_path = os.path.join(stats_dir, f'svm_combo_pipeline_{timestamp}.json')
with open(stats_path, 'w') as f_out:
    json.dump(metadata, f_out, indent=2)
print(f"\nStats saved: {stats_path}")

# Also save alongside the script for convenience
local_path = os.path.join(SCRIPT_DIR, f'svm_combo_pipeline_{timestamp}_metadata.json')
with open(local_path, 'w') as f_out:
    json.dump(metadata, f_out, indent=2)
print(f"Local copy:  {local_path}")

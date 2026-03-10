"""
GRID SEARCH — SVM RBF Wetland-Only (Stage 2)
=============================================
Trains on classes 1–5 only (background already removed).
Tunes: C x gamma → 4 runs

Truth-source class mapping:
  1 = Fen (Graminoid)
  2 = Fen (Woody)
  3 = Marsh
  4 = Shallow Open Water
  5 = Swamp

Usage:
  Run AFTER model_svm_rbf_background_grid_search.py.
  Point BEST_BG_SCALER_PATH to the scaler saved by Stage 1.
  The same scaler MUST be reused — do not refit.
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

# ── GPU / CPU backend ─────────────────────────────────────────────────────────
try:
    from cuml.svm import SVC
    BACKEND  = "cuML (GPU)"
    USE_CUML = True
except ImportError:
    from sklearn.svm import SVC
    BACKEND  = "sklearn (CPU)"
    USE_CUML = False

print(f"SVM backend: {BACKEND}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    SCRIPT_DIR, '..', '..', 'random_forest_all',
    'random_forest_spatial_middle', 'wetland_dataset_middle_split.npz'
)

# !! Update this to the scaler produced by Stage 1 grid search !!
BEST_BG_SCALER_PATH = os.path.join(
    SCRIPT_DIR, '..', 'svm_rbf_background', 'svm_rbf_bg_scaler.pkl'
)

# ── Hyperparameter grid ───────────────────────────────────────────────────────
C_OPTIONS     = [1.0, 10.0]
GAMMA_OPTIONS = [0.001, 'scale']

# Per-class weight dampening for Fen (Graminoid) — class 1 tends to
# dominate with balanced weights due to very low support
CLASS1_DAMPEN = 0.4

# ── Load data ─────────────────────────────────────────────────────────────────
data = np.load(DATA_PATH)
X_train_raw = data['X_train']
y_train_raw = data['y_train']
X_test_raw  = data['X_test']
y_test_raw  = data['y_test']
test_row_min = int(data['test_row_min'])
test_row_max = int(data['test_row_max'])
data.close()

print(f"Loaded: {DATA_PATH}")
print(f"Train: {X_train_raw.shape[0]:,}  |  Test: {X_test_raw.shape[0]:,}\n")

# ── Scale using Stage 1 scaler (MUST NOT refit) ───────────────────────────────
print(f"Loading Stage 1 scaler: {BEST_BG_SCALER_PATH}")
scaler = joblib.load(BEST_BG_SCALER_PATH)
X_train_full = scaler.transform(X_train_raw.astype(np.float32))
X_test_full  = scaler.transform(X_test_raw.astype(np.float32))
print("Scaler applied.\n")

# ── Filter to wetland pixels only (classes 1–5) ───────────────────────────────
train_mask = y_train_raw != 0
X_train = X_train_full[train_mask]
y_train = y_train_raw[train_mask].astype(np.int32)

print(f"Training on wetland pixels only: {X_train.shape[0]:,} samples")
unique, counts = np.unique(y_train, return_counts=True)
class_names = {1: "Fen (Graminoid)", 2: "Fen (Woody)", 3: "Marsh",
               4: "Shallow Open Water", 5: "Swamp"}
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls} ({class_names[cls]}): {cnt:,}")
print()

# ── Per-class weights ────────────────────────────────────────────────────────
n_total   = len(y_train)
n_classes = len(unique)
weight_dict = {
    int(cls): float(n_total / (n_classes * cnt))
    for cls, cnt in zip(unique, counts)
}
weight_dict[1] = weight_dict[1] * CLASS1_DAMPEN   # dampen Fen (Graminoid)

print("Class weights (with CLASS1_DAMPEN applied):")
for cls, w in weight_dict.items():
    print(f"  Class {cls} ({class_names[cls]}): {w:.4f}")
print()

# ── Keep full test set to evaluate pipeline end-to-end ───────────────────────
# Stage 2 only predicts on the pixels Stage 1 would pass through.
# For a fair apples-to-apples comparison, we re-filter at test time too.
test_mask  = y_test_raw != 0
X_test_s2  = X_test_full[test_mask]
y_test_s2  = y_test_raw[test_mask].astype(np.int32)

labels_s2  = [1, 2, 3, 4, 5]

print(f"Test set (wetland pixels): {X_test_s2.shape[0]:,}\n")

# ── Grid search ───────────────────────────────────────────────────────────────
results = []
total   = len(C_OPTIONS) * len(GAMMA_OPTIONS)
run_num = 0

for C in C_OPTIONS:
    for gamma in GAMMA_OPTIONS:
        run_num += 1
        gamma_label = str(gamma)
        print(f"{'='*60}")
        print(f"RUN {run_num}/{total}  C={C}  gamma={gamma_label}  [{BACKEND}]")
        print(f"{'='*60}")

        t_start = datetime.now()

        svm_kwargs = dict(
            kernel='rbf',
            C=C,
            gamma=gamma if gamma != 'scale' else 'scale',
            class_weight=weight_dict,
            probability=True,
        )
        if USE_CUML:
            svm_kwargs.pop('probability')

        model = SVC(**svm_kwargs)
        model.fit(X_train, y_train)

        t_end     = datetime.now()
        train_sec = (t_end - t_start).total_seconds()

        y_pred = model.predict(X_test_s2)
        if USE_CUML:
            y_pred = np.array(y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_s2, y_pred, labels=labels_s2, average=None, zero_division=0
        )
        prec_avg, rec_avg, f1_avg, _ = precision_recall_fscore_support(
            y_test_s2, y_pred, labels=labels_s2, average='weighted', zero_division=0
        )
        conf_matrix  = confusion_matrix(y_test_s2, y_pred, labels=labels_s2)
        accuracy     = accuracy_score(y_test_s2, y_pred)
        mean_wet_f1  = float(np.mean(f1))

        print(f"  Accuracy:        {accuracy:.4f}")
        print(f"  Weighted F1:     {f1_avg:.4f}")
        print(f"  Mean Wetland F1: {mean_wet_f1:.4f}")
        per_f1 = {labels_s2[i]: round(float(f1[i]), 4) for i in range(len(labels_s2))}
        print(f"  Per-class F1:    {per_f1}")
        print(f"  Train time:      {train_sec:.1f}s")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'svm_rbf_wetland_C{C}_gamma{gamma_label}_{timestamp}.pkl'
        model_path = os.path.join(SCRIPT_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  Model saved: {model_filename}")

        metadata = {
            'timestamp': timestamp,
            'trained_datetime': t_end.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_stage': 'Stage 2 — wetland multi-class SVM (grid search, classes 1-5)',
            'backend': BACKEND,
            'split_method': 'middle_row_band',
            'test_row_min': test_row_min,
            'test_row_max': test_row_max,
            'classes': labels_s2,
            'note': (
                'Class 0 filtered before training. Test evaluated on wetland pixels only. '
                'Class 1 (Fen Graminoid) weight dampened by CLASS1_DAMPEN factor. '
                'For full-pipeline metrics, combine with Stage 1 output using the combo script.'
            ),
            'overall_metrics': {
                'accuracy':           float(accuracy),
                'precision_weighted': float(prec_avg),
                'recall_weighted':    float(rec_avg),
                'f1_weighted':        float(f1_avg),
                'mean_wetland_f1':    mean_wet_f1,
            },
            'per_class_metrics': {
                str(labels_s2[i]): {
                    'class_name':  class_names[labels_s2[i]],
                    'precision':   float(precision[i]),
                    'recall':      float(recall[i]),
                    'f1_score':    float(f1[i]),
                    'support':     int(support[i]),
                }
                for i in range(len(labels_s2))
            },
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_labels': labels_s2,
            'hyperparameters': {
                'kernel':            'rbf',
                'C':                 C,
                'gamma':             gamma_label,
                'class_weight':      'balanced then dampened',
                'class1_dampen':     CLASS1_DAMPEN,
                'feature_scaling':   'StandardScaler (fitted in Stage 1)',
            },
            'dataset': {
                'source':          DATA_PATH,
                'class_0_filtered': True,
                'n_train':         int(X_train.shape[0]),
                'n_test':          int(X_test_s2.shape[0]),
                'n_features':      int(X_train.shape[1]),
            },
            'class_weights':    {str(k): round(v, 6) for k, v in weight_dict.items()},
            'train_seconds':    train_sec,
            'model_file':       model_filename,
            'scaler_file':      BEST_BG_SCALER_PATH,
        }

        meta_path = os.path.join(
            SCRIPT_DIR,
            f'svm_rbf_wetland_C{C}_gamma{gamma_label}_{timestamp}_metadata.json'
        )
        with open(meta_path, 'w') as f_out:
            json.dump(metadata, f_out, indent=2)

        results.append({
            'C': C,
            'gamma': gamma_label,
            'accuracy':       round(accuracy, 4),
            'f1_weighted':    round(float(f1_avg), 4),
            'mean_wetland_f1': round(mean_wet_f1, 4),
            'train_secs':     round(train_sec, 1),
            'model_file':     model_filename,
        })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("GRID SEARCH SUMMARY — SVM RBF Wetland-Only  (sorted by mean_wetland_f1 desc)")
print(f"{'='*70}")
results.sort(key=lambda x: x['mean_wetland_f1'], reverse=True)
print(f"{'C':>6}  {'gamma':>7}  {'accuracy':>9}  {'wt_f1':>7}  {'mean_wet_f1':>11}  {'secs':>7}")
print(f"{'-'*6}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*11}  {'-'*7}")
for r in results:
    print(f"{r['C']:>6}  {r['gamma']:>7}  {r['accuracy']:>9.4f}  "
          f"{r['f1_weighted']:>7.4f}  {r['mean_wetland_f1']:>11.4f}  {r['train_secs']:>7.1f}s")

print(f"\nBest model (highest Mean Wetland F1): C={results[0]['C']}, gamma={results[0]['gamma']}")
print(f"  -> {results[0]['model_file']}")

summary_path = os.path.join(
    SCRIPT_DIR,
    f'grid_search_summary_wetland_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
)
with open(summary_path, 'w') as f_out:
    json.dump(results, f_out, indent=2)
print(f"Summary saved: {summary_path}")

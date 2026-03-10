"""
GRID SEARCH — SVM RBF Background vs Wetland (Stage 1)
=======================================================
Tunes: C x gamma (binary classification: 0=Background, 1=Wetland)
Uses cuML SVC (GPU) on Colab — falls back to sklearn SVC on CPU.

Grid: C in [1, 10]  x  gamma in [0.001, 'scale']  → 4 runs

Truth-source class mapping:
  0 = Background
  1 = Fen (Graminoid)
  2 = Fen (Woody)
  3 = Marsh
  4 = Shallow Open Water
  5 = Swamp
  (Stage 1 collapses 1-5 → 1 "Wetland")
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
from sklearn.preprocessing import StandardScaler

# ── GPU / CPU backend ─────────────────────────────────────────────────────────
try:
    from cuml.svm import SVC
    BACKEND = "cuML (GPU)"
    USE_CUML = True
except ImportError:
    from sklearn.svm import SVC
    BACKEND = "sklearn (CPU)"
    USE_CUML = False

print(f"SVM backend: {BACKEND}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Update this path if running on Colab ──────────────────────────────────────
DATA_PATH = os.path.join(
    SCRIPT_DIR, '..', '..', 'random_forest_all',
    'random_forest_spatial_middle', 'wetland_dataset_middle_split.npz'
)

# ── Hyperparameter grid ───────────────────────────────────────────────────────
C_OPTIONS     = [1.0, 10.0]
GAMMA_OPTIONS = [0.001, 'scale']

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

# ── Scale features (required for SVM) ────────────────────────────────────────
print("Fitting StandardScaler on training data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw.astype(np.float32))
X_test  = scaler.transform(X_test_raw.astype(np.float32))
scaler_path = os.path.join(SCRIPT_DIR, 'svm_rbf_bg_scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved: {scaler_path}\n")

# ── Binary labels ─────────────────────────────────────────────────────────────
y_train = (y_train_raw != 0).astype(np.int32)
y_test  = (y_test_raw  != 0).astype(np.int32)
labels  = [0, 1]

print(f"Binary class distribution (train):")
unique, counts = np.unique(y_train, return_counts=True)
for cls, cnt in zip(unique, counts):
    name = "Background" if cls == 0 else "Wetland"
    print(f"  Class {cls} ({name}): {cnt:,} ({cnt/len(y_train)*100:.1f}%)")
print()

# ── Grid search ───────────────────────────────────────────────────────────────
results  = []
total    = len(C_OPTIONS) * len(GAMMA_OPTIONS)
run_num  = 0

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
            class_weight='balanced',
            probability=True,
        )
        # cuML does not support probability=True natively — remove if cuML
        if USE_CUML:
            svm_kwargs.pop('probability')

        model = SVC(**svm_kwargs)
        model.fit(X_train, y_train)

        t_end     = datetime.now()
        train_sec = (t_end - t_start).total_seconds()

        y_pred = model.predict(X_test)
        if USE_CUML:
            import cupy as cp
            y_pred = cp.asnumpy(y_pred) if hasattr(y_pred, 'get') else np.array(y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average=None, zero_division=0
        )
        prec_avg, rec_avg, f1_avg, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average='weighted', zero_division=0
        )
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
        accuracy    = accuracy_score(y_test, y_pred)
        f1_bg      = float(f1[0])
        f1_wetland = float(f1[1])

        print(f"  Accuracy:       {accuracy:.4f}")
        print(f"  Weighted F1:    {f1_avg:.4f}")
        print(f"  Background F1:  {f1_bg:.4f}")
        print(f"  Wetland F1:     {f1_wetland:.4f}")
        print(f"  Train time:     {train_sec:.1f}s")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'svm_rbf_bg_C{C}_gamma{gamma_label}_{timestamp}.pkl'
        model_path = os.path.join(SCRIPT_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  Model saved: {model_filename}")

        metadata = {
            'timestamp': timestamp,
            'trained_datetime': t_end.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_stage': 'Stage 1 — binary background/wetland classification (grid search)',
            'backend': BACKEND,
            'split_method': 'middle_row_band',
            'test_row_min': test_row_min,
            'test_row_max': test_row_max,
            'overall_metrics': {
                'accuracy':           float(accuracy),
                'precision_weighted': float(prec_avg),
                'recall_weighted':    float(rec_avg),
                'f1_weighted':        float(f1_avg),
                'f1_background':      f1_bg,
                'f1_wetland':         f1_wetland,
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
                'kernel':       'rbf',
                'C':            C,
                'gamma':        gamma_label,
                'class_weight': 'balanced',
                'feature_scaling': 'StandardScaler',
            },
            'dataset': {
                'source':    DATA_PATH,
                'n_train':   int(X_train.shape[0]),
                'n_test':    int(X_test.shape[0]),
                'n_features': int(X_train.shape[1]),
            },
            'train_seconds': train_sec,
            'model_file': model_filename,
            'scaler_file': 'svm_rbf_bg_scaler.pkl',
        }

        meta_path = os.path.join(
            SCRIPT_DIR,
            f'svm_rbf_bg_C{C}_gamma{gamma_label}_{timestamp}_metadata.json'
        )
        with open(meta_path, 'w') as f_out:
            json.dump(metadata, f_out, indent=2)

        results.append({
            'C': C,
            'gamma': gamma_label,
            'accuracy':     round(accuracy, 4),
            'f1_weighted':  round(float(f1_avg), 4),
            'f1_background': round(f1_bg, 4),
            'f1_wetland':   round(f1_wetland, 4),
            'train_secs':   round(train_sec, 1),
            'model_file':   model_filename,
        })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("GRID SEARCH SUMMARY — SVM RBF Background  (sorted by f1_wetland desc)")
print(f"{'='*70}")
results.sort(key=lambda x: x['f1_wetland'], reverse=True)
print(f"{'C':>6}  {'gamma':>7}  {'accuracy':>9}  {'wt_f1':>7}  {'bg_f1':>7}  {'wet_f1':>7}  {'secs':>7}")
print(f"{'-'*6}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
for r in results:
    print(f"{r['C']:>6}  {r['gamma']:>7}  {r['accuracy']:>9.4f}  "
          f"{r['f1_weighted']:>7.4f}  {r['f1_background']:>7.4f}  "
          f"{r['f1_wetland']:>7.4f}  {r['train_secs']:>7.1f}s")

print(f"\nBest model (highest Wetland F1): C={results[0]['C']}, gamma={results[0]['gamma']}")
print(f"  -> {results[0]['model_file']}")

summary_path = os.path.join(
    SCRIPT_DIR,
    f'grid_search_summary_bg_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
)
with open(summary_path, 'w') as f_out:
    json.dump(results, f_out, indent=2)
print(f"Summary saved: {summary_path}")

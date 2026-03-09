from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# GRID SEARCH — RF Wetland-Only (Stage 2)
# Tunes: n_estimators x max_depth
# No StandardScaler (RF is scale-invariant)
# ======================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'random_forest_spatial_middle', 'wetland_dataset_middle_split.npz')

# ---- Hyperparameter grid ----
N_ESTIMATORS_OPTIONS = [100, 200, 300]
MAX_DEPTH_OPTIONS    = [25, 35, None]   # None = unlimited depth
MIN_SAMPLES_LEAF     = 20               # kept fixed
CLASS1_DAMPEN        = 0.4              # kept fixed

# ======================================
# LOAD & FILTER DATA (done once)
# ======================================
data = np.load(DATA_PATH)
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']
y_test  = data['y_test']
test_row_min = int(data['test_row_min'])
test_row_max = int(data['test_row_max'])
data.close()

print(f"Loaded: {DATA_PATH}")
print(f"Before filtering — Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

train_mask = y_train != 0
test_mask  = y_test  != 0
X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test  = X_test[test_mask]
y_test  = y_test[test_mask]

print(f"After filtering Class 0 — Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}\n")

# ======================================
# CLASS WEIGHTS (computed once, shared)
# ======================================
unique_classes, class_counts = np.unique(y_train, return_counts=True)
n_total   = len(y_train)
n_classes = len(unique_classes)

class_weight_dict = {
    int(cls): float(n_total / (n_classes * count))
    for cls, count in zip(unique_classes, class_counts)
}
class_weight_dict[1] = class_weight_dict[1] * CLASS1_DAMPEN

labels = sorted(class_weight_dict.keys())

print("Class weights (shared across all runs):")
for cls, w in class_weight_dict.items():
    count = class_counts[list(unique_classes).index(cls)]
    print(f"  Class {cls}: weight={w:.4f}  (n={count:,})")
print()

# ======================================
# GRID SEARCH LOOP
# ======================================
results = []
total_runs = len(N_ESTIMATORS_OPTIONS) * len(MAX_DEPTH_OPTIONS)
run_num = 0

for n_est in N_ESTIMATORS_OPTIONS:
    for max_d in MAX_DEPTH_OPTIONS:
        run_num += 1
        depth_label = str(max_d) if max_d is not None else 'None'
        print(f"{'='*60}")
        print(f"RUN {run_num}/{total_runs} — n_estimators={n_est}, max_depth={depth_label}")
        print(f"{'='*60}")

        t_start = datetime.now()

        rf_model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=42,
            class_weight=class_weight_dict,
            verbose=0,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

        t_end = datetime.now()
        train_secs = (t_end - t_start).total_seconds()

        y_pred = rf_model.predict(X_test)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average=None
        )
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, average='weighted'
        )
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
        accuracy = accuracy_score(y_test, y_pred)
        mean_wetland_f1 = float(np.mean(f1))

        print(f"  Accuracy:         {accuracy:.4f}")
        print(f"  Weighted F1:      {f1_avg:.4f}")
        print(f"  Mean Wetland F1:  {mean_wetland_f1:.4f}")
        print(f"  Per-class F1:     { {labels[i]: round(f1[i], 3) for i in range(len(labels))} }")
        print(f"  Train time:       {train_secs:.1f}s\n")

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename    = f'rf_wo_grid_est{n_est}_depth{depth_label}_{timestamp}.pkl'
        metadata_filename = f'rf_wo_grid_est{n_est}_depth{depth_label}_{timestamp}_metadata.json'

        joblib.dump(rf_model, os.path.join(SCRIPT_DIR, model_filename))

        metadata = {
            'timestamp': timestamp,
            'trained_datetime': t_end.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_stage': 'Stage 2 — wetland classification (grid search)',
            'split_method': 'middle_row_band',
            'test_row_min': test_row_min,
            'test_row_max': test_row_max,
            'classes': labels,
            'overall_metrics': {
                'accuracy':           float(accuracy),
                'precision_weighted': float(precision_avg),
                'recall_weighted':    float(recall_avg),
                'f1_weighted':        float(f1_avg),
                'mean_wetland_f1':    mean_wetland_f1,
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
                'n_estimators':         n_est,
                'max_depth':            max_d,
                'min_samples_leaf':     MIN_SAMPLES_LEAF,
                'feature_scaling':      'none',
                'class_weight':         'recalculated_over_classes_1_to_5',
                'class1_dampen_factor': CLASS1_DAMPEN,
                'n_jobs':               -1,
                'random_state':         42,
            },
            'dataset': {
                'source':           '../random_forest_spatial_middle/wetland_dataset_middle_split.npz',
                'class_0_filtered': True,
                'n_train':          int(X_train.shape[0]),
                'n_test':           int(X_test.shape[0]),
                'n_features':       int(X_train.shape[1]),
            },
            'class_weights':  {str(k): float(v) for k, v in class_weight_dict.items()},
            'train_seconds':  train_secs,
        }

        with open(os.path.join(SCRIPT_DIR, metadata_filename), 'w') as f:
            json.dump(metadata, f, indent=2)

        results.append({
            'n_estimators':    n_est,
            'max_depth':       depth_label,
            'accuracy':        round(accuracy, 4),
            'f1_weighted':     round(f1_avg, 4),
            'mean_wetland_f1': round(mean_wetland_f1, 4),
            'train_secs':      round(train_secs, 1),
            'model_file':      model_filename,
        })

# ======================================
# SUMMARY TABLE
# ======================================
print(f"\n{'='*60}")
print("GRID SEARCH SUMMARY  (sorted by mean_wetland_f1 desc)")
print(f"{'='*60}")
results.sort(key=lambda x: x['mean_wetland_f1'], reverse=True)
print(f"{'n_est':>6}  {'depth':>6}  {'accuracy':>9}  {'wt_f1':>7}  {'wetland_f1':>11}  {'secs':>6}")
print(f"{'-'*6}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*11}  {'-'*6}")
for r in results:
    print(f"{r['n_estimators']:>6}  {r['max_depth']:>6}  {r['accuracy']:>9.4f}  "
          f"{r['f1_weighted']:>7.4f}  {r['mean_wetland_f1']:>11.4f}  {r['train_secs']:>6.1f}s")

# Save summary JSON
summary_path = os.path.join(SCRIPT_DIR, f'grid_search_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSummary saved to: {summary_path}")

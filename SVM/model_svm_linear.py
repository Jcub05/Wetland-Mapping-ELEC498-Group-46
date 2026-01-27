from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# LOAD THE DATA
# ======================================
print("Loading data...")
try:
    # Use path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'wetland_dataset_1.5M_4Training.npz')
    
    data = np.load(data_path)
    X = data['X'] # 1.5m by 64
    y = data['y'] # 1.5m by 1 (the class label)
    
    # Calculated Class weights (for normalization)
    class_weights = data['class_weights']
    # Convert to dict
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    data.close()
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset not found at {data_path}")
    exit(1)

# ======================================
# TRAIN + TEST THE MODEL
# ======================================
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVC pipeline
# 1. StandardScaler: Critical for SVM convergence and performance
# 2. LinearSVC: Faster linear SVM implementation for large datasets
print("Initializing LinearSVC model...")
# dual=False is preferred when n_samples > n_features
svm_pipeline = make_pipeline(
    StandardScaler(),
    LinearSVC(
        class_weight=class_weight_dict,
        random_state=42,
        dual=False,
        max_iter=1000,
        verbose=1
    )
)

# Training command
print(f"Starting training on {len(X_train)} samples...")
start_time = datetime.now()
svm_pipeline.fit(X_train, y_train)
end_time = datetime.now()
print(f"Training completed in {end_time - start_time}")

# Testing command
print("Predicting on test set...")
y_pred = svm_pipeline.predict(X_test)

# ====================================== 
# EVALUATE THE MODEL
# ====================================== 
print("Evaluating model...")
# Calculate detailed metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"MODEL EVALUATION RESULTS (LinearSVC)")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (weighted): {precision_avg:.4f}")
print(f"Recall (weighted): {recall_avg:.4f}")
print(f"F1-Score (weighted): {f1_avg:.4f}")

# Detailed per-class metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(len(class_weight_dict))]))
# Confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)


# ======================================
# SAVE THE MODEL, TIME STAMP AND VERSION
# ======================================
print(f"\n{'='*60}")
print("SAVING MODEL")
print(f"{'='*60}")

version = 1
while os.path.exists(f'svm_linear_wetland_model_v{version}.pkl'):
    version += 1

# Create filename with version + timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'svm_linear_wetland_model_v{version}_{timestamp}.pkl'
metadata_filename = f'svm_linear_wetland_model_v{version}_{timestamp}_metadata.json'

# Save the model
joblib.dump(svm_pipeline, model_filename)

# Save the metadata
metadata = {
    'version': version,
    'timestamp': timestamp,
    'model_type': 'LinearSVC',
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_duration': str(end_time - start_time),
    
    # Overall metrics
    'overall_metrics': {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision_avg),
        'recall_weighted': float(recall_avg),
        'f1_weighted': float(f1_avg),
    },
    
    # Per-class metrics
    'per_class_metrics': {
        str(i): {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
        for i in range(len(class_weight_dict))
    },
    
    # Confusion matrix
    'confusion_matrix': conf_matrix.tolist(),
    
    'hyperparameters': {
        'dual': False,
        'class_weight': 'custom',
        'random_state': 42,
        'max_iter': 1000,
        'standard_scaler': True
    },
    'dataset': {
        'source': '../wetland_dataset_1.5M_4Training.npz',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
    },
    'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()},
}

with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print("MODEL SAVED")
print(f"{'='*60}")
print(f"Model saved to: {model_filename}")
print(f"Metadata saved to: {metadata_filename}")
print(f"{'='*60}")
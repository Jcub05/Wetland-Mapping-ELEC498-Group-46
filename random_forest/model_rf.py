from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# LOAD THE DATA
# ======================================
data = np.load('../wetland_dataset_1.5M_4Training.npz')
X = data['X'] # 1.5m by 64
y = data['y'] # 1.5m by 1 (the class label)

# Calculated Class weights (for normalization)
class_weights = data['class_weights']
# Convert to dict
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
data.close()

# ======================================
# TRAIN + TEST THE MODEL
# ======================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight=class_weight_dict,
    verbose = 2,
    n_jobs = -1,
    # Can add more to this if we want to customize more, like max depth, min samples at leaf (default is 1, could overfit)
    )

# Training command
rf_model.fit(X_train, y_train)

#Testing command
y_pred = rf_model.predict(X_test)

# ====================================== 
# EVALUATE THE MODEL
# ====================================== 
# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"MODEL EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed per-class metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(6)]))
# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ======================================
# SAVE THE MODEL, TIME STAMP AND VERSION
# ======================================
print(f"\n{'='*60}")
print("SAVING MODEL")
print(f"{'='*60}")

version = 1
while os.path.exists(f'rf_wetland_model_v{version}.pkl'):
    version += 1

# Create filename with version + timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'rf_wetland_model_v{version}_{timestamp}.pkl'
metadata_filename = f'rf_wetland_model_v{version}_{timestamp}_metadata.json'

# Save the model
joblib.dump(rf_model, model_filename)

# Save the metadata
metadata = {
    'version': version,
    'timestamp': timestamp,
    'trained_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'accuracy': float(accuracy),
    'hyperparameters': {
        'n_estimators': 100,
        'class_weight': 'custom',
        'n_jobs': -1,
        'random_state': 42,
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


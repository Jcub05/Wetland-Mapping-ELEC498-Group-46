from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


data = np.load('../wetland_dataset_1.5M.npz')
X = data['X'] # 1.5m by 64
y = data['y'] # 1.5m by 1 (the class label)

class_weights = data['class_weights']

data.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

rf_model.fit(X_train, y_train)

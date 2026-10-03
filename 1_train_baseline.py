# 1_train_baseline.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Generate synthetic baseline data
X, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_informative=3, 
    random_state=42
)

# Map generic features to business logic
feature_names = ["Income", "Age", "Credit_Score", "Debt_Ratio", "Years_Employed"]
reference_data = pd.DataFrame(X, columns=feature_names)
reference_data['target'] = y

# 2. Train a baseline Random Forest model
X_train = reference_data.drop('target', axis=1)
y_train = reference_data['target']

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. Save the model and reference dataset for monitoring
joblib.dump(model, 'baseline_model.pkl')
reference_data.to_csv('reference_data.csv', index=False)

print("✅ Baseline model trained. 'reference_data.csv' and 'baseline_model.pkl' saved.")
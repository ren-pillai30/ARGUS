# 3_detect_anomalies.py
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load baseline and production data
reference_data = pd.read_csv('reference_data.csv')
current_data = pd.read_csv('current_data.csv')

# Drop the target variable since anomaly detection is unsupervised
X_ref = reference_data.drop('target', axis=1)
X_curr = current_data.drop('target', axis=1)

# 2. Train the Isolation Forest on the "normal" reference baseline
# contamination=0.05 assumes a 5% natural outlier rate in the baseline
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_ref)

# 3. Predict anomalies on the drifted production data
# IsolationForest outputs 1 for normal inliers, and -1 for anomalies
predictions = iso_forest.predict(X_curr)

# Map numeric predictions to booleans for our dashboard later
current_data['is_anomaly'] = [True if p == -1 else False for p in predictions]

# 4. Calculate metrics and save
anomaly_count = current_data['is_anomaly'].sum()
total_count = len(current_data)

print("🔍 Anomaly Detection Complete.")
print(f"Flagged {anomaly_count} extreme anomalies out of {total_count} records ({(anomaly_count/total_count)*100:.2f}%).")

# Save the dataset with anomaly flags appended
current_data.to_csv('current_data_with_anomalies.csv', index=False)
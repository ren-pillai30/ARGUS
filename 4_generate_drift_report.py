# 4_generate_drift_report.py
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Load the baseline and production data
reference_data = pd.read_csv('reference_data.csv')
current_data = pd.read_csv('current_data.csv')

# Drop the 'is_anomaly' flag generated in Phase 3 so we only compare the original schema
if 'is_anomaly' in current_data.columns:
    current_data = current_data.drop('is_anomaly', axis=1)

# 2. Initialize the Evidently Report with the DataDriftPreset
drift_report = Report(metrics=[DataDriftPreset()])

# 3. Run the statistical tests comparing the current data to the reference baseline
# Note: In 0.7+, we pass the datasets and it returns a Snapshot object
report_result = drift_report.run(current_data=current_data, reference_data=reference_data)

# 4. Export the interactive report as an HTML file from the Snapshot object
report_result.save_html('drift_report.html')

print("📊 Evidently AI Data Drift Report generated and saved as 'drift_report.html'.")
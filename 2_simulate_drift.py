# 2_simulate_drift.py
import pandas as pd
import numpy as np

# 1. Load the original reference baseline
reference_data = pd.read_csv('reference_data.csv')

# 2. Create the "current" production dataset
current_data = reference_data.copy()

# 3. Inject Artificial Drift
# Scenario: A sudden macroeconomic shift increases nominal Income 
# but severely impacts Credit Scores across the board.

# Shift 1: Increase Income artificially by 50%
current_data['Income'] = current_data['Income'] * 1.5

# Shift 2: Decrease Credit Scores and introduce heavy random noise
noise = np.random.normal(loc=0, scale=2.5, size=len(current_data))
current_data['Credit_Score'] = current_data['Credit_Score'] - 3.0 + noise

# 4. Save the drifted production dataset
current_data.to_csv('current_data.csv', index=False)

print("⚠️ Production data drift simulated. 'current_data.csv' saved.")
# shadow_evaluator.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

class ShadowModelEvaluator:
    def __init__(self, primary_model_path='baseline_model.pkl'):
        # Load the stable production model
        self.primary_model = joblib.load(primary_model_path)

    def train_candidate_model(self, current_data_path='current_data.csv'):
        """Trains a candidate (canary) model on the incoming drifted production data."""
        df = pd.read_csv(current_data_path)
        
        # If target column is missing in current data, fallback to using the model's pseudo-labels or reference target
        if 'target' not in df.columns:
            ref_df = pd.read_csv('reference_data.csv')
            df['target'] = self.primary_model.predict(df.drop(columns=['is_anomaly'], errors='ignore'))

        X = df.drop(columns=['target', 'is_anomaly'], errors='ignore')
        y = df['target']

        candidate = RandomForestClassifier(random_state=42)
        candidate.fit(X, y)
        return candidate

    def evaluate_canary(self, current_data_path='current_data.csv', divergence_threshold=0.15):
        """Routes live batch data to both models in shadow mode and evaluates safety."""
        df = pd.read_csv(current_data_path)
        X = df.drop(columns=['target', 'is_anomaly'], errors='ignore')

        # 1. Get prediction probabilities from both primary and candidate models
        primary_probs = self.primary_model.predict_proba(X)[:, 1]
        
        candidate = self.train_candidate_model(current_data_path)
        candidate_probs = candidate.predict_proba(X)[:, 1]

        # 2. Calculate Mean Absolute Probability Divergence
        prob_divergence = np.mean(np.abs(primary_probs - candidate_probs))

        # 3. Canary Promotion Gate
        promoted = prob_divergence <= divergence_threshold

        return {
            "mean_probability_divergence": round(float(prob_divergence), 4),
            "divergence_threshold": divergence_threshold,
            "canary_promoted": promoted,
            "action": "PROMOTE_CANDIDATE_TO_PRODUCTION" if promoted else "REJECT_CANDIDATE_ROLLBACK"
        }

if __name__ == "__main__":
    print("🚦 Initializing Shadow Model & Canary Evaluator...")
    evaluator = ShadowModelEvaluator()
    results = evaluator.evaluate_canary()

    print("\n📊 Shadow Canary Evaluation Results:")
    for k, v in results.items():
        print(f" - {k}: {v}")

    if results['canary_promoted']:
        print("\n✅ Canary passed safety checks. Candidate model promoted safely to production.")
    else:
        print("\n❌ Canary validation failed! Candidate model rejected due to high probability divergence.")
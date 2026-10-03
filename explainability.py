# explainability.py
import pandas as pd
import numpy as np
import joblib
import shap
import warnings
from scipy.stats import kendalltau

# Suppress shap warnings for cleaner terminal output
warnings.filterwarnings("ignore")

class ExplainableDriftEngine:
    def __init__(self, model_path='baseline_model.pkl'):
        # Load the baseline model
        self.model = joblib.load(model_path)
        # Initialize the TreeExplainer for Random Forest
        self.explainer = shap.TreeExplainer(self.model)

    def calculate_feature_importance(self, df: pd.DataFrame, sample_size=200) -> pd.Series:
        """Calculates mean absolute SHAP values to rank feature importance robustly."""
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(sample_df)
        
        # Handle different SHAP output formats (List, 3D ndarray, or 2D ndarray)
        if isinstance(shap_values, list):
            # Binary classification list: index 1 is the positive class
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 3D array: (samples, features, classes) -> take positive class index 1
            vals = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        else:
            vals = shap_values

        # Ensure 1D mean absolute impact per feature
        mean_abs_shap = np.abs(vals).mean(axis=0)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=1)
            
        return pd.Series(mean_abs_shap, index=df.columns)

    def measure_attribution_drift(self, ref_df: pd.DataFrame, curr_df: pd.DataFrame):
        """Measures if the model's decision-making logic has shifted using Kendall's Tau."""
        ref_importance = self.calculate_feature_importance(ref_df)
        curr_importance = self.calculate_feature_importance(curr_df)
        
        # Rank the features (1 is the most important feature)
        ref_ranks = ref_importance.rank(ascending=False)
        curr_ranks = curr_importance.rank(ascending=False)
        
        # Kendall's Tau measures correlation between the two rankings
        tau, p_value = kendalltau(ref_ranks, curr_ranks)
        
        return {
            "kendall_tau_score": round(tau, 4),
            "reference_top_feature": ref_importance.idxmax(),
            "current_top_feature": curr_importance.idxmax(),
            "logic_shift_detected": tau < 0.80  # Alert if correlation drops below 80%
        }

if __name__ == "__main__":
    print("🧠 Initializing SHAP Explainability Engine...")
    
    ref = pd.read_csv('reference_data.csv').drop(columns=['target'], errors='ignore')
    curr = pd.read_csv('current_data.csv').drop(columns=['target', 'is_anomaly'], errors='ignore')
    
    engine = ExplainableDriftEngine()
    metrics = engine.measure_attribution_drift(ref, curr)
    
    print("\n📊 Feature Attribution Drift Results:")
    for key, value in metrics.items():
        print(f" - {key}: {value}")
        
    if metrics['logic_shift_detected']:
        print("\n⚠️ ALERT: The features driving model decisions have significantly shifted! (Concept Drift)")
    else:
        print("\n✅ Model decision logic remains highly correlated despite incoming data shifts.")
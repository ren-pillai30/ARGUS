# universal_engine.py
import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings("ignore")

class UniversalMLOpsEngine:
    """Dynamically trains and monitors models for any user-uploaded tabular dataset."""
    
    @staticmethod
    def validate_and_preprocess(df: pd.DataFrame):
        """Extracts numeric features and handles missing values."""
        numeric_df = df.select_dtypes(include=['number']).dropna()
        return numeric_df

    @staticmethod
    def train_dynamic_baseline(df: pd.DataFrame):
        """Trains a baseline model on the fly using the last numeric column as a synthetic target if missing."""
        numeric_df = UniversalMLOpsEngine.validate_and_preprocess(df)
        if numeric_df.shape[1] < 2:
            raise ValueError("Dataset must contain at least 2 numerical columns.")
            
        # If no explicit target column exists, create a binary pseudo-target based on median split of the last column
        X = numeric_df.iloc[:, :-1]
        y = (numeric_df.iloc[:, -1] > numeric_df.iloc[:, -1].median()).astype(int)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        return model, X

    @staticmethod
    def run_dynamic_anomaly_scan(df: pd.DataFrame, contamination=0.05):
        """Runs Isolation Forest anomaly detection on any custom dataset."""
        X = UniversalMLOpsEngine.validate_and_preprocess(df)
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X)
        
        result_df = X.copy()
        result_df['is_anomaly'] = (preds == -1)
        return result_df

    @staticmethod
    def compute_dynamic_shap(model, X_ref, X_curr):
        """Calculates SHAP attribution drift using Kendall's Tau for custom models."""
        explainer = shap.TreeExplainer(model)
        
        def get_shap_ranking(data):
            sample = data.sample(n=min(100, len(data)), random_state=42)
            shap_vals = explainer.shap_values(sample)
            if isinstance(shap_vals, list):
                vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                vals = shap_vals[:, :, 1] if shap_vals.shape[2] > 1 else shap_vals[:, :, 0]
            else:
                vals = shap_vals
            mean_abs = np.abs(vals).mean(axis=0)
            if mean_abs.ndim > 1:
                mean_abs = mean_abs.mean(axis=1)
            return pd.Series(mean_abs, index=data.columns).rank(ascending=False)

        ref_ranks = get_shap_ranking(X_ref)
        curr_ranks = get_shap_ranking(X_curr)
        
        tau, _ = kendalltau(ref_ranks, curr_ranks)
        return round(tau, 4)
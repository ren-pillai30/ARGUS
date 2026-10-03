# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sqlite3
import numpy as np
from universal_engine import UniversalMLOpsEngine
from triage_db import init_triage_db, fetch_triage_queue, update_triage_status

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="ARGUS | Universal MLOps Platform", page_icon="🛡️")

# 2. Sleek Muted Dark Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #C9D1D9; }
    div[data-testid="metric-container"] {
        background-color: #161B22; border: 1px solid #30363D; padding: 12px; border-radius: 8px;
    }
    h1, h2, h3 { color: #F0F6FC !important; font-family: 'Inter', sans-serif; }
    p, label { color: #8B949E !important; }
    hr { border-bottom: 1px solid #21262D; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Universal Data Ingestion & Controls (BYOD)
st.sidebar.title("📁 Workspace Data Hub")
st.sidebar.markdown("Upload **any** tabular dataset (CSV) to analyze model behavior, data drift, and anomalies in real-time.")

uploaded_file = st.sidebar.file_uploader("Upload Custom Production CSV", type=["csv"])

# Global Anomaly Contamination Control
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Pipeline Settings")
contamination = st.sidebar.slider("Anomaly Contamination Rate", 0.01, 0.20, 0.05, 0.01)

# Fallback to default demo datasets if no file is uploaded
if uploaded_file is not None:
    active_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded custom dataset: {uploaded_file.name} ({len(active_df):,} rows)")
else:
    try:
        active_df = pd.read_csv('current_data.csv')
        st.sidebar.info("Using default demo dataset. Upload your own CSV above anytime!")
    except FileNotFoundError:
        st.sidebar.error("No dataset found. Please upload a CSV.")
        st.stop()

# Header
st.title("🛡️ ARGUS Universal MLOps Command Center")
st.markdown("Autonomous monitoring, explainable feature attribution, shadow canary validation, and multi-industry anomaly triage.")
st.divider()

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Universal Telemetry & Anomalies", 
    "🧠 Explainable Drift (SHAP)", 
    "🧪 Chaos & Stress Workbench", 
    "👥 HITL Triage Queue",
    "⚡ Live Interactive Scoring"
])

# ================= TAB 1: TELEMETRY & ANOMALIES =================
with tab1:
    st.subheader("Dynamic Outlier & Anomaly Detection")
    st.markdown("Scanning your active dataset for point-in-time anomalies using dynamic Isolation Forest profiling.")
    
    try:
        scanned_df = UniversalMLOpsEngine.run_dynamic_anomaly_scan(active_df, contamination)
        total_recs = len(scanned_df)
        anomaly_count = int(scanned_df['is_anomaly'].sum())
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records Analyzed", f"{total_recs:,}")
        c2.metric("Anomalies Flagged", f"{anomaly_count:,}", f"{(anomaly_count/total_recs)*100:.1f}% rate", delta_color="inverse")
        c3.metric("Dataset Health", "WARNING" if anomaly_count > (total_recs * 0.1) else "STABLE")
        
        st.markdown("#### Flagged Outlier Records")
        anomalies_only = scanned_df[scanned_df['is_anomaly']]
        st.dataframe(anomalies_only, use_container_width=True)
    except Exception as e:
        st.error(f"Error running anomaly scan: {e}")

# ================= TAB 2: EXPLAINABILITY (SHAP) =================
with tab2:
    st.subheader("Explainable AI: Feature Attribution Drift")
    st.markdown("Measures whether your model's core decision logic remains stable using **Kendall's Tau** correlation (computed on clean, non-anomalous records).")
    
    if st.button("Compute Dynamic SHAP Attribution"):
        with st.spinner("Filtering out anomalies and computing SHAP values..."):
            try:
                # 1. Isolate clean data by removing anomalies based on current contamination rate
                scanned_for_clean = UniversalMLOpsEngine.run_dynamic_anomaly_scan(active_df, contamination)
                clean_df = scanned_for_clean[scanned_for_clean['is_anomaly'] == False].drop(columns=['is_anomaly'])
                
                if len(clean_df) < 10:
                    st.error("⚠️ Contamination rate is too high! Too few clean records remain to compute stable SHAP values. Lower the slider in the sidebar.")
                else:
                    # 2. Split clean data into reference and current halves for attribution drift
                    mid = len(clean_df) // 2
                    ref_half = clean_df.iloc[:mid]
                    curr_half = clean_df.iloc[mid:]
                    
                    model, X_r = UniversalMLOpsEngine.train_dynamic_baseline(ref_half)
                    _, X_c = UniversalMLOpsEngine.train_dynamic_baseline(curr_half)
                    
                    tau = UniversalMLOpsEngine.compute_dynamic_shap(model, X_r, X_c)
                    
                    s1, s2 = st.columns(2)
                    s1.metric("Kendall's Tau Attribution Score", tau)
                    s2.metric("Logic Stability Status", "STABLE" if tau >= 0.70 else "DRIFT DETECTED")
                    
                    if tau < 0.70:
                        st.error("⚠️ Feature importance ranking has shifted significantly between clean dataset splits.")
                    else:
                        st.success("✅ Model decision logic is stable across clean dataset splits.")
            except Exception as e:
                st.error(f"Error computing SHAP attribution: {e}")
    else:
        st.info("Click the button above to run explainability analysis on your active dataset.")

# ================= TAB 3: CHAOS WORKBENCH =================
with tab3:
    st.subheader("Data Pipeline Chaos Engineering Workbench")
    st.markdown("Inject artificial Gaussian noise or scale numerical columns to stress-test your monitoring pipelines.")
    
    numeric_cols = active_df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        target_col = st.selectbox("Select Column to Perturb", numeric_cols)
        noise_mult = st.slider("Multiplier / Noise Scale", 0.1, 5.0, 1.5, 0.1)
        
        if st.button("Inject Chaos & Preview"):
            perturbed_df = active_df.copy()
            perturbed_df[target_col] = perturbed_df[target_col] * noise_mult
            st.success(f"Successfully scaled column `{target_col}` by factor of {noise_mult}!")
            st.dataframe(perturbed_df.head(5), use_container_width=True)
    else:
        st.warning("Active dataset must contain numeric columns for chaos injection.")

# ================= TAB 4: HITL TRIAGE QUEUE =================
with tab4:
    st.subheader("Human-in-the-Loop (HITL) Triage Audit Store")
    st.markdown("Review and disposition flagged anomalies into SQLite audit compliance logs.")
    
    init_triage_db()
    triage_df = fetch_triage_queue()
    if not triage_df.empty:
        st.dataframe(triage_df, use_container_width=True, hide_index=True)
        
        rec_id = st.selectbox("Select Record ID to Review", triage_df['id'].tolist())
        disposition = st.selectbox("Assign Disposition", ["DATA_CORRUPTION", "VALID_EDGE_CASE"])
        notes = st.text_input("Reviewer Notes", "Inspected custom dataset anomaly.")
        
        if st.button("Commit Review Decision"):
            update_triage_status(rec_id, disposition, notes)
            st.success(f"Record {rec_id} updated to {disposition}.")
            st.rerun()
    else:
        st.info("Triage queue is empty. Run an anomaly scan or log custom records to populate.")

# ================= TAB 5: LIVE INTERACTIVE SCORING =================
with tab5:
    st.subheader("⚡ Dynamic Live Inference & Scoring Portal")
    st.markdown("Enter values for your active dataset's numerical columns to score incoming instances in real-time.")
    
    try:
        model, X_features = UniversalMLOpsEngine.train_dynamic_baseline(active_df)
        user_inputs = {}
        
        col_inputs, col_output = st.columns(2)
        with col_inputs:
            st.markdown("#### Input Feature Values")
            for col in X_features.columns:
                default_val = float(active_df[col].mean())
                user_inputs[col] = st.number_input(f"{col}", value=default_val)
                
            score_btn = st.button("Score Live Instance")
            
        with col_output:
            st.markdown("#### Prediction Result")
            if score_btn:
                input_df = pd.DataFrame([user_inputs])
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1]
                
                st.metric("Predicted Class / Output", str(prediction))
                st.metric("Confidence Probability", f"{proba * 100:.2f}%")
                st.success("✅ Live inference executed successfully against dynamic baseline.")
            else:
                st.info("Configure input parameters on the left and click score.")
    except Exception as e:
        st.warning(f"Unable to initialize live scoring for this dataset format: {e}")
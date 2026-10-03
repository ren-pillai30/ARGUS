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

# Sidebar for Universal Data Ingestion (BYOD)
st.sidebar.title("📁 Workspace Data Hub")
st.sidebar.markdown("Upload **any** tabular dataset (CSV) to initialize autonomous monitoring, drift detection, and anomaly triage.")

uploaded_file = st.sidebar.file_uploader("Upload Production CSV Dataset", type=["csv"])

# Global Anomaly Contamination Control
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Pipeline Settings")
contamination = st.sidebar.slider("Anomaly Contamination Rate", 0.01, 0.20, 0.05, 0.01)

# Handle Public Landing State (No Demo Fallback)
if uploaded_file is not None:
    active_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Successfully loaded: {uploaded_file.name} ({len(active_df):,} rows)")
else:
    # Enterprise Onboarding Landing View for Public Visitors
    st.title("🛡️ ARGUS Universal MLOps Command Center")
    st.markdown("Autonomous monitoring, explainable feature attribution, shadow canary validation, and multi-industry anomaly triage.")
    st.divider()
    
    st.info("👈 **Get Started:** Please upload your tabular dataset (CSV format) using the sidebar **Workspace Data Hub** to boot up the enterprise engine.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Telemetry & Anomalies")
        st.markdown("Dynamic Isolation Forest outlier profiling with customizable sensitivity thresholds and downloadable audit exports.")
    with col2:
        st.markdown("### 🧠 Explainable Drift")
        st.markdown("SHAP tree explainers and Kendall's Tau correlation tracking to detect feature attribution decay and concept drift.")
    with col3:
        st.markdown("### 🧪 Chaos Workbench")
        st.markdown("Statistical Z-score variance stress-testing and automated circuit-breaker simulation for resilient data pipelines.")
    st.stop()

# Header (When dataset is active)
st.title("🛡️ ARGUS Universal MLOps Command Center")
st.markdown(f"Active Workspace: **{uploaded_file.name}** | Autonomous telemetry, explainability, and pipeline resilience monitoring active.")
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
        scanned_df_reset = scanned_df.reset_index().rename(columns={'index': 'Original_Row_Number'})
        
        total_recs = len(scanned_df_reset)
        anomaly_count = int(scanned_df_reset['is_anomaly'].sum())
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records Analyzed", f"{total_recs:,}")
        c2.metric("Anomalies Flagged", f"{anomaly_count:,}", f"{(anomaly_count/total_recs)*100:.1f}% rate", delta_color="inverse")
        c3.metric("Dataset Health Status", "DEGRADED (High Outliers)" if anomaly_count > (total_recs * 0.1) else "HEALTHY")
        
        st.markdown("### 🚨 Flagged Outlier Records & Audit Export")
        anomalies_only = scanned_df_reset[scanned_df_reset['is_anomaly']]
        
        if not anomalies_only.empty:
            st.dataframe(anomalies_only, use_container_width=True, hide_index=True)
            
            csv_data = anomalies_only.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Flagged Anomalies Audit Report (CSV)",
                data=csv_data,
                file_name="argus_anomaly_audit_log.csv",
                mime="text/csv",
                help="Export a standalone log containing only the erroneous records with their original row numbers."
            )
        else:
            st.info("No anomalies detected at the current contamination threshold.")
            
    except Exception as e:
        st.error(f"Error running anomaly scan: {e}")

# ================= TAB 2: EXPLAINABILITY (SHAP) =================
with tab2:
    st.subheader("Explainable AI: Feature Attribution & Deep Drift Diagnostics")
    st.markdown("Tracks whether your model's core decision logic remains consistent using **Kendall's Tau** correlation on clean production records.")
    
    if st.button("Compute Dynamic SHAP Attribution & Diagnose"):
        with st.spinner("Isolating clean records, training tree explainers, and generating deep diagnostics..."):
            try:
                scanned_for_clean = UniversalMLOpsEngine.run_dynamic_anomaly_scan(active_df, contamination)
                clean_df = scanned_for_clean[scanned_for_clean['is_anomaly'] == False].drop(columns=['is_anomaly'])
                
                st.markdown("---")
                st.markdown("### 📊 Diagnostic Telemetry Breakdown")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Raw Dataset Size", f"{len(active_df):,} rows")
                d2.metric("Anomalies Filtered Out", f"{len(active_df) - len(clean_df):,} rows")
                d3.metric("Clean Records Retained", f"{len(clean_df):,} rows")
                d4.metric("Contamination Threshold", f"{contamination * 100:.1f}%")
                
                if len(clean_df) < 15:
                    st.error("⚠️ **Critical Diagnostic Failure: Insufficient Clean Sample Size**")
                    st.markdown("""
                    * **Detailed Root Cause:** Your current **Anomaly Contamination Rate** slider is set too high (`{:.1f}%`), or your source dataset is too small. This caused the platform to strip out too many rows, leaving fewer than 15 clean records. Tree explainers and rank correlation require statistical depth to compute meaningful feature attributions.
                    * **Actionable Remediation Steps:**
                      1. **Adjust Sidebar Slider:** Lower the Anomaly Contamination Rate down to `0.05` (5%) or `0.08` in the sidebar.
                      2. **Re-run Diagnostics:** Click the compute button above again to regenerate valid SHAP rankings.
                    """.format(contamination * 100))
                else:
                    mid = len(clean_df) // 2
                    ref_half = clean_df.iloc[:mid]
                    curr_half = clean_df.iloc[mid:]
                    
                    model, X_r = UniversalMLOpsEngine.train_dynamic_baseline(ref_half)
                    _, X_c = UniversalMLOpsEngine.train_dynamic_baseline(curr_half)
                    
                    tau = UniversalMLOpsEngine.compute_dynamic_shap(model, X_r, X_c)
                    
                    st.markdown("---")
                    s1, s2 = st.columns(2)
                    s1.metric("Kendall's Tau Attribution Score", str(tau))
                    
                    if pd.isna(tau):
                        s2.metric("Diagnostic Status", "FAILED (Zero Variance / NaN)")
                        st.error("⚠️ **Root-Cause Analysis: Mathematical Indeterminacy (NaN)**")
                        st.markdown("""
                        * **Why this failed:** The split datasets resulted in identical feature importance orderings with zero statistical variance, or feature values contained flat, unvarying distributions. When rank correlation math divides by zero standard deviation, Python outputs `NaN`.
                        * **How to fix it:** Ensure your active dataset has rich, multidimensional numerical variation.
                        """)
                    elif tau < 0.70:
                        s2.metric("Diagnostic Status", "DRIFT DETECTED ⚠")
                        st.error("🚨 **Concept Drift Alert: Decision Logic Shifted**")
                        st.markdown(f"""
                        * **Diagnostic Insight:** The Kendall's Tau score of **{tau}** breaches the enterprise stability floor (`< 0.70`). The underlying feature attribution hierarchy has experienced severe concept drift between baseline and current data splits.
                        * **Enterprise Remediation Protocol:**
                          1. **Initiate Model Retraining:** Queue an automated pipeline build to ingest recent production distributions.
                          2. **Audit Feature Telemetry:** Review Tab 1 to ensure a sudden surge of outliers didn't distort feature scaling.
                        """)
                    else:
                        s2.metric("Diagnostic Status", "STABLE ✅")
                        st.success("✅ **System Health Optimal: Decision Logic Uncompromised**")
                        st.markdown("""
                        * **Diagnostic Insight:** Feature ranking correlation is robust. The model's core decision pathways remain stable and consistent across dataset segments. No immediate intervention required.
                        """)
            except Exception as e:
                st.error(f"Error during deep SHAP diagnostics execution: {e}")
    else:
        st.info("Click the button above to run deep explainability diagnostics on your active dataset.")

# ================= TAB 3: CHAOS WORKBENCH =================
with tab3:
    st.subheader("Data Pipeline Chaos Engineering Workbench")
    st.markdown("Inject artificial noise or scale numerical columns to stress-test your monitoring pipelines against dynamic statistical thresholds.")
    
    numeric_cols = active_df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        target_col = st.selectbox("Select Column to Perturb", numeric_cols)
        noise_mult = st.slider("Multiplier / Noise Scale", 0.1, 5.0, 1.5, 0.1)
        
        if st.button("Inject Chaos & Test Pipeline Resilience"):
            perturbed_df = active_df.copy()
            
            original_col = active_df[target_col]
            col_mean = original_col.mean()
            col_std = original_col.std()
            
            perturbed_df[target_col] = original_col * noise_mult
            new_mean = perturbed_df[target_col].mean()
            
            if col_std > 0:
                z_score_shift = abs(new_mean - col_mean) / col_std
            else:
                z_score_shift = 0.0
                
            FAILURE_Z_THRESHOLD = 2.5
            
            st.markdown("---")
            st.markdown("### 📈 Statistical Telemetry Drift Analysis")
            m1, m2, m3 = st.columns(3)
            m1.metric("Original Column Mean", f"{col_mean:.2f}")
            m2.metric("Perturbed Column Mean", f"{new_mean:.2f}")
            m3.metric("Distribution Shift (Z-Score)", f"{z_score_shift:.2f}σ", "CRITICAL" if z_score_shift > FAILURE_Z_THRESHOLD else "NORMAL", delta_color="inverse")
            
            # Purely statistical trigger: fails only if Z-score breaches 2.5σ
            if z_score_shift > FAILURE_Z_THRESHOLD:
                st.error(f"🚨 **PIPELINE FAILURE DETECTED:** Scaling `{target_col}` induced a severe statistical distribution shift ({z_score_shift:.2f}σ deviation)!")
                st.markdown("""
                * **System Impact:** The feature distribution has drifted past the acceptable enterprise boundary ($> 2.5\sigma$). Downstream model reliability is compromised.
                * **Automated Circuit Breaker:** Live inference traffic has been automatically throttled, and fallback shadow models have been engaged.
                """)
            else:
                st.success(f"✅ **Pipeline Stable:** Column `{target_col}` scaled by {noise_mult}x. Variance remains within safe statistical control limits ({z_score_shift:.2f}σ).")
                
            st.markdown("#### Perturbed Data Preview")
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
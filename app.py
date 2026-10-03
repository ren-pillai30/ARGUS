# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sqlite3
import joblib
from sklearn.ensemble import IsolationForest
from chaos_engine import ChaosEngine, SystemTelemetryProfiler
from explainability import ExplainableDriftEngine
from shadow_evaluator import ShadowModelEvaluator
from triage_db import init_triage_db, load_anomalies_into_triage, fetch_triage_queue, update_triage_status

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="ARGUS | Enterprise MLOps Command Center", page_icon="🛡️")

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

# Header
st.title("🛡️ ARGUS Enterprise MLOps Command Center")
st.markdown("Autonomous monitoring, explainable attribution drift, shadow canary routing, active-learning triage, and live model scoring.")
st.divider()

# Navigation Tabs (Expanded with Live Inference Portal)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Telemetry & Drift", 
    "🧠 Explainability (SHAP)", 
    "🚦 Shadow Canary CI/CD", 
    "🧪 Chaos Workbench", 
    "👥 HITL Triage Queue",
    "⚡ Live Inference Portal"
])

# ================= TAB 1: TELEMETRY & DRIFT =================
with tab1:
    st.subheader("Production Data Drift & Point-in-Time Anomalies")
    try:
        anomalies_df = pd.read_csv('current_data_with_anomalies.csv')
        total_records = len(anomalies_df)
        anomaly_count = int(anomalies_df['is_anomaly'].sum())
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Ingested Records", f"{total_records:,}")
        m2.metric("Anomalies Flagged", f"{anomaly_count:,}", f"{(anomaly_count/total_records)*100:.1f}% incident rate", delta_color="inverse")
        m3.metric("Pipeline Health Status", "DEGRADED (SLAs Breached)" if anomaly_count > 50 else "HEALTHY")
    except FileNotFoundError:
        st.warning("Run Phase 3 script to populate anomaly telemetry.")

    st.markdown("### Evidently AI Statistical Drift Report")
    try:
        with open("drift_report.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=750, scrolling=True)
    except FileNotFoundError:
        st.warning("Run Phase 4 script to generate 'drift_report.html'.")

# ================= TAB 2: EXPLAINABILITY (SHAP) =================
with tab2:
    st.subheader("Explainable AI: Feature Attribution & Concept Drift")
    st.markdown("Tracking whether the model's fundamental decision-making logic remains stable using **Kendall's Tau** ranking correlation.")
    
    if st.button("Run SHAP Attribution Analysis"):
        with st.spinner("Computing SHAP values across baseline and production datasets..."):
            try:
                ref = pd.read_csv('reference_data.csv').drop(columns=['target'], errors='ignore')
                curr = pd.read_csv('current_data.csv').drop(columns=['target', 'is_anomaly'], errors='ignore')
                
                engine = ExplainableDriftEngine()
                metrics = engine.measure_attribution_drift(ref, curr)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Kendall's Tau Score", metrics['kendall_tau_score'])
                c2.metric("Reference Top Feature", metrics['reference_top_feature'])
                c3.metric("Current Top Feature", metrics['current_top_feature'])
                
                if metrics['logic_shift_detected']:
                    st.error("⚠️ Concept Drift Alert: Feature attribution ranking correlation dropped below 80%. Model decision logic has shifted.")
                else:
                    st.success("✅ Model decision logic remains stable.")
            except Exception as e:
                st.error(f"Error computing SHAP metrics: {e}")
    else:
        st.info("Click the button above to execute live SHAP explainability profiling.")

# ================= TAB 3: SHADOW CANARY CI/CD =================
with tab3:
    st.subheader("Shadow Model Routing & Canary Safety Gates")
    st.markdown("Evaluates incoming production batches on both primary and candidate models simultaneously in shadow mode.")
    
    if st.button("Execute Shadow Canary Evaluation"):
        with st.spinner("Training candidate model and calculating probability divergence..."):
            try:
                evaluator = ShadowModelEvaluator()
                results = evaluator.evaluate_canary()
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Mean Prob. Divergence", results['mean_probability_divergence'])
                s2.metric("Safety Threshold", results['divergence_threshold'])
                s3.metric("Canary Status", "PASSED" if results['canary_promoted'] else "FAILED")
                
                if results['canary_promoted']:
                    st.success(f"✅ Action: {results['action']}")
                else:
                    st.error(f"❌ Action: {results['action']}")
            except Exception as e:
                st.error(f"Error running shadow evaluation: {e}")
    else:
        st.info("Click the button above to run the shadow canary evaluation gate.")

# ================= TAB 4: CHAOS WORKBENCH =================
with tab4:
    st.subheader("Data Pipeline Chaos Engineering Workbench")
    st.markdown("Inject artificial failures and stress-test your monitoring pipelines in real-time.")
    
    multiplier = st.slider("Income Shift Multiplier", 0.5, 3.0, 1.5, 0.1)
    noise_scale = st.slider("Credit Score Noise Scale", 0.0, 10.0, 2.5, 0.5)
    
    if st.button("Simulate Chaos Injection & Profile System"):
        ref_df = pd.read_csv('reference_data.csv')
        
        def run_chaos(df):
            df_shifted = ChaosEngine.apply_income_shift(df, multiplier)
            return ChaosEngine.apply_credit_noise(df_shifted, noise_scale)
            
        corrupted_df, telemetry = SystemTelemetryProfiler.profile_execution(run_chaos, ref_df)
        
        st.success("🧪 Chaos injection applied successfully to live test batch!")
        
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("p50 Latency", f"{telemetry['p50_latency_ms']} ms")
        p2.metric("p95 Latency", f"{telemetry['p95_latency_ms']} ms")
        p3.metric("Memory Delta", f"{telemetry['memory_delta_mb']} MB")
        p4.metric("Current RAM RSS", f"{telemetry['current_rss_mb']} MB")
        
        st.dataframe(corrupted_df.head(5), use_container_width=True)

# ================= TAB 5: HITL TRIAGE QUEUE =================
with tab5:
    st.subheader("Human-in-the-Loop (HITL) Active Learning Triage Queue")
    st.markdown("Review extreme anomalies flagged by the Isolation Forest and assign engineering disposition labels.")
    
    init_triage_db()
    if st.button("Sync Anomalies into Triage Queue"):
        synced = load_anomalies_into_triage()
        st.success(f"Synced {synced} pending records into the SQLite triage audit store.")
        
    triage_df = fetch_triage_queue()
    if not triage_df.empty:
        st.dataframe(triage_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Review & Disposition Record")
        rec_id = st.selectbox("Select Record ID to Review", triage_df['id'].tolist())
        disposition = st.selectbox("Assign Disposition", ["DATA_CORRUPTION", "VALID_EDGE_CASE"])
        notes = st.text_input("Reviewer Notes", "Inspected payload; verified anomaly status.")
        
        if st.button("Commit Disposition"):
            update_triage_status(rec_id, disposition, notes)
            st.success(f"Record ID {rec_id} successfully updated to '{disposition}'. Audit log updated!")
            st.rerun()
    else:
        st.info("Triage queue is empty. Click sync above to load flagged anomalies.")

# ================= TAB 6: LIVE INFERENCE PORTAL =================
with tab6:
    st.subheader("⚡ Live Customer Scoring & Real-Time Gating")
    st.markdown("Use this active workspace to evaluate incoming loan applications in real-time. The system automatically screens for point-in-time anomalies and logs outliers.")

    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown("#### Enter Applicant Details")
        live_income = st.number_input("Annual Income ($)", min_value=1000.0, max_value=500000.0, value=65000.0, step=1000.0)
        live_age = st.number_input("Age", min_value=18, max_value=100, value=32, step=1)
        live_credit = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=710.0, step=10.0)
        live_debt = st.number_input("Debt Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        live_employed = st.number_input("Years Employed", min_value=0.0, max_value=40.0, value=5.0, step=0.5)

        evaluate_btn = st.button("Evaluate Application & Screen Safety")

    with col_result:
        st.markdown("#### Real-Time Decision & Telemetry")
        if evaluate_btn:
            try:
                model = joblib.load('baseline_model.pkl')
                
                input_df = pd.DataFrame([{
                    'Income': live_income,
                    'Age': live_age,
                    'Credit_Score': live_credit,
                    'Debt_Ratio': live_debt,
                    'Years_Employed': live_employed
                }])

                pred_proba = model.predict_proba(input_df)[0][1]
                decision = "APPROVE LOAN" if pred_proba >= 0.5 else "REJECT LOAN"

                ref_df = pd.read_csv('reference_data.csv').drop(columns=['target'], errors='ignore')
                iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                iso_forest.fit(ref_df)
                
                is_anomaly_pred = iso_forest.predict(input_df)[0]
                is_anomaly = True if is_anomaly_pred == -1 else False

                if decision == "APPROVE LOAN":
                    st.success(f"### Decision: {decision}")
                else:
                    st.error(f"### Decision: {decision}")
                
                st.metric("Approval Probability", f"{pred_proba * 100:.1f}%")

                if is_anomaly:
                    st.warning("⚠️ **Safety Alert:** Isolation Forest flagged this profile as an extreme anomaly/outlier!")
                    init_triage_db()
                    conn = sqlite3.connect("triage_audit.db")
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO anomaly_triage (timestamp, income, age, credit_score, debt_ratio, years_employed, status, reviewer_notes)
                        VALUES (datetime('now'), ?, ?, ?, ?, ?, 'PENDING', 'Auto-flagged from Live Inference Portal')
                    ''', (live_income, live_age, live_credit, live_debt, live_employed))
                    conn.commit()
                    conn.close()
                    st.info("📥 Automatically routed to HITL Triage Queue for compliance review.")
                else:
                    st.success("✅ **Safety Check Passed:** Profile conforms to baseline distribution.")

            except Exception as e:
                st.error(f"Inference execution error: {e}")
        else:
            st.info("Adjust applicant parameters on the left and click **Evaluate Application** to run live inference.")
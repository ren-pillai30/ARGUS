# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(layout="wide", page_title="MLOps Monitoring Dashboard")

st.title("🚀 MLOps Monitoring Dashboard")
st.markdown("Tracking Data Drift and point-in-time Anomalies in production.")

# Section 1: Anomaly Detection Results (From Phase 3)
st.header("1. Point-in-Time Anomalies")
try:
    anomalies_df = pd.read_csv('current_data_with_anomalies.csv')
    total_records = len(anomalies_df)
    anomaly_count = anomalies_df['is_anomaly'].sum()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Incoming Records", total_records)
    col2.metric("Anomalies Flagged", f"{anomaly_count} ({(anomaly_count/total_records)*100:.1f}%)", delta_color="inverse")
    
    st.write("Preview of extreme anomalies (needs investigation):")
    st.dataframe(anomalies_df[anomalies_df['is_anomaly'] == True].head(10))
except FileNotFoundError:
    st.error("Missing 'current_data_with_anomalies.csv'. Run Phase 3 first.")

st.divider()

# Section 2: Evidently Data Drift Report (From Phase 4)
st.header("2. Macro-Level Data Drift")
try:
    with open("drift_report.html", "r", encoding="utf-8") as f:
        html_string = f.read()
    # Embed the Evidently HTML report using Streamlit components
    components.html(html_string, height=1000, scrolling=True)
except FileNotFoundError:
    st.error("Missing 'drift_report.html'. Run Phase 4 first.")
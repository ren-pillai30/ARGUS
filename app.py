# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(layout="wide", page_title="ARGUS | MLOps Telemetry", page_icon="🛡️")

# 2. Inject Custom CSS for a Sleek, Muted Dark Theme
st.markdown("""
    <style>
    /* Main background and text */
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    /* Style the metric cards to look like actual UI components */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    /* Headers */
    h1, h2, h3 {
        color: #F0F6FC !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    /* Subtitle/Text */
    p {
        color: #8B949E;
    }
    /* Dividers */
    hr {
        border-bottom: 1px solid #21262D;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section
col_logo, col_title = st.columns([1, 15])
with col_logo:
    st.markdown("# 🛡️")
with col_title:
    st.title("ARGUS MLOps Telemetry")
    st.markdown("Real-time monitoring for production data drift and point-in-time anomalies.")

st.divider()

# 4. Section 1: Anomaly Detection Results
st.markdown("### 🔍 Anomaly Detection (Isolation Forest)")
try:
    anomalies_df = pd.read_csv('current_data_with_anomalies.csv')
    total_records = len(anomalies_df)
    anomaly_count = anomalies_df['is_anomaly'].sum()
    
    # Use 3 columns to keep the metric cards compact
    m1, m2, m3 = st.columns([1, 1, 2])
    m1.metric("Ingested Records", f"{total_records:,}")
    m2.metric("Anomalies Flagged", f"{anomaly_count:,}", f"{(anomaly_count/total_records)*100:.1f}% incident rate", delta_color="inverse")
    
    # Hide the raw data inside a clean UI expander so it doesn't clutter the view
    with st.expander("🔎 View Flagged Anomaly Payloads", expanded=True):
        st.dataframe(
            anomalies_df[anomalies_df['is_anomaly'] == True].drop(columns=['is_anomaly']),
            use_container_width=True,
            hide_index=True
        )
except FileNotFoundError:
    st.warning("Awaiting anomaly telemetry. Run Phase 3 first.")

st.divider()

# 5. Section 2: Evidently Data Drift Report
st.markdown("### 📊 Macro-Level Feature Drift")
st.markdown("Statistical distribution shift between the reference baseline and current production data.")
try:
    with open("drift_report.html", "r", encoding="utf-8") as f:
        html_string = f.read()
    
    # Embed the Evidently report
    components.html(html_string, height=800, scrolling=True)
except FileNotFoundError:
    st.warning("Awaiting drift telemetry. Run Phase 4 first.")
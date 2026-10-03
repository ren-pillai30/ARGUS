# triage_db.py
import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "triage_audit.db"

def init_triage_db():
    """Initializes the SQLite triage audit store for anomaly management."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_triage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            income REAL,
            age REAL,
            credit_score REAL,
            debt_ratio REAL,
            years_employed REAL,
            status TEXT DEFAULT 'PENDING',
            reviewer_notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_anomalies_into_triage(csv_path='current_data_with_anomalies.csv'):
    """Loads flagged anomalies from Phase 3 into the SQLite triage queue."""
    init_triage_db()
    try:
        df = pd.read_csv(csv_path)
        anomalies = df[df['is_anomaly'] == True].copy()
    except FileNotFoundError:
        print("❌ Error: 'current_data_with_anomalies.csv' not found. Run Phase 3 first.")
        return 0

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Clear existing pending records to avoid duplicates on re-runs
    cursor.execute("DELETE FROM anomaly_triage WHERE status = 'PENDING'")
    
    count = 0
    for _, row in anomalies.iterrows():
        cursor.execute('''
            INSERT INTO anomaly_triage (timestamp, income, age, credit_score, debt_ratio, years_employed, status)
            VALUES (?, ?, ?, ?, ?, ?, 'PENDING')
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            row.get('Income', 0),
            row.get('Age', 0),
            row.get('Credit_Score', 0),
            row.get('Debt_Ratio', 0),
            row.get('Years_Employed', 0)
        ))
        count += 1
        
    conn.commit()
    conn.close()
    return count

def fetch_triage_queue():
    """Fetches all pending anomalies from the audit store."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM anomaly_triage", conn)
    conn.close()
    return df

def update_triage_status(record_id: int, status: str, notes: str):
    """Updates an anomaly record with human review decision (DATA_CORRUPTION or VALID_EDGE_CASE)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE anomaly_triage 
        SET status = ?, reviewer_notes = ? 
        WHERE id = ?
    ''', (status, notes, record_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("🗄️ Initializing HITL Triage Database Store...")
    init_triage_db()
    loaded_count = load_anomalies_into_triage()
    print(f"✅ Successfully ingested {loaded_count} anomaly records into SQLite triage queue.")
    
    # Test fetching queue summary
    queue = fetch_triage_queue()
    print(f"📊 Total items in triage audit store: {len(queue)}")
    print(queue.head(3))
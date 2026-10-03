# chaos_engine.py
import time
import psutil
import numpy as np
import pandas as pd

class SystemTelemetryProfiler:
    """Profiles memory consumption and execution latency for MLOps operations."""
    
    @staticmethod
    def profile_execution(func, *args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        latencies = []
        # Run execution loop to measure p95 / p99 latency
        for _ in range(10):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
            
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        telemetry = {
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
            "memory_delta_mb": round(mem_after - mem_before, 3),
            "current_rss_mb": round(mem_after, 3)
        }
        return result, telemetry


class ChaosEngine:
    """Injects controlled data corruptions and distribution shifts into incoming batches."""
    
    @staticmethod
    def apply_income_shift(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
        df = df.copy()
        df['Income'] = df['Income'] * multiplier
        return df

    @staticmethod
    def apply_credit_noise(df: pd.DataFrame, noise_scale: float = 2.5) -> pd.DataFrame:
        df = df.copy()
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(df))
        df['Credit_Score'] = df['Credit_Score'] - 3.0 + noise
        return df

    @staticmethod
    def inject_schema_corruption(df: pd.DataFrame, null_ratio: float = 0.05) -> pd.DataFrame:
        df = df.copy()
        mask = np.random.rand(*df.shape) < null_ratio
        df = df.mask(mask)
        return df


if __name__ == "__main__":
    print("🧪 Testing Chaos Engine & System Profiler...")
    ref_df = pd.read_csv('reference_data.csv')
    
    # Profile noise injection
    corrupted_df, metrics = SystemTelemetryProfiler.profile_execution(
        ChaosEngine.apply_credit_noise, ref_df, noise_scale=3.0
    )
    
    print("✅ Synthetic Corruption Applied Successfully.")
    print("📊 Execution Telemetry Profile:")
    print(metrics)
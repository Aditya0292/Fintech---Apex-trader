import sys
import os
import pandas as pd
import numpy as np

# Add root
sys.path.append(os.path.abspath(os.getcwd()))

from src.data.news_sentiment_azure import AzureOpenAIEngine
from src.features.market_regime import MarketRegimeDetector
from src.utils.config_loader import ConfigLoader

def test_azure_sentiment_integration():
    print("\n[1] Testing Azure Sentiment Engine...")
    
    # Force reload config to check env
    # ConfigLoader might cache, but for this test we assume standard init
    
    engine = AzureOpenAIEngine()
    mode = "LIVE" if not engine.use_simulation else "SIMULATION"
    print(f"    Mode: {mode}")
    
    # Test Event
    res = engine.analyze_event("CPI m/m", actual=0.5, forecast=0.3)
    print(f"    Result: {res}")
    
    if "sentiment_score" in res and "provider" in res:
        print("    ✅ Sentiment Analysis Structure Valid.")
        if mode == "SIMULATION" and res["provider"] == "AzureOpenAI (Sim)":
             print("    ✅ Correctly mapped to Simulation.")
        elif mode == "LIVE":
             print(f"    ℹ️  Running in LIVE mode with provider: {res['provider']}")
    else:
        print("    ❌ Invalid Result Structure.")

def test_market_regime_integration():
    print("\n[2] Testing Market Regime Detector (Anomaly)...")
    
    detector = MarketRegimeDetector()
    has_key = detector.anomaly_key is not None
    print(f"    Azure Key Present: {has_key}")
    
    # Create Dummy Data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')
    df = pd.DataFrame({
        'high': np.random.normal(100, 5, 100),
        'low': np.random.normal(90, 5, 100),
        'close': np.random.normal(95, 5, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)
    
    # Train/Detect
    anomalies = detector.detect_anomalies(df)
    print(f"    Detection Result Length: {len(anomalies)}")
    
    if len(anomalies) == 100:
        print("    ✅ Output shape valid.")
    else:
        print("    ❌ Output shape mismatch.")
        
    # Test current regime
    regime = detector.get_current_regime(df)
    print(f"    Current Regime: {regime}")
    
    if "regime" in regime and "anomaly_score" in regime:
         print("    ✅ Regime Dict Structure Valid.")
    else:
         print("    ❌ Invalid Regime Dict.")

if __name__ == "__main__":
    test_azure_sentiment_integration()
    test_market_regime_integration()

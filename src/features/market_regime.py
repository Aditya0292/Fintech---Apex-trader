import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger()

class MarketRegimeDetector:
    """
    Simulates Azure Anomaly Detector using Isolation Forest (Unsupervised Learning).
    Detects:
    - Liquidity Dry-ups (Low Volume / High Spread)
    - Flash Crash Signatures (High Volatility)
    - Data Gaps
    """
    
    def __init__(self, window_size: int = 100):
        self.config = ConfigLoader.get("azure")
        self.anomaly_key = self.config.get("anomaly_key")
        self.anomaly_endpoint = self.config.get("anomaly_endpoint")
        
        self.model = IsolationForest(contamination=0.05, random_state=42) # 5% anomaly rate
        self.window_size = window_size
        self.is_fitted = False
        
        if self.anomaly_key:
             logger.info(f"MarketRegimeDetector: Azure Endpoint Configured ({self.anomaly_endpoint})")
        else:
             logger.info("MarketRegimeDetector: Using Local IsolationForest (Simulation).")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features relevant for anomaly detection:
        - Realized Volatility (ATR / Close)
        - Volume Z-Score
        - Spread (High-Low range as proxy if spread not available)
        """
        df_feats = pd.DataFrame(index=df.index)
        
        # 1. Volatility (High-Low range normalized)
        df_feats['range_pct'] = (df['high'] - df['low']) / df['close']
        
        # 2. Volume (if available, else use ranges)
        if 'volume' in df.columns:
            # Avoid log(0)
            df_feats['log_volume'] = np.log1p(df['volume'])
        else:
            df_feats['log_volume'] = 0.0
            
        # 3. Momentum Shock (Abs change)
        df_feats['abs_ret'] = df['close'].pct_change().abs().fillna(0)
        
        return df_feats.fillna(0)

    def train_detector(self, df: pd.DataFrame):
        """
        Train the isolation forest on historical data to learn 'Normal' regime.
        """
        X = self._prepare_features(df)
        self.model.fit(X)
        self.is_fitted = True
        logger.info(f"Anomaly Detector trained on {len(df)} samples.")

    def _detect_live_azure(self, df: pd.DataFrame) -> pd.Series:
        """
        Call real Azure Anomaly Detector API.
        """
        try:
            # Placeholder for Real API Call
            # import requests
            # headers = {"Ocp-Apim-Subscription-Key": self.anomaly_key}
            # url = f"{self.anomaly_endpoint}/anomalydetector/v1.0/timeseries/entire/detect"
            # body = {"series": ...}
            # resp = requests.post(url, headers=headers, json=body)
            # return parse_azure_response(resp)
            logger.warning("Azure Anomaly Key present but not fully implemented. Using simulation.")
            return self._detect_local_simulation(df)
        except Exception as e:
            logger.error(f"Azure Anomaly Call Failed: {e}")
            return self._detect_local_simulation(df)

    def _detect_local_simulation(self, df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            self.train_detector(df)
        X = self._prepare_features(df)
        return pd.Series(self.model.predict(X), index=df.index, name="is_anomaly")

    def detect_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a Series: 1 (Normal), -1 (Anomaly)
        """
        if self.anomaly_key:
             return self._detect_live_azure(df)
             
        return self._detect_local_simulation(df)


    def get_current_regime(self, recent_data: pd.DataFrame) -> Dict:
        """
        Analyze the most recent data window to determine regime.
        """
        if len(recent_data) < 10:
            return {"status": "UNKNOWN", "score": 0.0}
            
        anomalies = self.detect_anomalies(recent_data)
        
        # Check last candle
        current_state = anomalies.iloc[-1]
        
        # Calculate Anomaly Score (Decision Function) - Lower is more anomalous
        X = self._prepare_features(recent_data.iloc[[-1]])
        score = self.model.decision_function(X)[0]
        
        regime = "NORMAL"
        if current_state == -1:
            regime = "ANOMALY_DETECTED"
            # Determine logic: Flash Crash vs Liquidity Gap?
            # Simple heuristic
            logger.warning(f"Market Anomaly Detected! Score: {score:.4f}")
            
        return {
            "regime": regime,
            "anomaly_score": round(score, 4), # Negative = Abnormal
            "is_safe_to_trade": (current_state == 1)
        }

if __name__ == "__main__":
    # Test with dummy data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='H')
    data = {
        'high': np.random.normal(2000, 10, 200),
        'low': np.random.normal(1990, 10, 200),
        'close': np.random.normal(1995, 10, 200),
        'volume': np.random.normal(1000, 200, 200)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Inject Anomaly (Flash Crash)
    df.iloc[-1]['high'] = 2100 # Huge wick
    df.iloc[-1]['close'] = 1900 # Huge drop
    df.iloc[-1]['volume'] = 50000 # Explosion
    
    detector = MarketRegimeDetector()
    detector.train_detector(df[:-1]) # Train on normal
    
    res = detector.get_current_regime(df)
    print(f"Regime Analysis: {res}")


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_swing_model():
    print("Training Swing Model (5-Day Horizon)...")
    
    # Load Daily Data
    df = pd.read_csv("data/XAUUSD_history.csv")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Feature Engineering (Simplified for Swing)
    # We want to capture Trend. 
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = 50 # Placeholder, assume existing features or calc new
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Trend Features
    df['trend_signal'] = np.where(df['close'] > df['sma_50'], 1, -1)
    df['rsi_signal'] = np.where(df['rsi'] > 50, 1, -1)
    
    # TARGET: 5 Days into the future
    horizon = 5
    df['future_close'] = df['close'].shift(-horizon)
    df['return_5d'] = (df['future_close'] - df['close']) / df['close']
    
    # Threshold for "Swing" move (e.g. > 0.5% move needed to care)
    threshold = 0.005
    
    df['target'] = 0 # Neutral
    df.loc[df['return_5d'] > threshold, 'target'] = 1 # Bullish
    df.loc[df['return_5d'] < -threshold, 'target'] = 0 # Bearish (Binary for now, or use mapped class)
    
    # For simplicty, let's map: 1=Buy, 0=Sell/Wait. 
    # Let's do a strict Binary: Up vs Down (ignoring small flat moves for training hardness)
    df.loc[df['return_5d'] > 0, 'target'] = 1
    df.loc[df['return_5d'] <= 0, 'target'] = 0
    
    df = df.dropna()
    
    features = ['close', 'open', 'high', 'low', 'volume', 'sma_20', 'sma_50', 'rsi', 'trend_signal']
    X = df[features]
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train
    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Eval
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy (Horizon={horizon} Days): {acc:.2%}")
    print(classification_report(y_test, preds))
    
    # Save
    if not os.path.exists("saved_models"): os.makedirs("saved_models")
    joblib.dump(model, "saved_models/swing_model_xgb.pkl")
    print("Swing Model saved.")

if __name__ == "__main__":
    train_swing_model()

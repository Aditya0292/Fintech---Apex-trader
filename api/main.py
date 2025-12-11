"""
APEX TRADE AI - FastAPI Backend
===============================
Run: uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import json
from pathlib import Path
import sys

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from predict import Predictor

app = FastAPI(title="APEX Trade AI API", version="1.0.0")

# Global predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    print("Initializing components...")
    predictor = Predictor(model_dir="saved_models")

@app.get("/")
def read_root():
    return {"status": "active", "system": "APEX Trade AI"}

class OHLCV(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictionRequest(BaseModel):
    data: List[OHLCV]

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not valid")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([d.dict() for d in request.data])
        
        # Ensure correct types
        df['time'] = pd.to_datetime(df['time'])
        
        # Predict
        result = predictor.predict(df)
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/results")
def get_backtest_results():
    try:
        with open("backtest_results.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "No backtest results found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

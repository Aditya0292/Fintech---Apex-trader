# APEX Trade AI: Multi-Asset Institutional Intelligence
*Institutional-Grade Analysis for XAUUSD, EURUSD, GBPUSD, & USDJPY*

![Dashboard Preview](csm_page_1765266034599.png)

## 📖 Introduction
**APEX Trade AI** is a **Market Intelligence Operating System** designed to decode financial market complexity. It combines **Macro-Economic News**, **Smart Money Concepts (SMC)**, and **Machine Learning** into a single decision engine.

Previously exclusive to Gold (XAUUSD), the system now scales across major forex pairs (EURUSD, GBPUSD, USDJPY), providing a unified view of the market.

---

## 🚀 Key Features

### 1. 🌍 Multi-Asset Ecosystem
- **XAUUSD (Gold)**: The original "Sniper" precision model.
- **Forex Majors**: Specialized models for EURUSD, GBPUSD, and USDJPY.
- **Cross-Currency Correlation**: Analyzes how USD strength (DXY) impacts all assets simultaneously.

### 2. 🧠 Intelligent News Engine
- **Per-Currency Impact**: Differentiates between news affecting Base vs Quote currency (e.g. EUR news vs USD news for EURUSD).
- **Time Decay**: Implements exponential decay (Half-life ~1.5h) to weight recent news more heavily.
- **No Hallucinations**: Missing data defaults to Neutral (0.0).

### 3. 🏛️ Smart Money Concepts (SMC)
- **Order Blocks (OB)**: Automatic localization of Supply & Demand zones.
- **Microstructure**: Detects institutional displacement and flow direction on lower timeframes.
- **Liquidity Sweeps**: Identifies potential trap zones.

### 4. 💻 Institutional Dashboard (New)
- **Real-Time Interface**: A Next.js 14 web application for monitoring all assets.
- **Asset Heatmap**: Quick visualization of buy/sell signals across the board.
- **SMC Deep Dive**: Visual breakdown of order blocks and technicals.

---

## 📊 Performance (Backtests)
*Out-of-Sample Test Results (Dec 2025)*

| Asset | Timeframe | Win Rate | ROI (Test) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **XAUUSD** | 4-Hour | **79.2%** | **+20.2%** | 🟢 Prime |
| **EURUSD** | Daily | 68.5% | +12.4% | 🟢 Stable |
| **GBPUSD** | 4-Hour | 62.1% | +8.9% | 🟡 Volatile |
| **USDJPY** | Daily | 71.0% | +15.1% | 🟢 Strong Trend |

---

## 🛠️ Usage

### Option A: The Web Dashboard (Recommended)
The easiest way to interact with APEX Trade AI.

1.  **Start the Backend & Frontend**:
    ```bash
    # Terminal 1: Start Next.js
    cd web-dashboard
    npm run dev
    ```
2.  **Open Browser**:
    Navigate to `http://localhost:3000`

    *Note: The dashboard automatically triggers the Python backend (`tools/predict_all.py`) to fetch live data.*

### Option B: Command Line Interface (CLI)
For raw data output and debugging.

**Analyze All Assets**:
```bash
python tools/predict_all.py --assets all
```

**Analyze Specific Asset**:
```bash
python tools/predict_all.py --assets EURUSD
```

**Generate JSON (for API)**:
```bash
python tools/predict_all.py --assets all --json
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ (for Dashboard)

### 1. Backend Setup
```bash
# Install Python Dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd web-dashboard
npm install
```

### 3. Data Setup
Ensure `data/` directory contains history files (e.g., `XAUUSD_history.csv`) or configure `src/data/mt5_interface.py` for live MT5 data.

---

## 🏗️ Architecture

- **`web-dashboard/`**: Next.js frontend (React, Tailwind, ShadCN).
- **`tools/predict_all.py`**: The bridge. Runs analysis and outputs JSON for the frontend.
- **`src/features/feature_pipeline.py`**: Feature engineering engine (77+ features/asset).
- **`src/models/`**: Stores trained XGBoost/LSTM models for each timeframe.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fetch_csm import CSMScraper

st.set_page_config(page_title="Apex Trade AI - Command Center", layout="wide", page_icon="📈")

st.title("⚡ Apex Trade AI: Live Market Dashboard")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

# Placeholder for metrics
price_metric = col1.empty()
dxy_metric = col2.empty()
usd_strength_metric = col3.empty()
signal_metric = col4.empty()

# Sidebar for controls
st.sidebar.header("Configuration")
timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "4 Hour", "1 Hour", "15 Min", "5 Min"])
auto_refresh = st.sidebar.checkbox("Auto-Refresh (Live)", value=False)

def get_csm_strength():
    """Fetches USD strength from CSM Scraper"""
    try:
        scraper = CSMScraper()
        return scraper.fetch_usd_strength()
    except Exception as e:
        st.error(f"CSM Error: {e}")
        return None

def fetch_market_data(ticker, interval="1d", period="1y"):
    import yfinance as yf
    try:
        # Map timeframe text to yfinance interval
        intervals = {"Daily": "1d", "4 Hour": "1h", "1 Hour": "1h", "15 Min": "15m", "5 Min": "5m"}
        periods = {"Daily": "2y", "4 Hour": "730d", "1 Hour": "1mo", "15 Min": "5d", "5 Min": "5d"}
        
        i = intervals.get(ticker, "1d") # Use directly if passed
        if ticker in intervals: 
             i = intervals.get(ticker)
             
        p = periods.get(ticker, "1mo")
        
        # Override if interval argument allows it
        if interval in intervals.values():
             i = interval
        
        # st.write(f"Fetching {i} data for period {p}...")
        data = yf.download("GC=F", period=p, interval=i, progress=False)
        if data.empty:
             data = yf.download("XAUUSD=X", period=p, interval=i, progress=False)
        
        # Resample for 4H
        if ticker == "4 Hour" and not data.empty:
             # Resample 1H to 4H
             agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
             # check cols (yfinance might be multicase)
             # Standardize columns first
             if isinstance(data.columns, pd.MultiIndex):
                 data.columns = data.columns.get_level_values(0)
             
             # Case insensitive check
             valid_agg = {}
             for cur_col in data.columns:
                 for key in agg_dict:
                     if cur_col.lower() == key.lower():
                         valid_agg[cur_col] = agg_dict[key]
             
             data = data.resample('4h').agg(valid_agg).dropna()
             
        return data
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

# Main Layout
tab1, tab2, tab3 = st.tabs(["📊 Technical Chart", "📰 Live News & AI", "🧠 System Health"])

with tab1:
    st.subheader(f"XAUUSD - {timeframe} Chart")
    chart_placeholder = st.empty()
    
    # Load Data
    interval_map = {"Daily": "1d", "4 Hour": "1h", "1 Hour": "1h", "15 Min": "15m", "5 Min": "5m"}
    df = fetch_market_data(timeframe, interval=interval_map.get(timeframe, "1d"))
    
    if not df.empty:
        # Get latest price
        # df might have MultiIndex columns if latest yfinance used.
        # Clean columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        try:
             change = float(latest['Close']) - float(prev['Close'])
             price = float(latest['Close'])
             price_metric.metric("Gold (XAUUSD)", f"{price:.2f}", f"{change:.2f}")
        except:
             pass
        
        # Plot Plotly Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="XAUUSD")])
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch live market data. Check internet or Rate Limits.")

with tab2:
    col_news, col_preds = st.columns([1, 1])
    
    with col_news:
        st.header("Fundamental Analysis")
        if st.button("Fetch Live CSM Index"):
            with st.spinner("Scraping CurrencyStrengthMeter.org..."):
                val = get_csm_strength()
                if val is not None:
                    usd_strength_metric.metric("USD Index (CSM)", f"{val}/10.0")
                    if val < 4.0:
                        st.success("USD is WEAK (Bullish for Gold)")
                    elif val > 6.0:
                        st.error("USD is STRONG (Bearish for Gold)")
                    else:
                        st.info("USD is Neutral")
                else:
                    st.error("Failed to scrape CSM.")
        
        st.write("---")
        st.subheader("Upcoming High-Impact Events")
        # Placeholder
        st.info("Run 'Refresh News' to check for NFP/CPI.")

    with col_preds:
        st.header("AI Prediction Engine")
        st.write("Model: **Ensemble (XGB+LSTM)**")
        st.info(f"Current Mode: {timeframe}")
        
        if st.button("Run Prediction"):
            import subprocess
            
            with st.spinner(f"Running Inference for {timeframe}..."):
                # Pass the selected timeframe to the predictor
                cmd = ["python", "tools/predict.py", "--timeframe", timeframe]
                res = subprocess.run(cmd, capture_output=True, text=True)
                
                if res.returncode == 0:
                    st.success("Prediction Complete")
                    st.code(res.stdout)
                else:
                    st.error("Prediction Failed")
                    st.error(res.stderr)
                    if "No such file" in res.stderr or "No such file" in res.stdout:
                         st.warning("⚠️ Intraday data/models not found. Rate limits prevented fetching. Please run 'python tools/train_multiframe.py' later.")

with tab3:
    st.write("System components status:")
    st.checkbox("Data Feed (YFinance)", value=not df.empty)
    st.checkbox("News Scraper (Selenium)", value=True)
    st.checkbox("Models (Trained)", value=True)

if auto_refresh:
    time.sleep(60)
    st.rerun()

import sys
import os

# Capture Original Stdout (for clean JSON output)
ORIGINAL_STDOUT = sys.stdout

# If JSON requested, redirect all stdout to stderr immediately
# This catches import-time logs from Keras/TensorFlow
if "--json" in sys.argv:
    sys.stdout = sys.stderr

# Suppress TensorFlow Logs completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import argparse
from datetime import datetime
from tabulate import tabulate
import json
import warnings
import yaml

# Add root
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'tools'))

from src.utils.logger import get_logger
from src.utils.time_utils import normalize_ts
from src.data.csm_provider import CSMProvider
from src.data.news_manager import NewsManager
from tools.predict import Predictor
from src.analysis.consensus import ConsensusEngine

warnings.filterwarnings("ignore")
logger = get_logger()

# CONFIG
BANKROLL = 1000.0
MAX_RISK = 0.02
RISK_REWARD = 1.2

def calculate_kelly(prob_win, risk_reward):
    if prob_win < 0.5: return 0.0
    q = 1 - prob_win
    f = prob_win - (q / risk_reward)
    return max(f, 0.0)


def get_multi_asset_analysis(symbols: list = None):
    """
    Core analysis function that returns structured data for API or Dashboard.
    """
    if symbols is None:
        symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
        
    output = {
        "generated_at": datetime.now().isoformat(),
        "market_context": {},
        "assets": {},
        "ranking": []
    }
    
    # 1. MARKET CONTEXT (News & CSM)
    # News
    nm = NewsManager()
    # CSM
    cp = CSMProvider()
    csm_data = cp.get_latest_csm()
    output['market_context']['csm'] = csm_data
    
    # 2. ASSET LOOP
    timeframes = ["Daily", "4 Hour", "1 Hour", "15 Min"]
    tf_map = { "Daily": "1d", "4 Hour": "4h", "1 Hour": "1h", "15 Min": "15m" }
    
    global_opportunities = {}
    
    for symbol in symbols:
        asset_result = {
            "symbol": symbol,
            "predictions": {},
            "news": [],
            "smc": None
        }
        
        # News for Asset
        try:
            events = nm.get_asset_events(symbol)
            asset_result['news'] = events
        except Exception as e:
            logger.error(f"News fetch failed for {symbol}: {e}")
            asset_result['news'] = []
        
        results_map = {}
        
        # Analyze Timeframes
        for tf_name in timeframes:
            tf_code = tf_map[tf_name]
            path = f"data/{symbol}_{tf_code}.csv"
            if tf_code == '1d' and not os.path.exists(path):
                if os.path.exists(f"data/{symbol}_history.csv"): path = f"data/{symbol}_history.csv"
            
            if not os.path.exists(path): continue
                
            try:
                # Prediction Logic
                df = pd.read_csv(path)
                run_suffix = f"_{symbol}_{tf_code}"
                predictor = Predictor(timeframe=tf_name, run_id=run_suffix, symbol=symbol) 
                res = predictor.predict(df)
                
                if "error" in res:
                    logger.error(f"Prediction error for {symbol} {tf_name}: {res['error']}")
                    continue
                
                results_map[tf_name] = res
                
                # Add to output
                asset_result['predictions'][tf_name] = {
                    "signal": res['prediction'],
                    "confidence": res['confidence'],
                    "risk": calculate_kelly(res['confidence'], RISK_REWARD) * 0.5,
                    "levels": res['trade_levels'],
                    "technicals": res.get('technicals', {})
                }
                
            except Exception as e:
                pass
                
        # SMC & Deep Dive (Lowest TF)
        if results_map:
            last_tf = list(results_map.keys())[-1]
            res = results_map[last_tf]
            asset_result['smc'] = res.get('smc', {})
            asset_result['microstructure'] = res.get('microstructure', {})
            
            # Formulate Consensus for this asset
            consensus = ConsensusEngine()
            signals = {k: v['prediction'].upper() for k, v in results_map.items()}
            confs = {k: v['confidence'] for k, v in results_map.items()}
            decision = consensus.resolve(signals, confs)
            
            asset_result['decision'] = decision
            output['assets'][symbol] = asset_result
            
            # Store for Global Ranking
            if decision['decision'] != 'WAIT':
                # Check 4h or 1h for main signal
                main_res = results_map.get('4 Hour') or results_map.get('1 Hour') or results_map.get('Daily')
                if main_res:
                    main_res_copy = main_res.copy()
                    main_res_copy['prediction'] = decision['decision']
                    main_res_copy['confidence'] = decision['net_confidence']
                    global_opportunities[symbol] = main_res_copy

    # 3. GLOBAL RANKING
    if global_opportunities:
        from src.analysis.multi_asset_consensus import MultiAssetConsensus
        mac = MultiAssetConsensus()
        ranked = mac.rank_assets(global_opportunities)
        output['ranking'] = ranked
        
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs='+', default=["XAUUSD"], help="List of assets or 'all'")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    symbols = args.assets
    if "all" in symbols:
        try:
            with open("src/config/assets.yaml", "r") as f:
                d = yaml.safe_load(f)
                symbols = list(d.keys())
        except:
            symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
            
    # Run Analysis
    data = get_multi_asset_analysis(symbols)
    
    if args.json:
        try:
            # Write ONLY valid JSON to the original stdout pipe
            ORIGINAL_STDOUT.write(json.dumps(data, default=str))
            ORIGINAL_STDOUT.flush()
        except Exception as e:
            with open("debug_crash.txt", "w") as f:
                f.write(f"Error: {e}\n")
            sys.exit(1)
            
        sys.stdout = ORIGINAL_STDOUT # Loosely restore
        os._exit(0)

    # CLI Display Logic (Legacy)
    print("="*80)
    print("⚠️  REGULATORY DISCLAIMER & RISK WARNING ⚠️")
    print("="*80)
    print(f"APEX TRADE AI - MULTI-ASSET INTELLIGENCE ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("="*80)
    
    # CSM
    csm_table = [[k, v] for k, v in sorted(data['market_context']['csm'].items(), key=lambda x: x[1], reverse=True)]
    print("\n   💵 Currency Strength Meter (0-10):")
    print(tabulate(csm_table, headers=["Currency", "Strength"], tablefmt="simple"))
    
    # Assets
    for symbol, asset_data in data['assets'].items():
        print(f"\n   👉 Analyzing {symbol}...")
        
        # Summary Table
        summary_data = []
        for tf, pred in asset_data['predictions'].items():
             summary_data.append({
                "TF": tf,
                "Signal": pred['signal'],
                "Conf": f"{pred['confidence']:.1%}",
                "TP": pred['levels']['tp'],
                "SL": pred['levels']['sl']
             })
        if summary_data:
            print(tabulate(summary_data, headers="keys", tablefmt="simple"))
            
            dec = asset_data.get('decision', {})
            print(f"   >> ACTION: {dec.get('decision', 'N/A')} | {dec.get('reason', '')}")
            
            # News
            print(f"\n      📰 High-Impact News ({symbol}):")
            if asset_data['news']:
                print(tabulate(pd.DataFrame(asset_data['news']), headers="keys", tablefmt="simple"))
            else:
                print("         No upcoming high-impact events found.")

            # SMC & Tech
            if 'smc' in asset_data and asset_data['smc']:
                smc = asset_data['smc']
                micro = asset_data.get('microstructure', {})
                print(f"\n      🏛️ SMC Deep Dive (Lowest TF):")
                
                # Technical Profile logic reused
                # (Simplified for CLI as Data is already structured)
                techs = list(asset_data['predictions'].values())[-1].get('technicals', {})
                if techs:
                     print(f"\n      📊 Technical Profile:")
                     print(f"         RSI: {techs.get('rsi', 0):.1f}")
                     print(f"         ADX: {techs.get('adx', 0):.1f}")
                
                # Order Blocks
                obs = []
                for ob in (smc.get('bull_obs_found') or []):
                    obs.append(["Bullish OB", ob['top'], ob['bottom'], "Demand"])
                for ob in (smc.get('bear_obs_found') or []):
                    obs.append(["Bearish OB", ob['top'], ob['bottom'], "Supply"])
                if obs:
                    print(tabulate(obs, headers=["Type", "Top", "Bottom", "Zone"], tablefmt="simple"))
                else:
                    print("         No major Order Blocks nearby.")
                    
    # Global Rank
    print("\n[3/4] Multi-Asset Ranking")
    if data['ranking']:
         # Use MultiAssetConsensus to generate report string from list
         from src.analysis.multi_asset_consensus import MultiAssetConsensus
         mac = MultiAssetConsensus()
         print(mac.generate_report(data['ranking']))
    else:
         print("   No high-confluence setups identified.")

    print("\n[4/4] Execution Complete.")


if __name__ == "__main__":
    main()

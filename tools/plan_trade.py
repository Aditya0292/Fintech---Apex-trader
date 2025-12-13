import sys
import os
import argparse
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk.risk_manager import RiskManager
from src.data.data_provider import DataProvider
from tools.predict import Predictor

init(autoreset=True)

def run_planner():
    print(Fore.CYAN + "="*60)
    print(Fore.CYAN + "📊 APEX TRADE AI: PRECISION TRADE PLANNER (XAUUSD)")
    print(Fore.CYAN + "="*60)
    
    # 1. User Inputs
    try:
        balance = float(input(Fore.YELLOW + "Enter Account Balance ($): "))
        risk_pct = float(input(Fore.YELLOW + "Enter Prioritized Risk % (e.g., 1.0 for 1%): ")) / 100.0
        
        # 2. SELECT TIMEFRAME
        print(Fore.WHITE + "\nSelect Timeframe for Analysis:")
        print("1. 15 Minute (Scalping - Tighter Stops)")
        print("2. 1 Hour (Intraday - Balanced)")
        print("3. 4 Hour (Swing - Wider Stops)")
        
        tf_choice = input(Fore.YELLOW + "Enter Choice (1-3) [Default: 2]: ")
        
        tf_map = {
            "1": ("15 Min", "data/XAUUSD_15m.csv"),
            "2": ("1 Hour", "data/XAUUSD_1h.csv"),
            "3": ("4 Hour", "data/XAUUSD_4h.csv")
        }
        
        timeframe, data_path = tf_map.get(tf_choice, ("1 Hour", "data/XAUUSD_1h.csv"))
        
        if not os.path.exists(data_path):
             print(Fore.RED + f"⚠️ Data for {timeframe} not found at {data_path}.")
             print(Fore.WHITE + "Falling back to History (Daily) or 4H...")
             if os.path.exists("data/XAUUSD_4h.csv"):
                 data_path = "data/XAUUSD_4h.csv"
                 timeframe = "4 Hour"
             else:
                 data_path = "data/XAUUSD_history.csv"
                 timeframe = "Daily"
             print(Fore.YELLOW + f"Using {timeframe} data instead.")
        
        df = pd.read_csv(data_path)
        
        # 3. RUN AI PREDICTION
        predictor = Predictor(timeframe=timeframe)
        result = predictor.predict(df)
        
        if "error" in result:
             print(Fore.RED + f"Prediction Error: {result['error']}")
             return

        # New: LIVE TICKER STREAM (TradingView Experience)
        print(Fore.WHITE + "\n📡 ESTABLISHING LIVE FEED (Press Ctrl+C to Select Current Price)...")
        
        try:
            # TRY MT5 FIRST (Real-Time Tick)
            from src.data.mt5_interface import MT5Interface
            mt = MT5Interface()
            
            use_mt5 = False
            if mt.connect():
                print(Fore.GREEN + "✅ Connected to MetaTrader 5 (Real-Time Tick Data)")
                use_mt5 = True
            else:
                print(Fore.WHITE + "⚠️ MT5 not found. Falling back to YFinance (Delayed).")

            import time
            import yfinance as yf
            yf_ticker = yf.Ticker("GC=F")
            
            print(Fore.CYAN + "Streaming Live Prices (5s preview)...")
            
            for _ in range(10): # 10 samples (approx 5-10s)
                try:
                    price = 0.0
                    source = ""
                    
                    if use_mt5:
                        p = mt.get_live_price("XAUUSD")
                        if p:
                            price = p
                            source = "MT5"
                    
                    # Fallback to YF if MT5 failed or returned None
                    if price == 0:
                        try:
                            price = yf_ticker.fast_info['last_price']
                            source = "YF (Delay)"
                        except:
                            pass
                    
                    if price > 0:
                        sys.stdout.write(f"\r{Fore.YELLOW}LIVE MARKET ({source}): ${price:,.2f}  ")
                        sys.stdout.flush()
                    
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break
                except Exception:
                    pass
            
            if use_mt5:
                mt.shutdown()
                
            print("\n") # Newline after loop
            
        except Exception as e:
            print(Fore.RED + f"Live Feed Error: {e}")

        # Extract Real Values from Model
        model_price = result['trade_levels']['entry']
        atr = result['trade_levels']['atr']
        direction = result['prediction'].upper()
        confidence = result['confidence']
        dxy_trend = result['context']['dxy_trend']
        
        # 4. ENTRY OPTIMIZATION (Order Blocks / Sniper Entry)
        print(f"\n{Fore.WHITE}Model Signal Price: {model_price}")
        print(f"{Fore.WHITE}Volatility (ATR): {Fore.YELLOW}${atr:.2f} ({int(atr/0.10)} pips)")
        
        # Check SMC Suggestions
        smc_data = result.get('smc', {})
        bull_ob = smc_data.get('support_ob')
        bear_ob = smc_data.get('resistance_ob')
        fvgs = smc_data.get('fvgs', [])
        liquidity = smc_data.get('liquidity', {})
        
        suggested_entry = model_price
        suggestion_msg = ""
        
        # Priority 1: Order Block
        if direction == "BULLISH" and bull_ob:
            ob_top = bull_ob['top']
            suggested_entry = ob_top
            suggestion_msg = f" (Sniper Entry: Bullish OB Top at {ob_top})"
            print(Fore.CYAN + f"🎯 SMART MONEY DETECTED: Bullish OB found at {ob_top}. High Quality Entry.")
        elif direction == "BEARISH" and bear_ob:
            ob_bot = bear_ob['bottom']
            suggested_entry = ob_bot
            suggestion_msg = f" (Sniper Entry: Bearish OB Bottom at {ob_bot})"
            print(Fore.CYAN + f"🎯 SMART MONEY DETECTED: Bearish OB found at {ob_bot}. High Quality Entry.")
            
        # Priority 2: Fair Value Gap (if no OB)
        elif direction == "BULLISH":
            # Find nearest bullish FVG below/at price
            valid_fvgs = [f for f in fvgs if f['type']=='bullish' and f['top'] <= model_price]
            if valid_fvgs:
                fvg = valid_fvgs[0] # Closest
                suggested_entry = fvg['top']
                suggestion_msg = f" (Sniper Entry: Bullish FVG Top at {fvg['top']})"
                print(Fore.CYAN + f"🎯 SMART MONEY DETECTED: Bullish FVG found at {fvg['top']}. Gap Fill Entry.")
            else:
                print(Fore.WHITE + "ℹ️ No immediate Bullish OB/FVG found.")
                
        elif direction == "BEARISH":
            valid_fvgs = [f for f in fvgs if f['type']=='bearish' and f['bottom'] >= model_price]
            if valid_fvgs:
                fvg = valid_fvgs[0]
                suggested_entry = fvg['bottom']
                suggestion_msg = f" (Sniper Entry: Bearish FVG Bottom at {fvg['bottom']})"
                print(Fore.CYAN + f"🎯 SMART MONEY DETECTED: Bearish FVG found at {fvg['bottom']}. Gap Fill Entry.")
            else:
                 print(Fore.WHITE + "ℹ️ No immediate Bearish OB/FVG found.")
        
        print(Fore.YELLOW + "🎯 OPTIMIZE ENTRY: Enter your specific Level or Press Enter for Recommendation.")
        real_price_input = input(Fore.YELLOW + f"Enter Desired Entry Price [Default {suggested_entry}{suggestion_msg}]: ")
        
        if real_price_input:
            current_price = float(real_price_input)
            print(Fore.GREEN + f"✅ Using Custom Entry: {current_price}")
            
            # CHECK FOR FVG CONFLICTS
            conflicting_fvg = None
            for fvg in fvgs:
                # Check if custom price is INSIDE an FVG zone
                if fvg['bottom'] <= current_price <= fvg['top']:
                    # If BULLISH trade but entering at BEARISH FVG (resistance) = CONFLICT
                    if direction == "BULLISH" and fvg['type'] == 'bearish':
                        conflicting_fvg = fvg
                        print(Fore.RED + f"⚠️ WARNING: Entering LONG at BEARISH FVG (${fvg['bottom']:.2f}-${fvg['top']:.2f})!")
                        print(Fore.YELLOW + "💡 SMC Logic: Bearish FVG acts as RESISTANCE. Price may reverse DOWN.")
                        print(Fore.YELLOW + "💡 Consider: Wait for FVG fill or enter SHORT instead.")
                        break
                    # If BEARISH trade but entering at BULLISH FVG (support) = CONFLICT  
                    elif direction == "BEARISH" and fvg['type'] == 'bullish':
                        conflicting_fvg = fvg
                        print(Fore.RED + f"⚠️ WARNING: Entering SHORT at BULLISH FVG (${fvg['bottom']:.2f}-${fvg['top']:.2f})!")
                        print(Fore.YELLOW + "💡 SMC Logic: Bullish FVG acts as SUPPORT. Price may reverse UP.")
                        print(Fore.YELLOW + "💡 Consider: Wait for FVG fill or enter LONG instead.")
                        break
            
            # If conflict detected, ask for confirmation
            if conflicting_fvg:
                fvg_action = input(Fore.YELLOW + "\nProceed with this entry anyway? (y/n) OR Reverse direction? (r): ").lower()
                if fvg_action == 'r':
                    direction = "BEARISH" if direction == "BULLISH" else "BULLISH"
                    print(Fore.CYAN + f"✅ Direction REVERSED to {direction} based on FVG logic.")
                elif fvg_action != 'y':
                    print(Fore.RED + "Trade cancelled. Adjust your entry level.")
                    return
            
            # Re-Scan SMC for this new price level
            if abs(current_price - model_price) > (atr * 2):
                print(Fore.WHITE + "🔄 Price difference detected. Re-scanning Smart Money Levels...")
                from src.analysis.smc_analyzer import SMCAnalyzer
                smc = SMCAnalyzer(df, timeframe=timeframe)
                new_levels = smc.get_nearest_structures(current_price)
                
                # Show new suggestions
                new_bull = new_levels.get('support_ob')
                new_bear = new_levels.get('resistance_ob')
                
                if direction == "BULLISH" and new_bull:
                     print(Fore.CYAN + f"🎯 CONFIRMATION: Valid Bullish OB found at ${new_bull['top']:.2f} (Time: {new_bull['time']}).")
                elif direction == "BEARISH" and new_bear:
                     print(Fore.CYAN + f"🎯 CONFIRMATION: Valid Bearish OB found at ${new_bear['bottom']:.2f} (Time: {new_bear['time']}).")
                else:
                     print(Fore.WHITE + "ℹ️ No specific Order Block structure found near this custom level.")

        else:
            current_price = float(suggested_entry)
            print(Fore.GREEN + f"✅ Using Suggestion: {current_price}")

        print(f"AI Signal: {Fore.GREEN if direction=='BULLISH' else Fore.RED}{direction} {Fore.WHITE}(Conf: {confidence:.2%})")
        print(f"Start Logic: {Fore.WHITE}Trend is {direction} and DXY is {dxy_trend}")

        direction_input = input(Fore.YELLOW + f"Confirm Direction (Default: {direction} based on AI): ").upper()
        
        if not direction_input:
            direction_input = "BUY" if direction == "BULLISH" else "SELL" if direction == "BEARISH" else "NEUTRAL"
            
        if direction_input == "NEUTRAL":
             print(Fore.RED + "AI is Neutral. No trade recommended.")
             return

        rm = RiskManager(bankroll=balance)
        plan = rm.generate_trade_plan(
            entry=current_price, 
            direction=direction_input, 
            atr=atr, 
            user_risk_pct=risk_pct
        )
        
        # 2. Display Plan
        print("\n" + Fore.CYAN + "-"*40)
        print(Fore.WHITE + "TRADING EXECUTION PLAN")
        print(Fore.CYAN + "-"*40)

        # Check for Forced Min Lot Warning
        if sizing.get('is_forced') and sizing['risk_amount'] > (balance * risk_pct):
             print(Fore.RED + "⚠️  WARNING: ACCOUNT TOO SMALL FOR SAFE RISK RULES.")
             print(Fore.RED + f"   Min Lot (0.01) Risk: ${sizing['risk_amount']:.2f}")
             print(Fore.RED + f"   Your {risk_pct*100}% Risk Limit: ${balance*risk_pct:.2f}")
             print(Fore.YELLOW + "   Trade exceeds your risk tolerance. Reduce Stop Loss (Use 15m) or Deposit Funds.")
             print(Fore.CYAN + "-"*40)
        
        
        headers = ["Metric", "Value"]
        table = [
            ["Action", f"{Fore.GREEN if plan['action']=='BUY' else Fore.RED}{plan['action']}"],
            ["Entry Price", f"{plan['entry']}"],
            ["Stop Loss (SL)", f"{Fore.RED}{plan['sl']} ({plan['sl_pips']} pips)"],
            ["Take Profit (TP)", f"{Fore.GREEN}{plan['tp']} ({plan['tp_pips']} pips)"],
            ["Risk/Reward", f"1 : {plan['rr_ratio']}"],
            ["Risk Amount", f"{Fore.RED}-${plan['risk_usd']:.2f}"],
            ["Projected Profit", f"{Fore.GREEN}+${plan['reward_usd']:.2f}"],
            ["LOT SIZE", f"{Fore.YELLOW}{Style.BRIGHT}{plan['lot_size']} Lots"]
        ]
        
        print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        
        # 3. Guidance
        print("\n" + Fore.WHITE + "💡 GUIDANCE:")
        print(f"* You have prioritized risking {Fore.RED}{risk_pct*100}%{Fore.WHITE} of your account.")
        print(f"* If this trade fails, you lose {Fore.RED}${plan['risk_usd']:.2f}{Fore.WHITE}.")
        print(f"* If it hits TP, you gain {Fore.GREEN}${plan['reward_usd']:.2f}{Fore.WHITE}.")
        print(f"* Logic: Tight {plan['sl_pips']} pip SL protected by volatility checks.")
        
        # 4. News Check (Bonus)
        dp = DataProvider()
        news = dp.get_high_impact_usd_news()
        if not news.empty:
             print(f"\n{Fore.YELLOW}⚠️ NEWS WARNING: High volatility expected today.")
        else:
             print(f"\n{Fore.GREEN}✅ Market Conditions: Stable (No high impact news detected).")

    except ValueError:
        print(Fore.RED + "Invalid Input. Please enter numbers for balance/risk.")

if __name__ == "__main__":
    run_planner()

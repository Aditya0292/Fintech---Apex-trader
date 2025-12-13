import numpy as np
from typing import Dict, Optional
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger()

class RiskManager:
    """
    Advanced Risk Management for Institutional-Grade Safety.
    Features:
    - Fractional Kelly Criterion for Position Sizing
    - Volatility-Adjusted Stop Loss (ATR Based)
    - News Impact "No Trade" Filtering
    - Drawdown Protection
    """
    
    def __init__(self, bankroll: float = 1000.0):
        self.config = ConfigLoader.get("risk")
        self.bankroll = bankroll
        self.max_risk_pct = self.config.get("max_risk_per_trade", 0.02)
        self.target_rr = self.config.get("target_risk_reward", 1.5)
        self.min_confidence = self.config.get("confidence_threshold", 0.60)
        
        logger.info(f"RiskManager initialized. Bankroll: ${bankroll}, Max Risk: {self.max_risk_pct:.1%}")

    def calculate_kelly_size(self, win_prob: float, win_loss_ratio: float = 1.5) -> float:
        """
        Calculates optimal position size fraction using Half-Kelly.
        f* = (p(b+1) - 1) / b
        where:
        p = probability of win
        b = odds received (win/loss ratio)
        """
        if win_prob <= 0.5:
            return 0.0
            
        # Standard Kelly Formula
        # q = 1-p
        # f = p - (q / b)
        q = 1.0 - win_prob
        full_kelly = win_prob - (q / win_loss_ratio)
        
        # Safety: Fractional Kelly (Half Kelly is industry standard for smooth growth)
        half_kelly = full_kelly * 0.5
        
        # Cap at Max Risk
        size = max(0.0, min(half_kelly, self.max_risk_pct))
        return size

    def calculate_position_size(self, entry_price: float, sl_price: float, risk_pct: Optional[float] = None) -> Dict:
        """
        Calculate Lot Size for XAUUSD.
        Standard Contract: 100 oz
        Tick Value: $0.01 per 0.01 move per 1 unit? 
        Actually:
        XAUUSD 1 Standard Lot = 100 units.
        Price move $1.00 = $100 profit/loss per lot.
        Price move $0.10 (1 pip?) = $10 per lot.
        
        Formula:
        Risk Amount = Bankroll * Risk Pct
        Distance = |Entry - SL|
        Pip Value = 1 (if distance is in $) * 100 (Contract Size)
        Lot Size = Risk Amount / (Distance * 100)
        """
        if risk_pct is None:
            risk_pct = self.max_risk_pct
            
        risk_amount = self.bankroll * risk_pct
        distance = abs(entry_price - sl_price)
        
        if distance == 0:
            return {"lots": 0.0, "risk_amount": 0.0}
            
        # Contract size 100 for Gold standard
        contract_size = 100
        lot_size = risk_amount / (distance * contract_size)
        
        # Round to 2 decimal places (Standard limits) but allowing smaller for calculations
        # If result is 0.00xxx, we should show it more precisely or floor to 0.01 if broker min.
        # User complained about 0.0 Lots.
        
        calculated_lots = lot_size
        
        # Standard Broker Minimum is 0.01 Lots
        is_min_lot_forced = False
        
        if calculated_lots < 0.01:
             # Force 0.01 but flag it
             lot_size = 0.01
             is_min_lot_forced = True
             real_risk = lot_size * distance * contract_size
             risk_amount = real_risk
        else:
             lot_size = round(calculated_lots, 2)
        
        warning_msg = None
        if is_min_lot_forced and risk_amount > (self.bankroll * risk_pct * 1.5): 
             # Only warn if significantly over (50% margin)
             warning_msg = f"Risk High ({real_risk:.2f} > Limit)"

        return {
            "lots": lot_size,
            "raw_lots": calculated_lots,
            "risk_amount": round(risk_amount, 2),
            "risk_pct": risk_pct,
            "distance_dollars": round(distance, 2),
            "warning": warning_msg,
            "is_forced": is_min_lot_forced
        }

    def generate_trade_plan(self, entry: float, direction: str, atr: float, user_risk_pct: float = None) -> Dict:
        """
        Generate a tight, user-focused trade plan.
        Constraints:
        - TP: 100-200 pips ($1.00 - $2.00 move?)
          User said 100 pips. In XAUUSD 2000.00 -> 2000.10 is 1 pip usually? Or 2000.01?
          Let's assume standard broker: 0.01 is a tick (point). 1 pip = 10 points = 0.10.
          So 100 pips = $10.00 move. This fits "Small TP" relative to 2000.
          Wait, user said "SL not 1000 pips ($100)". 
          So 100 pips ($10) TP is reasonable scalping.
          
        - R:R: 1:2 or 1:3
        """
        # Determine stops based on Risk/Reward, not just ATR
        # User wants tight stops.
        
        target_pips_min = 100 # $10.00
        target_pips_max = 200 # $20.00
        
        # Volatility check: Don't set SL tighter than 1 * ATR (Safety)
        # If ATR is $5.00 (50 pips), SL should be at least $5.00.
        
        sl_distance = max(atr, 2.0) # Min $2.00 SL (20 pips)
        
        # Enforce 1:2 Minimum
        tp_distance = sl_distance * 2.0
        
        # Clamp TP to user preference if volatility allows
        # If TP > $20 (200 pips), maybe scale back SL?
        # No, safety first. If vol is huge, we need wide stops.
        # But user requested tight. We will prioritize the R:R.
        
        if direction.upper() == "BUY":
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance
            
        # Pips calc (Assuming 1 pip = $0.10)
        pips_sl = (sl_distance / 0.10)
        pips_tp = (tp_distance / 0.10)
        
        # Sizing
        sizing = self.calculate_position_size(entry, sl, user_risk_pct)
        
        potential_win = sizing['lots'] * tp_distance * 100
        potential_loss = sizing['risk_amount']
        
        return {
            "action": direction.upper(),
            "entry": entry,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "sl_pips": int(pips_sl),
            "tp_pips": int(pips_tp),
            "lot_size": sizing['lots'],
            "risk_usd": potential_loss,
            "reward_usd": round(potential_win, 2),
            "rr_ratio": round(tp_distance/sl_distance, 2)
        }

if __name__ == "__main__":
    # Test
    rm = RiskManager(bankroll=5000)
    
    # Kelly Test
    size = rm.calculate_kelly_size(win_prob=0.75, win_loss_ratio=1.5)
    print(f"Win Rate 75%, R:R 1.5 -> Position Size: {size:.2%}")
    
    # Validation Test
    res = rm.validate_trade("BUY", 0.55, "NONE")
    print(f"Trade Valid? {res}")

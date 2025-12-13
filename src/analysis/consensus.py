from typing import Dict, List, Optional

class ConsensusEngine:
    """
    Multi-Timeframe Consensus Engine.
    Resolves conflicts between HTF (Daily/4H) and LTF (1H/15m) models.
    """
    
    def __init__(self):
        pass
        
    def resolve(self, signals: Dict[str, str], confidences: Dict[str, float]) -> Dict:
        """
        signals: {'Daily': 'BULLISH', '4H': 'BEARISH', '1H': 'BULLISH', '15m': 'NEUTRAL'}
        confidences: {'Daily': 0.75, ...}
        """
        
        # Standardize inputs
        htf_signal = signals.get('4 Hour', 'NEUTRAL')
        ltf_signal = signals.get('15 Min', signals.get('1 Hour', 'NEUTRAL'))
        
        htf_conf = confidences.get('4 Hour', 0.0)
        ltf_conf = confidences.get('15 Min', confidences.get('1 Hour', 0.0))
        
        decision = "WAIT"
        reason = "Indecision"
        action_type = "hold"
        
        # Logic Matrix
        if htf_signal == 'BULLISH':
            if ltf_signal == 'BULLISH':
                decision = "STRONG BUY"
                reason = "Confluence: Trend Alignment (HTF + LTF Bullish)"
                action_type = "trend_follow"
            elif ltf_signal == 'BEARISH':
                decision = "BUY DIP?"
                reason = "Conflict: HTF Bullish vs LTF Bearish (Pullback?)"
                action_type = "counter_trend_entry"
            else:
                decision = "BUY (WEAK)"
                reason = "HTF Bullish, LTF Neutral"
                
        elif htf_signal == 'BEARISH':
            if ltf_signal == 'BEARISH':
                decision = "STRONG SELL"
                reason = "Confluence: Trend Alignment (HTF + LTF Bearish)"
                action_type = "trend_follow"
            elif ltf_signal == 'BULLISH':
                decision = "SELL RIP?"
                reason = "Conflict: HTF Bearish vs LTF Bullish (Rally?)"
                action_type = "counter_trend_entry"
            else:
                decision = "SELL (WEAK)"
                reason = "HTF Bearish, LTF Neutral"
                
        else: # HTF Neutral
            if ltf_signal != 'NEUTRAL':
                decision = f"SCALP {ltf_signal}"
                reason = "HTF Neutral, Playing LTF Momentum"
                action_type = "scalp"
                
        return {
            "decision": decision,
            "reason": reason,
            "action_type": action_type,
            "net_confidence": (htf_conf + ltf_conf) / 2
        }

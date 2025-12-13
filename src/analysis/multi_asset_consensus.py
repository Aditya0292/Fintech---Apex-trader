
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger()

class MultiAssetConsensus:
    """
    Aggregates predictions across multiple assets to find the highest-probability setup.
    """
    
    def __init__(self):
        pass
        
    def rank_assets(self, predictions: Dict[str, Dict]) -> List[Dict]:
        """
        Rank assets by confidence and confluence.
        predictions: { 'XAUUSD': result_dict, 'EURUSD': result_dict, ... }
        """
        ranked = []
        
        for symbol, res in predictions.items():
            # Extract key metrics
            direction = res.get('prediction', 'Neutral')
            confidence = res.get('confidence', 0.0)
            
            # SMC Quality
            smc = res.get('smc', {})
            has_ob = 1 if (direction == 'Bullish' and smc.get('support_ob')) or \
                          (direction == 'Bearish' and smc.get('resistance_ob')) else 0
            
            # Microstructure
            micro = res.get('microstructure', {})
            displacement = 1 if micro.get('displacement') else 0
            
            # Context
            context = res.get('context', {})
            dxy_alignment = 0
            if symbol == 'XAUUSD' or symbol.endswith('USD'):
                # Bearish DXY supports Bullish Asset
                dxy_trend = context.get('dxy_trend', 'Neutral')
                if direction == 'Bullish' and dxy_trend == 'Bearish': dxy_alignment = 1
                if direction == 'Bearish' and dxy_trend == 'Bullish': dxy_alignment = 1
            
            # Score Calculation
            # Base = Confidence * 100
            # Bonuses: OB (+10), Displace (+10), DXY (+15)
            score = (confidence * 100) + (has_ob * 10) + (displacement * 10) + (dxy_alignment * 15)
            
            ranked.append({
                'symbol': symbol,
                'direction': direction,
                'score': round(score, 1),
                'confidence': f"{confidence:.1%}",
                'factors': {
                    'structure': bool(has_ob),
                    'momentum': bool(displacement),
                    'macro': bool(dxy_alignment)
                }
            })
            
        # Sort by score desc
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked

    def generate_report(self, ranked_assets: List[Dict]) -> str:
        if not ranked_assets: return "No data for consensus."
        
        from tabulate import tabulate
        
        # Table Data
        table_data = []
        for asset in ranked_assets:
            factors = asset['factors']
            confluence_str = f"S:{factors['structure']}/M:{factors['momentum']}/E:{factors['macro']}"
            # Scale score_bar to a reasonable length, e.g., max 10 chars for a score of 100
            # Max possible score is 135, so divide by 13.5 to get max 10 blocks
            score_bar_length = min(10, int(asset['score'] / 13.5)) 
            score_bar = "█" * score_bar_length
            
            # Safe confidence display
            conf_val = asset['confidence']
            if isinstance(conf_val, str):
                conf_display = conf_val
            else:
                conf_display = f"{conf_val:.1%}"

            table_data.append([
                asset['symbol'],
                asset['direction'],
                conf_display,
                f"{asset['score']:.2f}",
                confluence_str,
                score_bar
            ])
            
        report = "\n   🏆 OPPORTUNITY RUNWAY (Sorted by Confluence Score)\n"
        report += tabulate(table_data, headers=["Asset", "Action", "Conf", "Score", "Factors (Str/Mom/Mac)", "Rating"], tablefmt="simple")
        
        # Best Pick Details
        best = ranked_assets[0]
        
        # Safe format
        b_conf = best['confidence']
        b_conf_str = b_conf if isinstance(b_conf, str) else f"{b_conf:.1%}"
        
        report += f"\n\n   💎 TOP PICK: {best['symbol']} ({best['direction']})\n"
        report += f"      Why? High Confidence ({b_conf_str}) & Robust Factors.\n"
        
        return report

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.utils.logger import get_logger
from src.utils.time_utils import normalize_ts
from src.utils.config_loader import config

logger = get_logger()

class NewsManager:
    """
    Manages news fetching, sentiment analysis, and impact scoring per currency.
    """
    
    def __init__(self):
        self.azure_enabled = config.get('azure', {}).get('use_simulation', True) is False
        self.news_weight = 1.0
        # Lazy load DataProvider to avoid circular imports if any
        self.dp = None

    def get_asset_events(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Fetch recent/upcoming high-impact events for the asset's currencies.
        """
        from src.data.data_provider import DataProvider
        if not self.dp: self.dp = DataProvider()
        
        # Get currencies
        assets_conf = config.get('assets', {})
        conf = assets_conf.get(symbol, {})
        currencies = conf.get('csm_currencies', [])
        if not currencies:
            base = symbol[:3]
            quote = symbol[3:]
            currencies = [base, quote]
            
        all_events = self.dp.fetch_economic_calendar()
        if not all_events: return []
        
        filtered = []
        for e in all_events:
            # Filter by Currency
            if e.get('country') in currencies:
                # Filter by Impact
                if e.get('impact') in ['High', 'Medium']:
                    filtered.append({
                        'time': e.get('date'),
                        'currency': e.get('country'),
                        'event': e.get('title'),
                        'impact': e.get('impact'),
                        'actual': e.get('actual', ''),
                        'forecast': e.get('forecast', '')
                    })
        
        # Sort by time (descending or ascending? dashboard usually wants recent)
        # Assuming date string sort works or convert to datetime
        # The XML date is YYYY-MM-DD HH:MM so string sort is fine
        filtered.sort(key=lambda x: x['time'], reverse=True) 
        
        return filtered[:limit]
        """
        Returns global USD/Market impact series (compatibility mode).
        """
        return self.get_currency_impact_series(dates, "USD")

    def get_currency_impact_series(self, dates: pd.DatetimeIndex, currency: str) -> pd.Series:
        """
        Generates a continuous impact score series for a specific currency based on news events.
        Apply decay.
        """
        # TODO: Fetch real historical events for the currency
        # For now, use the simulation logic but randomized appropriately or 
        # driven by a deterministic seed for consistency.
        
        logger.info(f"Generating synthetic news impact for {currency}")
        
        # Create sparse events
        n_events = len(dates) // 100 # 1% of bars have news
        if n_events == 0: n_events = 1
        
        event_indices = np.random.choice(len(dates), n_events, replace=False)
        impacts = np.zeros(len(dates))
        
        # Assign random impacts (-1 to 1) 
        impacts[event_indices] = np.random.uniform(-1, 1, n_events) * 2.0 # Raw Impact
        
        # Apply exponential decay
        series = pd.Series(impacts, index=dates)
        decayed = series.ewm(halflife='2h', times=dates).mean()
        
        # Scale
        return decayed

    def aggregate_impact_for_symbol(self, dates: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """
        Aggregates news impact for a symbol based on its base/quote currencies.
        Reads config from assets.yaml via config dict.
        """
        assets = config.get('assets', {})
        asset_conf = assets.get(symbol, {})
        
        currencies = asset_conf.get('csm_currencies', [])
        if not currencies:
            # Fallback for symbols not in config (e.g. BTCUSD?)
            if symbol.endswith('USD'): currencies = ['USD']
            else: currencies = ['USD']
            
        weight = asset_conf.get('news_weight', 1.0)
        
        impact_df = pd.DataFrame(index=dates)
        total_impact = pd.Series(0.0, index=dates)
        
        for curr in currencies:
            imp = self.get_currency_impact_series(dates, curr)
            impact_df[f'news_impact_{curr}'] = imp
            
            # CP: Determine polarity
            # If curr is Base (starts with), impact is Positive for Pair.
            # If curr is Quote (ends with), impact is Negative for Pair.
            # e.g. EURUSD: EUR is Base (+), USD is Quote (-)
            # e.g. USDJPY: USD is Base (+), JPY is Quote (-)
            
            polarity = 1.0
            if symbol.endswith(curr): # Quote
                polarity = -1.0
            elif symbol.startswith(curr): # Base
                 polarity = 1.0
            
            # For XAUUSD, USD is Quote (Ends with USD), so -1.0. Correct.
            
            total_impact += (imp * polarity)
            
        impact_df['news_impact_net'] = total_impact * weight
        
        return impact_df

import yaml
import os
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger()

class ConfigLoader:
    _config = None
    DEFAULT_CONFIG_PATH = "src/config/config.yaml"

    @classmethod
    def load_config(cls, path=None):
        if cls._config is not None and path is None:
            return cls._config

        config_path = path or cls.DEFAULT_CONFIG_PATH
        
        # Default Config Structure
        default_config = {
            "risk": {
                "max_risk_per_trade": 0.02,
                "target_risk_reward": 1.5,
                "max_drawdown_limit": 0.10,
                "confidence_threshold": 0.60
            },
            "data": {
                "pairs": ["XAUUSD"],
                "timeframes": ["15m", "1h", "4h", "1d"],
                "history_length_days": 1000,
                "news_cache_seconds": 600
            },
            "azure": {
                "openai_model": "gpt-4o",
                "anomaly_detector_endpoint": "https://simulated-endpoint",
                "use_simulation": True
            },
            "models": {
                "ensemble_weights": {"xgb": 0.4, "lstm": 0.3, "lgbm": 0.3}
            },
            "features": {
                "window_size": 50,
                "min_periods": 200,
                "swing_length": 10,
                "movement_threshold": 0.001,
                "fvg_min_size": 0.0002,
                "pivot_types": ["traditional", "fibonacci", "camarilla", "woodie", "demark"]
            }
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Deep merge is better, but simple update for now
                    default_config.update(user_config or {})
                    logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
        else:
            logger.warning(f"Config file not found at {config_path}. Using Defaults.")
            # Optionally create it?
            
        # Load Assets Config
        assets_path = "src/config/assets.yaml"
        if os.path.exists(assets_path):
             try:
                with open(assets_path, 'r') as f:
                    assets_config = yaml.safe_load(f)
                    default_config['assets'] = assets_config
                    logger.info(f"Loaded assets config from {assets_path}")
             except Exception as e:
                logger.error(f"Failed to load assets config: {e}")
        
        cls._config = default_config
        return default_config

    @classmethod
    def get(cls, section, key=None):
        """Safe getter for config values"""
        cfg = cls.load_config()
        sec = cfg.get(section, {})
        if key:
            return sec.get(key)
        return sec

# Global Accessor
config = ConfigLoader.load_config()

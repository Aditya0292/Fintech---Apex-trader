import logging
import logging.handlers
import os
import yaml
from pathlib import Path

def load_config():
    config_path = Path("src/config/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
log_config = config.get("logging", {})

def get_logger(name="apex_ai"):
    logger = logging.getLogger(name)
    
    # If logger already has handlers, assume initialized
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, log_config.get("level", "INFO").upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(module)s:%(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    import sys
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler (Rotating)
    log_file = log_config.get("file", "apex_trade_ai.log")
    if log_file:
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=log_config.get("rotate_max_bytes", 10485760), 
                backupCount=log_config.get("rotate_backup_count", 5)
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")
            
    return logger

import os
import pickle
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from src.utils.logger import get_logger
from src.utils.config_loader import config

logger = get_logger()

class ModelLoader:
    @staticmethod
    def load(path: str, expected_features: int = 71):
        """
        Loads a model (Keras .h5 or Sklearn .pkl) and validates input compatibility.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        logger.info(f"Loading model from {path}...")
        
        try:
            if path.endswith('.h5') or path.endswith('.keras'):
                model = keras_load_model(path, compile=False)
                # Check Keras Input Shape
                # shape is usually (None, steps, features) or (None, features)
                input_shape = model.input_shape
                
                # Handle list of inputs or single input
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]
                    
                model_features = input_shape[-1]
                
                if model_features != expected_features:
                     # Warn but allow if it's a specific sub-model, but for main pipeline strictly warn
                     logger.warning(f"Shape Mismatch: Model expects {model_features} features, but schema defines {expected_features}.")
                     # raise ValueError(f"Model expects {model_features} features, schema has {expected_features}")
                else:
                    logger.info(f"Model input shape validated: {model_features} features.")
                    
                return model
                
            elif path.endswith('.pkl'):
                # Try pickle then joblib
                try:
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                except:
                    model = joblib.load(path)
                    
                # Check Sklearn n_features_in_
                if hasattr(model, 'n_features_in_'):
                    if model.n_features_in_ != expected_features:
                         logger.warning(f"Shape Mismatch: Model expects {model.n_features_in_} features, but schema defines {expected_features}.")
                    else:
                        logger.info(f"Model features validated: {model.n_features_in_}")
                
                return model
            else:
                raise ValueError(f"Unsupported file format: {path}")
                
        except Exception as e:
            logger.error(f"Failed to load model {path}: {e}")
            raise e

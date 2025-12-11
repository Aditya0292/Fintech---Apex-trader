
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape, Concatenate, Add
from tensorflow.keras.optimizers import Adam

class ModelFactory:
    @staticmethod
    def get_xgboost(input_dim):
        """Returns complex XGBoost Classifier"""
        # Note: input_dim not strictly used for init, but kept for signature consistency
        return xgb.XGBClassifier(
            n_estimators=200, 
            learning_rate=0.03, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            n_jobs=-1,
            random_state=42
        )
    
    @staticmethod
    def get_lightgbm(input_dim):
        """Returns complex LightGBM Classifier"""
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            objective='multiclass',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )
        
    @staticmethod
    def get_bilstm_attention(input_shape):
        """
        Bidirectional LSTM with Self-Attention.
        input_shape: (time_steps, features)
        """
        inputs = Input(shape=input_shape)
        
        # BiLSTM
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm_out = LayerNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Self Attention via MultiHead
        # Query, Key, Value = lstm_out
        attn_out = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(lstm_out, lstm_out)
        attn_out = LayerNormalization()(attn_out + lstm_out) # Residual
        
        # Pooling
        gap = GlobalAveragePooling1D()(attn_out)
        
        # Dense Head
        x = Dense(64, activation='relu')(gap)
        x = Dropout(0.3)(x)
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attention")
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
        
    @staticmethod
    def get_transformer(input_shape):
        """
        Transformer Encoder Block.
        """
        inputs = Input(shape=input_shape)
        
        # Embedding / Projection (Optional, here we skip and go direct to encoder)
        x = Dense(64)(inputs) # Project to d_model=64
        
        # Encoder Block 1
        attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization()(Add()([x, attn_out]))
        
        # Feed Forward
        ffn = Dense(128, activation='relu')(x)
        ffn = Dense(64)(ffn)
        x = LayerNormalization()(Add()([x, ffn]))
        
        # Global Pooling
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.3)(x)
        
        # Head
        x = Dense(32, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="Transformer_Encoder")
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

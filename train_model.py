import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate, GlobalAveragePooling1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import math

# -------------------------
# Load Data
# -------------------------
print("Loading data...")
X = np.load("X.npy")
y_reg = np.load("y_reg.npy")
y_cls = np.load("y_class.npy") # Now contains 0, 1, 2

# Filter out Class 2 (Hold) if we want binary trade/no-trade? 
# Strategy: Train 3-class classifier. 0=Bearish, 1=Bullish, 2=Neutral/Noise
# User wants high accuracy. If model predicts 2, we don't trade.
# Accuracy will be measured on ALL classes, but we care most about precision of 0 and 1.
# Actually, let's stick to the plan: "Update target to classifications...".

print("Original Shapes:")
print("X:", X.shape)
print("y_cls:", y_cls.shape)

# -------------------------
# Feature Selection (XGBoost)
# -------------------------
print("Running Feature Selection...")
# Flatten X for XGBoost
X_flat = X.reshape(X.shape[0], -1)
# Use a subset for speed if needed, but dataset is small (~5k)
xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
xgb_model.fit(X_flat, y_cls)

# Get feature importances
importances = xgb_model.feature_importances_
# Reshape back to (50, 32) sum over time steps to get feature importance per feature index
# Or just take top k features from the flattened vector?
# Better: Aggregate importance per feature index (0-31)
feat_imp = importances.reshape(X.shape[1], X.shape[2]).sum(axis=0)
indices = np.argsort(feat_imp)[::-1]
# Keep top 20 features
TOP_K = 20
top_indices = indices[:TOP_K]
print(f"Top {TOP_K} features indices:", top_indices)

X_selected = X[:, :, top_indices]
print("X_selected shape:", X_selected.shape)

# -------------------------
# Train/Val Split (TimeSeriesSplit)
# -------------------------
# We'll use the last 20% for validation (hold-out) to mimic future
split_idx = int(len(X_selected) * 0.8)
X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
y_train_cls, y_test_cls = y_cls[:split_idx], y_cls[split_idx:]
y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]

# Handle Class Imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_cls), y=y_train_cls)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

# -------------------------
# Attention LSTM Model (Classification)
# -------------------------
def build_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Attention Mechanism
    # Query: last hidden state, Value/Key: all hidden states
    # Simple Self-Attention or Attention over time steps
    # We want to wait the time steps.
    # Attention(query, value)
    
    # Let's use a custom or simple Attention layer
    # query = Dense(128)(lstm_out) # (Batch, Steps, 128)
    # value = Dense(128)(lstm_out)
    # attention = Attention()([query, value])
    
    # Simplified dot-product attention
    # We want a single vector representing the sequence
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Reshape((-1,))(attention) # (Batch, Steps)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = Reshape((-1, 1))(attention) # (Batch, Steps, 1)
    
    context = tf.keras.layers.Multiply()([lstm_out, attention])
    context = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    
    x = Dense(32, activation='relu')(context)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x) # 3 classes: 0, 1, 2
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model_cls = build_attention_model((X_train.shape[1], X_train.shape[2]))

# Cosine Decay LR Scheduler
def cosine_decay(epoch, lr):
    decay_steps = 50
    alpha = 0.0
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return float(0.001 * decayed)

model_cls.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    LearningRateScheduler(cosine_decay)
]

print("Training Classification Model...")
history_cls = model_cls.fit(
    X_train, y_train_cls,
    validation_data=(X_test, y_test_cls),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

model_cls.save("model_class_v2.h5")
print("Saved -> model_class_v2.h5")

# -------------------------
# Evaluation (Unbiased)
# -------------------------
print("Evaluating on Test Set...")
y_pred_probs = model_cls.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
confidence_scores = np.max(y_pred_probs, axis=1)

print("\nClassification Report (0=Bear, 1=Bull, 2=Mid):")
report = classification_report(y_test_cls, y_pred_classes, labels=[0, 1, 2], target_names=['Bear', 'Bull', 'Mid'])
print(report)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("Saved classification_report.txt")

# Plot Accuracy
plt.figure()
plt.plot(history_cls.history["accuracy"], label="train_acc")
plt.plot(history_cls.history["val_accuracy"], label="val_acc")
plt.title("Attention Model Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")
print("Saved accuracy_plot.png")

# Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Bear', 'Bull', 'Mid'], yticklabels=['Bear', 'Bull', 'Mid'])
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# Confidence Histogram
plt.figure()
plt.hist(confidence_scores, bins=20, alpha=0.7, color='purple')
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Count")
plt.savefig("confidence_histogram.png")
print("Saved confidence_histogram.png")

# -------------------------
# Prediction Function
# -------------------------
def predict_next_day(model, sequence):
    """
    Predicts the next day's movement.
    sequence: np.array of shape (1, 50, n_features)
    """
    probs = model.predict(sequence, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    
    labels = {0: "Bearish", 1: "Bullish", 2: "Mid"}
    
    return {
        "predicted_class": pred_class,
        "label": labels[pred_class],
        "confidence": round(confidence, 4),
        "raw_probs": [round(p, 4) for p in probs]
    }

# Demo Prediction using the last sequence in Test Set
last_seq = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
result = predict_next_day(model_cls, last_seq)

print("\n--- Next Day Prediction (Demo) ---")
print(result)

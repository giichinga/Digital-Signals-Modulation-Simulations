"""
classifier.py
=============
Neural network that learns ADAPTIVE MODULATION SELECTION.

Input  : SNR_dB only  (the one thing a real transmitter can estimate
         ahead of time, e.g. via pilot symbols / channel sounding)
Output : Best_Modulation (BPSK / QPSK / 16-QAM)

NOTE ON TARGET LEAKAGE (why BER columns are not inputs):
    modulation_dataset.csv also contains BER_BPSK, BER_QPSK, BER_16QAM.
    Those were used to CREATE the Best_Modulation label at dataset-
    generation time (see get_best_modulation() in dataset_generator.py).
    If we also fed those same BER values into the model as features, the
    model would just be re-deriving the label from its own ingredients
    (trivial 100% accuracy that means nothing). We deliberately exclude
    them here and use SNR_dB alone.

Run:
    python classifier.py
Requires modulation_dataset.csv (from dataset_generator.py).
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from scipy.special import erfc

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ── 1. Load dataset ─────────────────────────────────────────────────────
df = pd.read_csv("modulation_dataset.csv")

FEATURE_COLUMNS = ["SNR_dB"]   # <-- the whole fix lives in this one line

X = df[FEATURE_COLUMNS].values
y = df["Best_Modulation"].values


# ── 2. Encode labels ─────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)


# ── 3. Train/test split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")


# ── 4. Scale features ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ── 5. Build the network ─────────────────────────────────────────────────
# Small on purpose: 1 input feature, a simple decision boundary. A large
# network here would just be overfitting capacity we don't need.
model = Sequential([
    Input(shape=(len(FEATURE_COLUMNS),)),
    Dense(16, activation="relu"),
    Dense(16, activation="relu"),
    Dense(3,  activation="softmax"),
])
model.summary()


# ── 6. Compile ────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ── 7. Train ──────────────────────────────────────────────────────────────
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
)


# ── 8. Evaluate ───────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")


# ── 9. Confusion matrix + report ─────────────────────────────────────────
y_pred = model.predict(X_test_scaled).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix")
print(f"Classes: {list(le.classes_)}")
print(cm)
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ── 10. Plot training history ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Validation")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.legend()
ax2.plot(history.history["loss"],     label="Train")
ax2.plot(history.history["val_loss"], label="Validation")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.show()


# ── 11. Decision boundary plot (SNR -> predicted modulation) ────────────
# This is the plot that visually proves the model learned the right
# switchover points, not just memorised rows.
snr_sweep = np.linspace(df["SNR_dB"].min(), df["SNR_dB"].max(), 400).reshape(-1, 1)
snr_sweep_scaled = scaler.transform(snr_sweep)
probs_sweep = model.predict(snr_sweep_scaled, verbose=0)

fig2, ax = plt.subplots(figsize=(9, 5))
for i, cls in enumerate(le.classes_):
    ax.plot(snr_sweep.ravel(), probs_sweep[:, i], label=cls)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Predicted probability")
ax.set_title("Learned Decision Boundary: SNR -> Modulation Choice")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
plt.show()


# ── 12. Persist model + preprocessing ────────────────────────────────────
model.save("modulation_classifier.keras")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")
print("\nSaved: modulation_classifier.keras, scaler.joblib, label_encoder.joblib")


# ── 13. Manual predictions across the three regimes ──────────────────────
def predict_modulation(snr_db):
    features = np.array([[snr_db]])
    features_scaled = scaler.transform(features)
    probs = model.predict(features_scaled, verbose=0)[0]
    class_idx = np.argmax(probs)
    modulation = le.inverse_transform([class_idx])[0]
    print(f"\nSNR = {snr_db} dB")
    print(f"  Probabilities -> {dict(zip(le.classes_, probs.round(3)))}")
    print(f"  Decision      -> {modulation}")
    return modulation

print("\n" + "=" * 60)
print("  Manual predictions across SNR regimes")
print("=" * 60)
predict_modulation(2)    # expect BPSK  (low SNR)
predict_modulation(7)    # expect QPSK  (mid SNR)
predict_modulation(15)   # expect 16-QAM (high SNR)
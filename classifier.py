import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore[import-not-found]
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore[import-not-found]
from scipy.special import erfc  # type: ignore[import-not-found]
import tensorflow as tf  # type: ignore[import-not-found]
from tensorflow import keras  # type: ignore[import-not-found]

# 1. Load
df = pd.read_csv("modulation_dataset.csv")

# 2. Features and labels
X = df[["SNR_dB", "BER_BPSK", "BER_QPSK", "BER_16QAM"]].values
y = df["Best_Modulation"].values

# 3. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {le.classes_}")   # shows order: 16-QAM=0, BPSK=1, QPSK=2

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

# 5. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 6. Build model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3,  activation='softmax')
])
model.summary()

# 7. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Train — ONCE, capture history
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)

# 9. Evaluate
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

# 10. Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy'); ax1.legend()
ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss'); ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss'); ax2.legend()
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# 11. Confusion matrix
y_pred = model.predict(X_test_scaled).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"Classes: {le.classes_}")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 12. Manual predictions
def predict_modulation(snr_db, model, scaler, le):
    EbN0_linear = 10 ** (snr_db / 10)
    ber_bpsk  = 0.5  * erfc(np.sqrt(EbN0_linear))
    ber_qpsk  = 0.5  * erfc(np.sqrt(EbN0_linear))
    ber_qam16 = 0.75 * erfc(np.sqrt(0.4 * EbN0_linear))
    features        = np.array([[snr_db, ber_bpsk, ber_qpsk, ber_qam16]])
    features_scaled = scaler.transform(features)
    probs      = model.predict(features_scaled, verbose=0)[0]
    class_idx  = np.argmax(probs)
    modulation = le.inverse_transform([class_idx])[0]
    print(f"\nSNR = {snr_db} dB")
    print(f"  BERs     → BPSK: {ber_bpsk:.6f}, QPSK: {ber_qpsk:.6f}, 16-QAM: {ber_qam16:.6f}")
    print(f"  Probs    → {dict(zip(le.classes_, probs.round(3)))}")
    print(f"  Decision → {modulation}")

predict_modulation(2,  model, scaler, le)
predict_modulation(7,  model, scaler, le)
predict_modulation(15, model, scaler, le)
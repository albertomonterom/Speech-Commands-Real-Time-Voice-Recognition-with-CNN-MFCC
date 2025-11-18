import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from config import FEATURES_PATH, MFCC_FEATURES, MFCC_FRAMES, TEST_SIZE, RANDOM_STATE, MODELS_DIR
import os

# === Cargar features ===
data = np.load(FEATURES_PATH)
X = data["X"]                # (N, 13, 32)
y_enc = data["y"]
classes = data["classes"]

# Añadir canal (CNN requiere 4D)
X = X[..., np.newaxis]       # (N, 13, 32, 1)

# One-hot encoding de etiquetas
y_onehot = to_categorical(y_enc, num_classes=len(classes))

# === Dividir en train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_enc
)

# === Definir modelo CNN ===
model = Sequential([
    Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(39, MFCC_FRAMES, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    Conv2D(64, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(128, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# === Entrenar ===
print("🧠 Entrenando CNN...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# === Evaluar ===
loss, acc = model.evaluate(X_test, y_test)
print(f"🎯 Precisión CNN en test: {acc:.3f}")

# === Guardar ===
cnn_path = os.path.join(MODELS_DIR, "cnn_words.keras")
labels_path = os.path.join(MODELS_DIR, "label_encoder.npy")

model.save(cnn_path)
np.save(labels_path, classes)

print("🎉 Modelo CNN guardado en:", cnn_path)
print("🎉 Etiquetas guardadas en:", labels_path)
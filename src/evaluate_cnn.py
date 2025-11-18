import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from config import FEATURES_PATH, MODELS_DIR, MFCC_FEATURES, MFCC_FRAMES
import seaborn as sns
import os

# === Cargar features ===
data = np.load(FEATURES_PATH)
X = data["X"]
y = data["y"]
classes = data["classes"]

# Cargar modelo
model_path = os.path.join(MODELS_DIR, "cnn_words.keras")
model = load_model(model_path)

# Añadir canal
X = X[..., np.newaxis]

# Predicciones
y_pred = np.argmax(model.predict(X), axis=1)

# === MÉTRICAS ===

# Matriz de confusión
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Matriz de Confusión — CNN")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Classification report
report = classification_report(y, y_pred, target_names=classes)
print("\n=== Reporte de Clasificación ===\n")
print(report)

errors = np.sum(y_pred != y)
wer = errors / len(y)

print(f"\n📉 Word Error Rate (WER): {wer:.3f}")

# Guardar reporte en archivo TXT
with open(os.path.join(MODELS_DIR, "cnn_classification_report.txt"), "w") as f:
    f.write(report)

print("\n📁 Reporte guardado en models/cnn_classification_report.txt")

import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from config import (
    DATASET_DIR, FEATURES_PATH,
    TARGET_LABELS, SAMPLE_RATE, SAMPLES_PER_FILE,
    MFCC_FEATURES, MFCC_FRAMES
)

# ----------------------------------------------------
#   PROCESAR UN SOLO AUDIO → Cargar → Pad → MFCC
# ----------------------------------------------------
def process_audio(path):
    # 1. Cargar audio (mono, 16 kHz)
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # 2. Ajustar duración: pad o truncado a 1 segundo
    if len(signal) < SAMPLES_PER_FILE:
        signal = np.pad(signal, (0, SAMPLES_PER_FILE - len(signal)))
    else:
        signal = signal[:SAMPLES_PER_FILE]

    # MFCC base
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=MFCC_FEATURES
    )

    # Delta (primer derivada)
    delta = librosa.feature.delta(mfcc)

    # Delta-Delta (segunda derivada)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Concatenar MFCC + delta + delta-delta
    mfcc = np.concatenate([mfcc, delta, delta2], axis=0)

    # Normalizar
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # 4. Ajustar número de frames a MFCC_FRAMES (32)
    if mfcc.shape[1] < MFCC_FRAMES:
        pad = MFCC_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :MFCC_FRAMES]

    return mfcc


# ----------------------------------------------------
#   PROCESAR TODO EL DATASET
# ----------------------------------------------------
def build_feature_set(max_files_per_label=3000):
    X = []
    y = []

    for label in TARGET_LABELS:
        folder = os.path.join(DATASET_DIR, label)
        wav_files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
        
        print(f"Procesando {label}... ({len(wav_files)} archivos)")

        wav_files = wav_files[:max_files_per_label]

        for file in wav_files:
            path = os.path.join(folder, file)
            try:
                mfcc = process_audio(path)
                X.append(mfcc)
                y.append(label)
            except Exception as e:
                print(f"[❌] Error procesando {path}: {e}")

    # Convertir a arrays de numpy
    X = np.array(X)
    y = np.array(y)

    # Codificar etiquetas
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Guardar dataset preprocesado
    np.savez_compressed(FEATURES_PATH, X=X, y=y_encoded, classes=encoder.classes_)
    print("💾 Features guardados en:", FEATURES_PATH)


# ----------------------------------------------------
#   EJECUCIÓN DIRECTA
# ----------------------------------------------------
if __name__ == "__main__":
    build_feature_set()
    print("✅ Preprocesamiento completado.")
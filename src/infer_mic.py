import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from pynput import keyboard
import threading
import os
from config import (
    SAMPLE_RATE, SAMPLE_RATE, MFCC_FEATURES, MFCC_FRAMES,
    SAMPLES_PER_FILE, MODELS_DIR
)

# === CARGAR MODELO Y ETIQUETAS ===
model_path = os.path.join(MODELS_DIR, "cnn_words.keras")
model = tf.keras.models.load_model(model_path)

labels_path = os.path.join(MODELS_DIR, "label_encoder.npy")
classes = np.load(labels_path, allow_pickle=True)

le = LabelEncoder()
le.classes_ = classes

recording = []
is_recording = False

# === CALLBACK: ACUMULA AUDIO ===
def audio_callback(indata, frames, time, status):
    if is_recording:
        recording.append(indata.copy())

# === INICIAR GRABACIÓN ===
def start_recording():
    global is_recording, recording
    print("🎙️ Grabando... Presiona 'e' para detener.")
    recording = []
    is_recording = True

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype='float32',
        callback=audio_callback
    ):
        while is_recording:
            sd.sleep(100)

# === DETENER GRABACIÓN ===
def stop_recording():
    global is_recording
    is_recording = False
    print("🛑 Grabación detenida.")
    process_audio()

# === PROCESAR AUDIO — IGUAL AL PREPROCESSING ===
def process_audio():
    if len(recording) == 0:
        print("❌ No se grabó audio.")
        return

    # === Unir audio grabado ===
    audio = np.concatenate(recording).flatten().astype(np.float32)
    
    # === 1. Recorte automático de silencio ===
    audio, _ = librosa.effects.trim(audio, top_db=30)
    
    # Si recorta demasiado, regresar al audio original
    if len(audio) < int(0.1 * SAMPLE_RATE):
        audio = audio  
    
    # === 2. Normalización de volumen ===
    audio = librosa.util.normalize(audio)

    # Pad o truncado
    if len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)))
    else:
        audio = audio[:SAMPLES_PER_FILE]

    # === MFCC + deltas + delta-deltas ===
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc = np.concatenate([mfcc, delta, delta2], axis=0)

    # === Normalización ===
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # Ajustar frames
    if mfcc.shape[1] < MFCC_FRAMES:
        pad = MFCC_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MFCC_FRAMES]

    # Dar forma a la CNN (1, 39, 32, 1)
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis]

    # === PREDICCIÓN ===
    pred = model.predict(mfcc_input, verbose=0)
    idx = np.argmax(pred)
    predicted_word = le.inverse_transform([idx])[0]
    confidence = float(np.max(pred))

    print(f"✅ Predicción: {predicted_word}  |  Confianza: {confidence:.2f}")

# === CONTROLES DE TECLADO ===
def on_press(key):
    try:
        if key.char == 's' and not is_recording:
            threading.Thread(target=start_recording).start()
        elif key.char == 'e' and is_recording:
            stop_recording()
        elif key.char == 'q':
            print("👋 Saliendo.")
            return False
    except:
        pass

# === MAIN ===
if __name__ == "__main__":
    print("⌨️ Presiona 's' para grabar, 'e' para detener, 'q' para salir.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

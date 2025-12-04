import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from pynput import keyboard
import threading
import os

from config import (
    SAMPLE_RATE as TARGET_SR,
    MFCC_FEATURES, MFCC_FRAMES,
    SAMPLES_PER_FILE, MODELS_DIR
)

# =========================================
#           CARGAR MODELO Y ETIQUETAS
# =========================================
model_path = os.path.join(MODELS_DIR, "cnn_words.keras")
model = tf.keras.models.load_model(model_path)

labels_path = os.path.join(MODELS_DIR, "label_encoder.npy")
classes = np.load(labels_path, allow_pickle=True)

le = LabelEncoder()
le.classes_ = classes

recording = []
is_recording = False


# =========================================
#   DETECTAR SAMPLE RATE NATIVO DEL MIC
# =========================================
def get_native_sr():
    try:
        device_info = sd.query_devices(sd.default.device[0], 'input')
        native_sr = int(device_info["default_samplerate"])
        print(f"🎧 Sample rate nativo del micrófono: {native_sr} Hz")
        return native_sr
    except Exception as e:
        print(f"⚠️ No se pudo leer el sample rate nativo ({e}). Usando 44100 Hz.")
        return 44100

NATIVE_SR = get_native_sr()


# =========================================
#        CALLBACK DEL STREAM DE AUDIO
# =========================================
def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️ Audio status:", status)
    if is_recording:
        recording.append(indata.copy())


# =========================================
#            INICIAR GRABACIÓN
# =========================================
def start_recording():
    global is_recording, recording
    print("🎙️ Grabando... Presiona 'e' para detener.")
    recording = []
    is_recording = True

    try:
        with sd.InputStream(
            channels=1,
            samplerate=NATIVE_SR,
            dtype="float32",
            callback=audio_callback
        ):
            while is_recording:
                sd.sleep(100)
    except Exception as e:
        print(f"❌ Error al abrir el micrófono: {e}")
        is_recording = False


# =========================================
#            DETENER GRABACIÓN
# =========================================
def stop_recording():
    global is_recording
    is_recording = False
    print("🛑 Grabación detenida.")
    process_audio()


# =========================================
#           PROCESAR AUDIO
# =========================================
def process_audio():
    if len(recording) == 0:
        print("❌ No se grabó audio.")
        return

    # Unir audio
    audio = np.concatenate(recording).flatten().astype(np.float32)

    # Recorte silencios
    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) < 0.1 * NATIVE_SR:
        print("⚠️ Audio demasiado corto. ¿Micrófono sin permiso o muy silencioso?")
        return

    # Normalización
    audio = librosa.util.normalize(audio)

    # RESAMPLE A 16,000 Hz
    if NATIVE_SR != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=NATIVE_SR, target_sr=TARGET_SR)

    # Pad / corte
    if len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)))
    else:
        audio = audio[:SAMPLES_PER_FILE]

    # MFCC + Δ + ΔΔ
    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=MFCC_FEATURES)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc = np.concatenate([mfcc, delta, delta2], axis=0)

    # Normalización
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # Ajustar frames
    if mfcc.shape[1] < MFCC_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MFCC_FRAMES - mfcc.shape[1])), 'constant')
    else:
        mfcc = mfcc[:, :MFCC_FRAMES]

    # Forma CNN
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis]

    # Predicción
    pred = model.predict(mfcc_input, verbose=0)
    idx = np.argmax(pred)
    predicted_word = le.inverse_transform([idx])[0]
    confidence = float(np.max(pred))

    print(f"✅ Predicción: {predicted_word} | Confianza: {confidence:.2f}")


# =========================================
#             TECLADO
# =========================================
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


# =========================================
#                MAIN
# =========================================
if __name__ == "__main__":
    print("⌨️ Presiona 's' para grabar, 'e' para detener, 'q' para salir.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
        
# Reconocimiento de Voz en Tiempo Real (CNN + MFCC)

Un sistema de reconocimiento de comandos de voz en tiempo real que utiliza el conjunto de datos **Comandos de Voz**, **MFCC + funciones delta + delta-delta** y una **CNN** optimizada para audios cortos.

Reconoce 15 palabras y funciona en cualquier equipo sin tener que volver a descargar el conjunto de datos.

---

## Características
- Extracción de características MFCC, delta, delta-delta
- Recorte de silencios + normalización de amplitud
- CNN con una precisión de aproximadamente el 90-93 %
- Inferencia de micrófono en tiempo real
- Totalmente portátil (solo se requiere el modelo entrenado)
  
---

## Instalación

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Entrenamiento (opcional)

```bash
python src/preprocess_features.py
python src/train_cnn.py
```

Estos scripts generarán:

- `features/mfcc_words.npz`
- `models/cnn_words.keras`
- `models/label_encoder.npy`

---

## Evaluación

Para evaluar el modelo, corre:

```bash
python src/evaluate_cnn.py
```

Este script mostrará:

- Precisión
- Matriz de confusión
- Informe de clasificación (precisión, recuperación, puntuación F1)
- Índice de error de palabras (WER)

---

## Reconocimiento en tiempo real

Para ejecutar la inferencia del micrófono en tiempo real:

```bash
python src/infer_mic.py
```

Controles del teclado:

- s → iniciar grabación
- e → detener grabación y predecir
- q → salir

Ejemplo de salida:

```bash
Recording...
Stopped.
Prediction: yes | Confidence: 0.94
```

# Reconocimiento de Voz en Tiempo Real (CNN + MFCC)

Un sistema de reconocimiento de comandos de voz en tiempo real que utiliza el conjunto de datos **Speech Commands**, **MFCC + funciones delta + delta-delta** y una **CNN** optimizada para audios cortos.

Reconoce 15 palabras y funciona en cualquier equipo sin tener que volver a descargar el conjunto de datos.

---

## Características
- Extracción de características MFCC, delta, delta-delta
- Recorte de silencios + normalización de amplitud
- CNN con una precisión de aproximadamente el 90-93 %
- Inferencia de micrófono en tiempo real
- Totalmente portátil (solo se requiere el modelo entrenado)
  
---

## Palabras reconocidas

El modelo reconoce los siguientes 15 comandos del conjunto Speech Commands:

```
bed  down  follow  go  happy  house  left  no  off  
on  right  stop  up  visual  yes
```

---

## Requisitos

- Python >= 3.11
  (El proyecto usa TensorFlow 2.20.0, que es compatible con estas versiones.)
- Micrófono funcional
- Windows, macOS o Linux

---

## Clonar el repositorio

Primero, clona el proyecto y entra en la carpeta:

```bash
git clone https://github.com/albertomonterom/Speech-Commands-CNN.git
cd Speech-Commands-CNN
```

(O descarga el ZIP desde GitHub.)

---

## Estructura del proyecto

```
.
├── models/
│   ├── cnn_classification_report.txt
│   ├── cnn_words.keras
│   └── label_encoder.npy
│
├── src/
│   ├── config.py
│   ├── download_dataset.py
│   ├── evaluate_cnn.py
│   ├── infer_mic.py
│   ├── preprocess_features.py
│   └── train_cnn.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación

Antes de instalar dependencias, actualiza pip:

```bash
python -m pip install --upgrade pip
```

Luego crea el entorno virtual:

```bash
python -m venv venv
```

Actívalo:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

Finalmente instala los requisitos:

```bash
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

# Speech Commands — Real-Time Voice Recognition (CNN + MFCC)

A real-time speech command recognition system using the **Speech Commands** dataset, **MFCC + delta + delta-delta features**, and a **CNN** optimized for short audio clips.

Recognizes 15 words and runs on any machine without re-downloading the dataset.

---

## Features
- MFCC, delta, delta-delta feature extraction  
- Silence trimming + amplitude normalization  
- CNN with ~90–93% accuracy  
- Real-time microphone inference  
- Fully portable (only the trained model is required)

---

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Training (optional)

```bash
python src/preprocess_features.py
python src/train_cnn.py
```

These scripts will generate:

- `features/mfcc_words.npz`
- `models/cnn_words.keras`
- `models/label_encoder.npy`

---

## Evaluation

To evaluate the trained model, run:

```bash
python src/evaluate_cnn.py
```

This script will display:

- accuracy
- confusion matrix
- classification report (precision, recall, F1-score)
- Word Error Rate (WER)

---

## Real-Time Recognition

To run real-time microphone inference:

```bash
python src/infer_mic.py
```

Keyboard controls:

- s → start recording
- e → stop recording and predict
- q → quit

Example output:

```bash
Recording...
Stopped.
Prediction: yes | Confidence: 0.94
```
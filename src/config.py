import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "speech_commands")
FEATURES_PATH = os.path.join(BASE_DIR, "features", "mfcc_words.npz")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "features"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset URL
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Words we selected (15 words)
TARGET_LABELS = [
    "bed",
    "down",
    "follow",
    "go",
    "happy",
    "house",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "visual",
    "yes"
]

# Audio config
SAMPLE_RATE = 16000
DURATION = 1.0
SAMPLES_PER_FILE = int(SAMPLE_RATE * DURATION)

# MFCC config
MFCC_FEATURES = 13
MFCC_FRAMES = 32

# Train/Test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

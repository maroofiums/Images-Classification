import torch 
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGS_DIR = os.path.join(PROJECT_ROOT,"logs")
DATA_DIR = os.path.join(PROJECT_ROOT,"data")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT,"checkpoints")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT,"outputs")

os.makedirs(LOGS_DIR,parents=True, exist_ok=True)
os.makedirs(DATA_DIR,parents=True, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR,parents=True, exist_ok=True)
os.makedirs(OUTPUTS_DIR,parents=True, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10

IMAGE_SIZE = 32

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

PIN_MEMORY = True

MODEL_NAME = "SimpleCNN"

CHECKPOINT_NAME = "best_model.pth"
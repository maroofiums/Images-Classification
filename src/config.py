from pathlib import Path
import os
import torch

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for directory in (LOGS_DIR, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset

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

# Training Hyperparameters

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
STEP_SIZE = 10

NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

# Model

MODEL_NAME = "SimpleCNN"
CHECKPOINT_NAME = "best_model.pth"

# Reproducibility

SEED = 42
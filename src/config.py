from pathlib import Path

import torch

# Project Paths

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
IMAGES_DIR: Path = PROJECT_ROOT / "images"

# Create required directories
for directory in (
    LOGS_DIR,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    IMAGES_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Device

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset

NUM_CLASSES: int = 10
IMAGE_SIZE: int = 32

CLASS_NAMES: tuple[str, ...] = (
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
BATCH_SIZE: int = 64
EPOCHS: int = 20

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4

STEP_SIZE: int = 10

NUM_WORKERS: int = 2
PIN_MEMORY: bool = torch.cuda.is_available()

# Model
MODEL_NAME: str = "SimpleCNN"
CHECKPOINT_NAME: str = "best_model.pth"

# Reproducibility

SEED: int = 42
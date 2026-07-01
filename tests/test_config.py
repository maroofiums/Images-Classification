from pathlib import Path

from src.config import (
    DEVICE,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
    NUM_CLASSES,
    IMAGE_SIZE,
    CLASS_NAMES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    PIN_MEMORY,
    MODEL_NAME,
    CHECKPOINT_NAME,
    SEED,
)


def test_device():
    """Device should be either CPU or CUDA."""
    assert DEVICE in ("cpu", "cuda")


def test_project_directories():
    """Required project directories should exist."""

    assert isinstance(DATA_DIR, Path)
    assert isinstance(CHECKPOINT_DIR, Path)
    assert isinstance(OUTPUT_DIR, Path)
    assert isinstance(LOGS_DIR, Path)

    assert CHECKPOINT_DIR.exists()
    assert OUTPUT_DIR.exists()
    assert LOGS_DIR.exists()


def test_dataset_configuration():
    """Dataset configuration should match CIFAR-10."""

    assert NUM_CLASSES == 10
    assert IMAGE_SIZE == 32

    assert len(CLASS_NAMES) == NUM_CLASSES


def test_training_configuration():
    """Training hyperparameters should be valid."""

    assert BATCH_SIZE > 0
    assert EPOCHS > 0
    assert LEARNING_RATE > 0
    assert WEIGHT_DECAY >= 0
    assert NUM_WORKERS >= 0


def test_pin_memory():
    """PIN_MEMORY should be boolean."""

    assert isinstance(PIN_MEMORY, bool)


def test_model_configuration():
    """Model configuration should be valid."""

    assert isinstance(MODEL_NAME, str)
    assert MODEL_NAME != ""

    assert CHECKPOINT_NAME.endswith(".pth")


def test_seed():
    """Random seed should be an integer."""

    assert isinstance(SEED, int)
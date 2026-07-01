import itertools
import random
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.optim import Optimizer

from src.config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SEED,
)


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_acc: float,
    filename: str,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: Trained model.
        optimizer: Optimizer.
        epoch: Current epoch.
        best_acc: Best validation accuracy.
        filename: Checkpoint filename.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_acc,
    }

    path = CHECKPOINT_DIR / filename

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | None,
    filename: str,
    device: str,
) -> tuple[nn.Module, Optimizer | None, int, float]:
    """
    Load a saved checkpoint.

    Args:
        model: Model instance.
        optimizer: Optimizer instance or None.
        filename: Checkpoint filename.
        device: Device to load checkpoint onto.

    Returns:
        Model, optimizer, epoch, best accuracy.
    """

    path = CHECKPOINT_DIR / filename

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_accuracy"]

    print(f"Checkpoint loaded from {path}")

    return model, optimizer, epoch, best_acc


def count_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def plot_history(
    history: dict[str, list[float]],
) -> None:
    """
    Plot training history.

    Args:
        history: Training history dictionary.
    """

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    # Loss Curve

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "loss_curve.png",
        dpi=300,
    )

    plt.close()

    # Accuracy Curve

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train",
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        OUTPUT_DIR / "acc_curve.png",
        dpi=300,
    )

    plt.close()


history: dict[str, list[float]] = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}


def ensure_dir(
    path: Path | str,
) -> None:
    """
    Create a directory if it does not exist.

    Args:
        path: Directory path.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    normalize: bool = False,
    filename: str = "confusion_matrix.png",
) -> None:
    """
    Plot and save the confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Class names.
        normalize: Whether to normalize the confusion matrix.
        filename: Output filename.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    if normalize:
        cm = cm.astype(float)
        cm /= cm.sum(
            axis=1,
            keepdims=True,
        )
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(10, 8))

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        class_names,
    )

    threshold = cm.max() / 2

    fmt = ".2f" if normalize else "d"

    for i, j in itertools.product(
        range(cm.shape[0]),
        range(cm.shape[1]),
    ):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            ha="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
        )

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.tight_layout()

    save_path = OUTPUT_DIR / filename

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

    print(
        f"Confusion matrix saved to {save_path}"
    )
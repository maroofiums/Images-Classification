import os
import random
from pathlib import Path
import itertools

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import torch

from src.config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SEED,
)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark= False

def save_checkpoint(model, optimizer, epoch, best_acc, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_acc,
    }

    path = CHECKPOINT_DIR / filename

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, filename, device):

    path = CHECKPOINT_DIR / filename

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_accuracy"]

    print(f"Checkpoint loaded from {path}")

    return model, optimizer, epoch, best_acc

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.plot(epochs,history["train_loss"],label="Train")
    plt.plot(epochs,history["val_loss"],label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "loss_curve.png"
        )
    )
    plt.close()
    plt.plot(epochs,history["train_acc"],label="Train")
    plt.plot(epochs,history["val_acc"],label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "acc_curve.png"
        )
    )
    plt.close()

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}



def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    normalize=False,
    filename="confusion_matrix.png",
):
    """
    Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List/Tuple of class names
        normalize: If True, normalize each row
        filename: Output image filename
    """

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

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
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
        )

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.tight_layout()

    save_path = OUTPUT_DIR / filename

    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Confusion matrix saved to {save_path}")
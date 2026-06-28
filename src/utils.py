import os
import random
from pathlib import Path

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
        "model_stat_dict":model.state_dict(),
        "optimizer_stat_dict":optimizer.state_dict(),
        "best_accuracy":best_acc,
    }

    path = os.path.join(
        CHECKPOINT_DIR,
        filename
    )

    torch.save(checkpoint,path)

    print(f"checkpoint saved to {path}")

def load_checkpoint(model, optimizer, filename, device):
    path = os.path.join(
        CHECKPOINT_DIR,
        filename
    )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_stat_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_stat_dict"])

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





def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
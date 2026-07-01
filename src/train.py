from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_NAME,
    STEP_SIZE,
)
from src.dataset import get_dataloaders
from src.engine import (
    train_one_epoch,
    validate_one_epoch,
)
from src.logger import app_logger
from src.metrics import print_metrics
from src.model import build_model
from src.utils import (
    count_parameters,
    plot_history,
    save_checkpoint,
    set_seed,
)


def train() -> None:
    """
    Train the image classification model.

    This function:
        - Sets the random seed.
        - Loads the datasets.
        - Builds the model.
        - Trains for multiple epochs.
        - Evaluates on the validation set.
        - Saves the best checkpoint.
        - Plots the training history.
    """

    app_logger.info("Training started.")

    
    # Reproducibility
    set_seed()

    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = build_model().to(DEVICE)

    app_logger.info(
        f"Trainable Parameters: {count_parameters(model):,}"
    )

    print(model)
    print(
        f"\nTrainable Parameters: "
        f"{count_parameters(model):,}"
    )

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=0.1,
    )

    # History
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_accuracy: float = 0.0

    # Training Loop
    for epoch in range(EPOCHS):

        app_logger.info(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        val_loss, val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(
            train_metrics["accuracy"]
        )
        history["val_acc"].append(
            val_metrics["accuracy"]
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        print(
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_metrics['accuracy']:.4f}"
        )

        print_metrics(train_metrics, "Train")

        print()

        print(
            f"Val Loss   : {val_loss:.4f} | "
            f"Val Acc   : {val_metrics['accuracy']:.4f}"
        )

        print_metrics(val_metrics, "Validation")

        app_logger.info(
            f"Train Loss={train_loss:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_accuracy:

            best_accuracy = val_metrics["accuracy"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_acc=best_accuracy,
                filename=CHECKPOINT_NAME,
            )

            app_logger.info(
                f"Best model saved "
                f"(Accuracy={best_accuracy:.4f})"
            )

            print("\nBest Model Updated!")

    # Training Finished
    app_logger.info("Training completed successfully.")

    print("\nTraining Finished.")

    plot_history(history)


if __name__ == "__main__":
    train()
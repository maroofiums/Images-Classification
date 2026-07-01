import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_NAME,
    STEP_SIZE
)
from src.dataset import get_dataloaders
from src.engine import train_one_epoch, validate_one_epoch
from src.model import build_model
from src.metrics import print_metrics
from src.utils import (
    set_seed,
    save_checkpoint,
    plot_history,
    count_parameters,
)
from src.logger import app_logger


def train():
    app_logger.info("Training started")

    set_seed()

    train_loader,val_loader, _ = get_dataloaders()
    
    model = build_model().to(DEVICE)
    print(model)
    print(f"Trainable Parameteres: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=0.1
    )


    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        
        app_logger.info(f"Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )
        val_loss, val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f} | "
            f"Val Acc   : {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]

            save_checkpoint(
                model,
                optimizer,
                epoch+1,
                best_accuracy,
                filename=CHECKPOINT_NAME
            )

            print("Best Model Updated!")

    print("Training Finish")
    plot_history(history)

if __name__ == "__main__":
    train()
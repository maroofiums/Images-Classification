from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import calculate_metrics


def train_one_epoch(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    device: str,
) -> tuple[float, dict[str, Any]]:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Training device ("cpu" or "cuda").

    Returns:
        Average training loss and evaluation metrics.
    """

    model.train()

    running_loss = 0.0

    all_predictions: list[int] = []
    all_labels: list[int] = []

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    avg_loss = running_loss / len(dataloader)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return avg_loss, metrics


def validate_one_epoch(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    device: str,
) -> tuple[float, dict[str, Any]]:
    """
    Validate the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Validation device.

    Returns:
        Average validation loss and evaluation metrics.
    """

    model.eval()

    running_loss = 0.0

    all_predictions: list[int] = []
    all_labels: list[int] = []

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
            )

    avg_loss = running_loss / len(dataloader)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return avg_loss, metrics
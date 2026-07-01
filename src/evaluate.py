from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    CHECKPOINT_NAME,
    CLASS_NAMES,
)
from src.dataset import get_dataloaders
from src.logger import app_logger
from src.metrics import (
    calculate_metrics,
    get_classification_report,
    get_confusion_matrix,
    print_metrics,
)
from src.model import build_model
from src.utils import (
    load_checkpoint,
    plot_confusion_matrix,
)


def evaluate() -> None:
    """
    Evaluate the trained model on the CIFAR-10 test dataset.

    This function:
        - Loads the best model checkpoint.
        - Evaluates the model on the test dataset.
        - Computes evaluation metrics.
        - Prints a classification report.
        - Generates confusion matrix visualizations.
    """

    app_logger.info("Starting model evaluation...")

    # Data
    _, _, test_loader = get_dataloaders()

    # Model
    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    model, _, epoch, best_acc = load_checkpoint(
        model=model,
        optimizer=None,
        filename=CHECKPOINT_NAME,
        device=DEVICE,
    )

    app_logger.info(
        f"Loaded checkpoint '{CHECKPOINT_NAME}' "
        f"(Epoch {epoch}, Best Accuracy: {best_acc:.4f})"
    )

    model.eval()

    running_loss = 0.0

    all_predictions: list[int] = []
    all_labels: list[int] = []

    progress_bar = tqdm(
        test_loader,
        desc="Evaluating",
        leave=False,
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    # Metrics
    average_loss = running_loss / len(test_loader)

    metrics: dict[str, Any] = calculate_metrics(
        all_labels,
        all_predictions,
    )

    app_logger.info(
        f"Test Loss: {average_loss:.4f}"
    )

    app_logger.info(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    # Console Output
    print("\n========== Test Results ==========\n")

    print(f"Average Loss : {average_loss:.4f}\n")

    print_metrics(metrics)

    print("\n========== Classification Report ==========\n")

    report = get_classification_report(
        all_labels,
        all_predictions,
    )

    print(report)

    print("\n========== Confusion Matrix ==========\n")

    confusion_matrix = get_confusion_matrix(
        all_labels,
        all_predictions,
    )

    print(confusion_matrix)

    # Save Confusion Matrix
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_predictions,
        class_names=CLASS_NAMES,
    )

    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_predictions,
        class_names=CLASS_NAMES,
        normalize=True,
        filename="normalized_confusion_matrix.png",
    )

    app_logger.info(
        "Confusion matrices saved successfully."
    )

    print("\n========== Class Names ==========\n")

    print(CLASS_NAMES)

    app_logger.info(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":
    evaluate()
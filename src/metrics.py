from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, precision,
        recall, and F1-score.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def get_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> np.ndarray:
    """
    Compute the confusion matrix.
    """

    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> str:
    """
    Generate the classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )


def print_metrics(
    metrics: dict[str, float],
    prefix: str = "",
) -> None:
    """
    Print evaluation metrics.

    Args:
        metrics: Dictionary returned by calculate_metrics().
        prefix: Optional prefix for each line.
    """

    if prefix:
        prefix = f"{prefix} "

    print(f"{prefix}Accuracy : {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall   : {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score : {metrics['f1_score']:.4f}")


if __name__ == "__main__":

    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2]

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print_metrics(metrics)

    print()

    print(get_classification_report(
        y_true,
        y_pred,
    ))

    print(get_confusion_matrix(
        y_true,
        y_pred,
    ))
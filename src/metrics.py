import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
    }

    return metrics

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def get_classification_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )

def print_metrics(metrics,text):

    print(f"{text} Accuracy : {metrics['accuracy']:.4f}")
    print(f"{text} Precision: {metrics['precision']:.4f}")
    print(f"{text} Recall   : {metrics['recall']:.4f}")
    print(f"{text} F1 Score : {metrics['f1_score']:.4f}")

if __name__ == "__main__":

    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2]

    metrics = calculate_metrics(y_true, y_pred)

    print_metrics(metrics)

    print(get_classification_report(y_true, y_pred))
    print(get_confusion_matrix(y_true, y_pred))
import torch
import torch.nn as nn

from src.config import (
    DEVICE,
    CHECKPOINT_NAME,
    CLASS_NAMES,
)
from src.dataset import get_dataloaders
from src.metrics import (
    calculate_metrics,
    get_classification_report,
    get_confusion_matrix,
    print_metrics,
)
from src.model import build_model
from src.utils import load_checkpoint,plot_confusion_matrix
from src.logger import app_logger


def evaluate():
    app_logger.info("Evaluation started")

    _,_,test_loader = get_dataloaders()

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    model, _, epoch, best_acc = load_checkpoint(
        model=model,
        optimizer=None,
        filename=CHECKPOINT_NAME, 
        device=DEVICE
    )

    print(f"\nLoaded checkpoint from epoch {epoch}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs,labels)

            running_loss += loss.item()
            predictions = torch.argmax(outputs,dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = running_loss / len(test_loader)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    print("\n<========== Test Results ==========>\n")

    print(f"Avg Loss: {avg_loss}")

    print_metrics(metrics,"->")

    print("<----- Classification Report ----->")

    print(
        get_classification_report(
            all_labels,
            all_predictions
        )
    )
        
    print("<----- Confusion Metrics ----->")

    print(
        get_confusion_matrix(
            all_labels,
            all_predictions
        )
    )

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
    
    print("\nClass Names:")
    print(CLASS_NAMES)


if __name__ == "__main__":
    evaluate()
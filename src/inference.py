from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import (
    DEVICE,
    CHECKPOINT_NAME,
    CLASS_NAMES,
    IMAGES_DIR,
)
from src.logger import inference_logger
from src.model import build_model
from src.utils import load_checkpoint

# Dataset Statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# Image Transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# Types
class PredictionResult(TypedDict):
    class_name: str
    confidence: float


# Model Loading
def load_model() -> nn.Module:
    """
    Load the trained model from the checkpoint.

    Returns:
        Loaded PyTorch model.
    """

    inference_logger.info("Loading trained model...")

    model = build_model().to(DEVICE)

    model, _, epoch, best_acc = load_checkpoint(
        model=model,
        optimizer=None,
        filename=CHECKPOINT_NAME,
        device=DEVICE,
    )

    model.eval()

    inference_logger.info(
        f"Checkpoint loaded successfully "
        f"(Epoch={epoch}, Best Accuracy={best_acc:.4f})"
    )

    return model


# Inference
def predict_image(
    image_path: Path | str,
    model: nn.Module,
) -> PredictionResult:
    """
    Predict the class of an image.

    Args:
        image_path: Path to the image.
        model: Loaded classification model.

    Returns:
        Predicted class name and confidence score.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    inference_logger.info(
        f"Running inference on: {image_path.name}"
    )

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    confidence = confidence.item()
    prediction = prediction.item()

    predicted_class = CLASS_NAMES[prediction]

    inference_logger.info(
        f"Prediction: {predicted_class} "
        f"({confidence:.4f})"
    )

    return {
        "class_name": predicted_class,
        "confidence": confidence,
    }


# Demo
if __name__ == "__main__":

    model = load_model()

    image_path = IMAGES_DIR / "dog.jpg"

    result = predict_image(
        image_path=image_path,
        model=model,
    )

    print(result)
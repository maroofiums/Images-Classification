from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import (
    DEVICE,
    CHECKPOINT_NAME,
    CLASS_NAMES,
)
from src.model import build_model
from src.utils import load_checkpoint
from src.logger import inference_logger


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

def load_model():

    model = build_model().to(DEVICE)

    model, _, _, _ = load_checkpoint(
        model=model,
        optimizer=None,
        filename=CHECKPOINT_NAME,
        device=DEVICE,
    )

    model.eval()

    return model


def predict_image(image_path, model):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

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
        f"Prediction: {predicted_class} ({confidence:.4f})"
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
    }


if __name__ == "__main__":

    model = load_model()

    image_path = Path("sample.jpg")

    result = predict_image(image_path, model)

    print(result)
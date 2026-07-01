from pathlib import Path

import torch
from PIL import Image

from src.inference import (
    load_model,
    predict_image,
)

from src.config import IMAGES_DIR


class DummyModel(torch.nn.Module):
    """
    A dummy model that always predicts class 0.
    """

    def forward(self, x):
        batch_size = x.shape[0]

        output = torch.zeros(batch_size, 10)

        output[:, 0] = 10.0

        return output


def test_predict_image(tmp_path):

    image_path = IMAGES_DIR / "airplane.jpg"

    image = Image.new(
        "RGB",
        (32, 32),
        color="white",
    )

    image.save(image_path)

    model = DummyModel()

    prediction = predict_image(
        image_path=image_path,
        model=model,
    )

    assert isinstance(prediction, dict)

    assert "class" in prediction
    assert "confidence" in prediction

    assert prediction["class"] == "airplane"

    assert 0.0 <= prediction["confidence"] <= 1.0



def test_load_model():

    model = load_model()

    assert model is not None
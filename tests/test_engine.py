import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from src.engine import train_one_epoch
from src.model import build_model


def test_train_one_epoch():

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))

    dataset = TensorDataset(images, labels)

    loader = DataLoader(dataset, batch_size=8)

    model = build_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    loss, metrics = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        "cpu",
    )

    assert isinstance(loss, float)
    assert loss > 0

    assert "accuracy" in metrics
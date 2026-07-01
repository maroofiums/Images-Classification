import torch

from src.model import build_model


def test_model_output_shape():

    model = build_model()

    x = torch.randn(8, 3, 32, 32)

    output = model(x)

    assert output.shape == (8, 10)


def test_model_forward_pass():

    model = build_model()

    x = torch.randn(1, 3, 32, 32)

    output = model(x)

    assert torch.is_tensor(output)


def test_model_parameters():

    model = build_model()

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    assert params > 0
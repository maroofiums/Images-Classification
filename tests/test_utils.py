import torch

from src.model import build_model
from src.utils import count_parameters


def test_count_parameters():

    model = build_model()

    params = count_parameters(model)

    assert isinstance(params, int)
    assert params > 0
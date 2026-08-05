import torch

from treestack_cnn.models import build_models
from treestack_cnn.utils import count_parameters


def test_all_models_produce_class_logits() -> None:
    models = build_models(in_channels=3, num_classes=10)
    inputs = torch.randn(4, 3, 32, 32)
    parameter_counts = []
    for model in models.values():
        outputs = model(inputs)
        assert outputs.shape == (4, 10)
        parameter_counts.append(count_parameters(model))
    assert all(count > 0 for count in parameter_counts)
    assert len(set(parameter_counts)) == 3


def test_models_accept_single_channel_images() -> None:
    for model in build_models(in_channels=1, num_classes=10).values():
        assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from treestack_cnn.config import TrainingConfig
from treestack_cnn.training import load_checkpoint, predict_probabilities, train_model


def test_training_checkpoint_and_probability_prediction(tmp_path: Path) -> None:
    torch.manual_seed(4)
    inputs = torch.randn(24, 1, 8, 8)
    labels = (inputs.mean(dim=(1, 2, 3)) > 0).long()
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    evaluation_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 2))
    checkpoint = tmp_path / "model.pt"
    config = TrainingConfig(
        epochs=2,
        batch_size=8,
        patience=2,
        device="cpu",
        scheduler="cosine",
        mixed_precision=True,
    )

    result = train_model(
        model,
        train_loader,
        evaluation_loader,
        config,
        torch.device("cpu"),
        checkpoint,
    )
    assert checkpoint.exists()
    assert 1 <= result.best_epoch <= 2
    assert len(result.history) == 2

    reloaded = nn.Sequential(nn.Flatten(), nn.Linear(64, 2))
    load_checkpoint(reloaded, checkpoint, torch.device("cpu"))
    predictions = predict_probabilities(reloaded, evaluation_loader, torch.device("cpu"))
    assert predictions.probabilities.shape == (24, 2)
    assert torch.allclose(
        torch.from_numpy(predictions.probabilities.sum(axis=1)), torch.ones(24), atol=1e-6
    )
    assert predictions.milliseconds_per_sample >= 0

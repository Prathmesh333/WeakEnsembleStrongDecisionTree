from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from treestack_cnn.config import TrainingConfig
from treestack_cnn.elite_training import (
    EliteWeightConfig,
    blend_states,
    train_elite_model,
)


def test_blend_states_averages_float_tensors_and_copies_counters() -> None:
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor(9, dtype=torch.int64),
    }
    blended = blend_states(first, second, first_weight=0.25)
    assert torch.allclose(blended["weight"], torch.tensor([4.0, 6.0]))
    assert blended["counter"].item() == 9


def test_elite_training_keeps_a_valid_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(7)
    inputs = torch.randn(24, 1, 4, 4)
    labels = (inputs.flatten(1).sum(dim=1) > 0).long()
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 2))
    checkpoint = tmp_path / "elite.pt"
    result = train_elite_model(
        model,
        loader,
        loader,
        TrainingConfig(
            epochs=3,
            batch_size=8,
            learning_rate=1e-2,
            patience=3,
            scheduler="cosine",
        ),
        EliteWeightConfig(ema_decay=0.8, averaging_start_epoch=2),
        torch.device("cpu"),
        checkpoint,
    )

    assert checkpoint.exists()
    assert len(result.history) == 3
    assert result.elite_kind in {"raw", "ema", "greedy_soup"}
    assert result.soup_checkpoint_count >= 1
    assert 0.0 <= result.best_validation_accuracy <= 1.0
    assert model(inputs[:2]).shape == (2, 2)

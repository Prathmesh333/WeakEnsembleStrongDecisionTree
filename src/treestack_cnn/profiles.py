from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import TrainingConfig


MODEL_DIVERSITY_PROFILES: dict[str, dict[str, Any]] = {
    "cnn1_spatial": {
        "optimizer": "adamw",
        "learning_rate_multiplier": 0.5,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "diversity_source": "spatial 4x4 pooling; ReLU; dense classifier; moderate dropout",
    },
    "cnn2_depthwise": {
        "optimizer": "adamw",
        "learning_rate_multiplier": 0.75,
        "weight_decay": 2e-4,
        "label_smoothing": 0.02,
        "diversity_source": "depthwise separable topology; SiLU; parameter-efficient channels",
    },
    "cnn3_residual": {
        "optimizer": "adamw",
        "learning_rate_multiplier": 0.6,
        "weight_decay": 3e-4,
        "label_smoothing": 0.04,
        "diversity_source": "deeper residual topology; GELU head; light dropout; wider channels",
    },
}


def model_training_config(base: TrainingConfig, model_name: str) -> TrainingConfig:
    """Return the model-specific settings used to encourage useful diversity."""
    profile = MODEL_DIVERSITY_PROFILES[model_name]
    return replace(
        base,
        optimizer=str(profile["optimizer"]),
        learning_rate=base.learning_rate * float(profile["learning_rate_multiplier"]),
        weight_decay=float(profile["weight_decay"]),
        label_smoothing=float(profile["label_smoothing"]),
    )

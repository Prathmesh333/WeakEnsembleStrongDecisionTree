from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    name: str = "fashion_mnist"
    root: str = "data"
    base_fraction: float = 0.60
    meta_fraction: float = 0.20
    test_fraction: float = 0.20
    num_workers: int = 0
    download: bool = True

    def validate(self) -> None:
        if self.name not in {"fashion_mnist", "cifar10"}:
            raise ValueError(f"Unsupported dataset: {self.name}")
        total = self.base_fraction + self.meta_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("base_fraction + meta_fraction + test_fraction must equal 1")
        if min(self.base_fraction, self.meta_fraction, self.test_fraction) <= 0:
            raise ValueError("All split fractions must be positive")


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    label_smoothing: float = 0.0
    device: str = "auto"


@dataclass(slots=True)
class TreeConfig:
    depths: list[int | None] = field(default_factory=lambda: [3, 5, 7, None])
    min_samples_leaf: list[int] = field(default_factory=lambda: [5, 10, 20])
    criteria: list[str] = field(default_factory=lambda: ["gini", "entropy"])
    cv_folds: int = 3


@dataclass(slots=True)
class ExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: ["fashion_mnist", "cifar10"])
    seeds: list[int] = field(default_factory=lambda: [17, 42, 73])
    output_dir: str = "artifacts"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    reuse_checkpoints: bool = True
    reuse_predictions: bool = True

    def validate(self) -> None:
        if not self.datasets:
            raise ValueError("At least one dataset is required")
        if not self.seeds:
            raise ValueError("At least one random seed is required")
        for name in self.datasets:
            DatasetConfig(name=name).validate()
        self.dataset.validate()
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.tree.cv_folds < 2:
            raise ValueError("tree.cv_folds must be at least 2")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    known = set(instance.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys for {type(instance).__name__}: {unknown}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        config.validate()
        return config

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The configuration root must be a mapping")

    nested = {
        "dataset": DatasetConfig,
        "training": TrainingConfig,
        "tree": TreeConfig,
    }
    top_level = {key: value for key, value in raw.items() if key not in nested}
    _merge_dataclass(config, top_level)
    for key, cls in nested.items():
        if key in raw:
            setattr(config, key, _merge_dataclass(cls(), raw[key]))
    config.validate()
    return config

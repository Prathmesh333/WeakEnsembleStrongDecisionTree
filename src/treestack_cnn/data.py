from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from .config import DatasetConfig, TrainingConfig


@dataclass(frozen=True, slots=True)
class SplitIndices:
    base: np.ndarray
    base_train: np.ndarray
    base_validation: np.ndarray
    meta: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "base": self.base.tolist(),
            "base_train": self.base_train.tolist(),
            "base_validation": self.base_validation.tolist(),
            "meta": self.meta.tolist(),
            "test": self.test.tolist(),
        }


@dataclass(slots=True)
class DatasetBundle:
    name: str
    class_names: list[str]
    num_classes: int
    in_channels: int
    splits: SplitIndices
    base_train: Dataset[Any]
    base_validation: Dataset[Any]
    meta: Dataset[Any]
    test: Dataset[Any]


@dataclass(slots=True)
class LoaderBundle:
    base_train: DataLoader[Any]
    base_validation: DataLoader[Any]
    meta: DataLoader[Any]
    test: DataLoader[Any]


class TransformSubset(Dataset[tuple[torch.Tensor, int]]):
    """Apply a split-specific transform to selected samples of a raw dataset."""

    def __init__(self, dataset: Dataset[Any], indices: np.ndarray, transform: Any) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        image, target = self.dataset[int(self.indices[position])]
        return self.transform(image), int(target)


def stratified_three_way_split(
    targets: np.ndarray,
    base_fraction: float,
    meta_fraction: float,
    test_fraction: float,
    seed: int,
    base_validation_fraction: float = 0.10,
) -> SplitIndices:
    """Produce disjoint class-stratified base/meta/test indices.

    The validation subset is drawn only from the base partition, so the meta
    partition is never used to fit or early-stop a CNN.
    """
    targets = np.asarray(targets)
    all_indices = np.arange(len(targets))
    base, remaining = train_test_split(
        all_indices,
        train_size=base_fraction,
        random_state=seed,
        shuffle=True,
        stratify=targets,
    )
    relative_meta = meta_fraction / (meta_fraction + test_fraction)
    meta, test = train_test_split(
        remaining,
        train_size=relative_meta,
        random_state=seed + 1,
        shuffle=True,
        stratify=targets[remaining],
    )
    base_train, base_validation = train_test_split(
        base,
        test_size=base_validation_fraction,
        random_state=seed + 2,
        shuffle=True,
        stratify=targets[base],
    )
    result = SplitIndices(
        base=np.sort(base),
        base_train=np.sort(base_train),
        base_validation=np.sort(base_validation),
        meta=np.sort(meta),
        test=np.sort(test),
    )
    _validate_disjoint_split(result, len(targets))
    return result


def _validate_disjoint_split(splits: SplitIndices, sample_count: int) -> None:
    base, meta, test = map(set, (splits.base, splits.meta, splits.test))
    if base & meta or base & test or meta & test:
        raise RuntimeError("The base, meta, and test partitions overlap")
    if len(base | meta | test) != sample_count:
        raise RuntimeError("The three partitions do not cover the complete dataset")
    if set(splits.base_train) & set(splits.base_validation):
        raise RuntimeError("The internal base train and validation partitions overlap")
    if set(splits.base_train) | set(splits.base_validation) != base:
        raise RuntimeError("The internal base split does not cover the base partition")


def _dataset_spec(name: str) -> tuple[type[Dataset[Any]], int, tuple[float, ...], tuple[float, ...]]:
    if name == "fashion_mnist":
        return datasets.FashionMNIST, 1, (0.2860,), (0.3530,)
    if name == "cifar10":
        return datasets.CIFAR10, 3, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    raise ValueError(f"Unsupported dataset: {name}")


def _transforms(name: str, mean: tuple[float, ...], std: tuple[float, ...]) -> tuple[Any, Any]:
    if name == "fashion_mnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return train_transform, evaluation_transform


def build_dataset(config: DatasetConfig, seed: int) -> DatasetBundle:
    config.validate()
    dataset_class, in_channels, mean, std = _dataset_spec(config.name)
    root = Path(config.root)
    training_raw = dataset_class(root=root, train=True, transform=None, download=config.download)
    test_raw = dataset_class(root=root, train=False, transform=None, download=config.download)
    complete = ConcatDataset([training_raw, test_raw])
    targets = np.concatenate(
        [np.asarray(training_raw.targets), np.asarray(test_raw.targets)]
    ).astype(np.int64)
    splits = stratified_three_way_split(
        targets,
        config.base_fraction,
        config.meta_fraction,
        config.test_fraction,
        seed,
    )
    train_transform, evaluation_transform = _transforms(config.name, mean, std)
    class_names = list(training_raw.classes)
    return DatasetBundle(
        name=config.name,
        class_names=class_names,
        num_classes=len(class_names),
        in_channels=in_channels,
        splits=splits,
        base_train=TransformSubset(complete, splits.base_train, train_transform),
        base_validation=TransformSubset(complete, splits.base_validation, evaluation_transform),
        meta=TransformSubset(complete, splits.meta, evaluation_transform),
        test=TransformSubset(complete, splits.test, evaluation_transform),
    )


def build_loaders(
    bundle: DatasetBundle,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    seed: int,
) -> LoaderBundle:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": training_config.batch_size,
        "num_workers": dataset_config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": dataset_config.num_workers > 0,
    }
    return LoaderBundle(
        base_train=DataLoader(bundle.base_train, shuffle=True, generator=generator, **common),
        base_validation=DataLoader(bundle.base_validation, shuffle=False, **common),
        meta=DataLoader(bundle.meta, shuffle=False, **common),
        test=DataLoader(bundle.test, shuffle=False, **common),
    )

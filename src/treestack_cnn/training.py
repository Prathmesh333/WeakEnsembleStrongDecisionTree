from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .utils import ensure_dir


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_accuracy: float
    training_seconds: float


@dataclass(slots=True)
class PredictionResult:
    probabilities: np.ndarray
    labels: np.ndarray
    total_seconds: float
    milliseconds_per_sample: float


def _epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    use_mixed_precision: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_mixed_precision,
            ):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_size = targets.size(0)
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    config: TrainingConfig,
    device: torch.device,
    checkpoint_path: str | Path,
) -> TrainingResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    if config.optimizer == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.scheduler == "cosine":
        scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )
    use_mixed_precision = config.mixed_precision and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    else:  # PyTorch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch_number in range(1, config.epochs + 1):
        train_loss, train_accuracy = _epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
        )
        validation_loss, validation_accuracy = _epoch(
            model,
            validation_loader,
            criterion,
            device,
            use_mixed_precision=use_mixed_precision,
        )
        if config.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(validation_accuracy)
        history.append(
            {
                "epoch": float(epoch_number),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_accuracy > best_accuracy + 1e-8:
            best_accuracy = validation_accuracy
            best_epoch = epoch_number
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Training completed without producing model weights")
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint_path)
    ensure_dir(checkpoint.parent)
    torch.save(best_state, checkpoint)
    return TrainingResult(history, best_epoch, best_accuracy, elapsed)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)


def predict_probabilities(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    use_mixed_precision: bool = False,
) -> PredictionResult:
    model.eval()
    probability_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, labels in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_mixed_precision and device.type == "cuda",
            ):
                logits = model(inputs.to(device, non_blocking=True))
            # Compute softmax in float32 even when CUDA inference uses float16 autocast.
            # Casting after a float16 softmax preserves its rounding error and can make
            # otherwise valid probability rows fail strict normalization checks.
            probability_batches.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            label_batches.append(labels.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    probabilities = np.concatenate(probability_batches).astype(np.float32, copy=False)
    labels = np.concatenate(label_batches).astype(np.int64, copy=False)
    return PredictionResult(
        probabilities=probabilities,
        labels=labels,
        total_seconds=elapsed,
        milliseconds_per_sample=elapsed * 1000.0 / len(labels),
    )

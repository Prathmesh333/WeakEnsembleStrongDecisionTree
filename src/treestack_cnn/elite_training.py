from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .training import _epoch
from .utils import ensure_dir


@dataclass(slots=True)
class EliteWeightConfig:
    ema_decay: float = 0.98
    averaging_start_epoch: int = 10
    minimum_improvement: float = 0.0

    def validate(self, total_epochs: int) -> None:
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be between zero and one")
        if not 1 <= self.averaging_start_epoch <= total_epochs:
            raise ValueError("averaging_start_epoch must be within the training run")
        if self.minimum_improvement < 0.0:
            raise ValueError("minimum_improvement must be non-negative")


@dataclass(slots=True)
class EliteTrainingResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation_accuracy: float
    training_seconds: float
    elite_kind: str
    soup_checkpoint_count: int


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def blend_states(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    first_weight: float,
) -> dict[str, torch.Tensor]:
    if first.keys() != second.keys():
        raise ValueError("State dictionaries do not have identical parameter keys")
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first_weight must be between zero and one")
    blended: dict[str, torch.Tensor] = {}
    for key in first:
        left = first[key]
        right = second[key]
        if left.shape != right.shape:
            raise ValueError(f"Incompatible tensor shape for {key}")
        if torch.is_floating_point(left):
            blended[key] = left.mul(first_weight).add(right, alpha=1.0 - first_weight)
        else:
            blended[key] = right.clone()
    return blended


def _validation_metrics_for_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    validation_loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    use_mixed_precision: bool,
) -> tuple[float, float]:
    model.load_state_dict(state)
    return _epoch(
        model,
        validation_loader,
        criterion,
        device,
        use_mixed_precision=use_mixed_precision,
    )


def train_elite_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    training_config: TrainingConfig,
    elite_config: EliteWeightConfig,
    device: torch.device,
    checkpoint_path: str | Path,
) -> EliteTrainingResult:
    """Train one model while retaining raw, EMA, and greedy-soup elites."""
    elite_config.validate(training_config.epochs)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    if training_config.optimizer == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            momentum=training_config.momentum,
            weight_decay=training_config.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    if training_config.scheduler == "cosine":
        scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )
    use_mixed_precision = training_config.mixed_precision and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    else:  # PyTorch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    ema_state: dict[str, torch.Tensor] | None = None
    soup_state: dict[str, torch.Tensor] | None = None
    soup_accuracy = -1.0
    soup_count = 0
    elite_state: dict[str, torch.Tensor] | None = None
    elite_accuracy = -1.0
    elite_kind = "raw"
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch_number in range(1, training_config.epochs + 1):
        train_loss, train_accuracy = _epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
        )
        validation_loss, raw_accuracy = _epoch(
            model,
            validation_loader,
            criterion,
            device,
            use_mixed_precision=use_mixed_precision,
        )
        raw_state = _cpu_state(model)
        if ema_state is None:
            ema_state = copy.deepcopy(raw_state)
        else:
            ema_state = blend_states(ema_state, raw_state, elite_config.ema_decay)

        ema_accuracy = float("nan")
        soup_candidate_accuracy = float("nan")
        if epoch_number >= elite_config.averaging_start_epoch:
            _, ema_accuracy = _validation_metrics_for_state(
                model,
                ema_state,
                validation_loader,
                criterion,
                device,
                use_mixed_precision,
            )
            if soup_state is None:
                soup_candidate = copy.deepcopy(raw_state)
                soup_candidate_accuracy = raw_accuracy
            else:
                soup_candidate = blend_states(
                    soup_state, raw_state, soup_count / (soup_count + 1.0)
                )
                _, soup_candidate_accuracy = _validation_metrics_for_state(
                    model,
                    soup_candidate,
                    validation_loader,
                    criterion,
                    device,
                    use_mixed_precision,
                )
            if soup_candidate_accuracy >= soup_accuracy + elite_config.minimum_improvement:
                soup_state = soup_candidate
                soup_accuracy = soup_candidate_accuracy
                soup_count += 1

        epoch_candidates: list[tuple[str, float, dict[str, torch.Tensor]]] = [
            ("raw", raw_accuracy, raw_state)
        ]
        if epoch_number >= elite_config.averaging_start_epoch:
            epoch_candidates.append(("ema", ema_accuracy, ema_state))
            if soup_state is not None:
                epoch_candidates.append(("greedy_soup", soup_accuracy, soup_state))
        candidate_kind, candidate_accuracy, candidate_state = max(
            epoch_candidates, key=lambda item: item[1]
        )
        improved = candidate_accuracy > elite_accuracy + 1e-8
        if improved:
            elite_accuracy = candidate_accuracy
            elite_state = copy.deepcopy(candidate_state)
            elite_kind = candidate_kind
            best_epoch = epoch_number
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        model.load_state_dict(raw_state)
        if training_config.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(raw_accuracy)
        history.append(
            {
                "epoch": float(epoch_number),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": raw_accuracy,
                "ema_validation_accuracy": ema_accuracy,
                "soup_candidate_validation_accuracy": soup_candidate_accuracy,
                "soup_best_validation_accuracy": soup_accuracy,
                "elite_validation_accuracy": elite_accuracy,
                "elite_kind": elite_kind,
                "soup_checkpoint_count": soup_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epochs_without_improvement >= training_config.patience:
            break

    elapsed = time.perf_counter() - started
    if elite_state is None:
        raise RuntimeError("Training completed without producing elite model weights")
    model.load_state_dict(elite_state)
    model.to(device)
    checkpoint = Path(checkpoint_path)
    ensure_dir(checkpoint.parent)
    torch.save(elite_state, checkpoint)
    return EliteTrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_accuracy=elite_accuracy,
        training_seconds=elapsed,
        elite_kind=elite_kind,
        soup_checkpoint_count=soup_count,
    )

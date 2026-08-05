from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .utils import ensure_dir


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        labels=np.arange(len(class_names)),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1_score": report[class_name]["f1-score"],
            "support": int(report[class_name]["support"]),
        }
        for class_name in class_names
    }
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(len(class_names))
        ).tolist(),
    }


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    title: str,
    path: str | Path,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    size = max(7.0, 0.75 * len(class_names))
    figure, axis = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(title=title, xlabel="Predicted class", ylabel="True class")
    figure.tight_layout()
    target = Path(path)
    ensure_dir(target.parent)
    figure.savefig(target, dpi=180)
    plt.close(figure)


def save_training_curves(history: list[dict[str, float]], path: str | Path) -> None:
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="Train")
    axes[1].plot(frame["epoch"], frame["validation_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    target = Path(path)
    ensure_dir(target.parent)
    figure.savefig(target, dpi=180)
    plt.close(figure)


def save_tree_visualization(
    estimator: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(24, 12))
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=4,
        fontsize=6,
        ax=axis,
    )
    axis.set_title("Decision-tree fusion rules (first four levels)")
    figure.tight_layout()
    target = Path(path)
    ensure_dir(target.parent)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def probability_feature_names(model_count: int, class_names: list[str]) -> list[str]:
    return [
        f"cnn_{model + 1}_p_{class_name}"
        for model in range(model_count)
        for class_name in class_names
    ]

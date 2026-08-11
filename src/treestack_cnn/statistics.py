from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .utils import ensure_dir


def exact_mcnemar(
    labels: np.ndarray,
    candidate_predictions: np.ndarray,
    reference_predictions: np.ndarray,
) -> dict[str, float | int]:
    """Return the exact paired test and correction-versus-harm counts."""
    labels = np.asarray(labels, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    reference = np.asarray(reference_predictions, dtype=np.int64)
    if labels.ndim != 1 or candidate.shape != labels.shape or reference.shape != labels.shape:
        raise ValueError("Labels and paired predictions must be aligned one-dimensional arrays")

    candidate_correct = candidate == labels
    reference_correct = reference == labels
    corrections = int(np.sum(candidate_correct & ~reference_correct))
    harms = int(np.sum(~candidate_correct & reference_correct))
    discordant = corrections + harms
    p_value = (
        float(binomtest(corrections, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    return {
        "samples": int(len(labels)),
        "candidate_accuracy": float(np.mean(candidate_correct)),
        "reference_accuracy": float(np.mean(reference_correct)),
        "accuracy_delta": float(np.mean(candidate_correct) - np.mean(reference_correct)),
        "corrections": corrections,
        "harms": harms,
        "net_corrections": corrections - harms,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": p_value,
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 2026,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("At least one scalar value is required")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def publication_summary(
    rows: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Write seed-level means, sample standard deviations, and bootstrap CIs."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No result rows were provided")
    main = frame.loc[frame["category"] == "main"].copy()
    summary_rows: list[dict[str, Any]] = []
    for (dataset, method, method_key), group in main.groupby(
        ["dataset", "method", "method_key"], sort=True
    ):
        accuracies = group["accuracy"].to_numpy(dtype=np.float64)
        f1_values = group["macro_f1"].to_numpy(dtype=np.float64)
        accuracy_low, accuracy_high = _bootstrap_mean_interval(accuracies)
        f1_low, f1_high = _bootstrap_mean_interval(f1_values, seed=2027)
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_key": method_key,
                "seeds": int(group["seed"].nunique()),
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
                "accuracy_ci95_low": accuracy_low,
                "accuracy_ci95_high": accuracy_high,
                "macro_f1_mean": float(f1_values.mean()),
                "macro_f1_std": float(f1_values.std(ddof=1)) if len(f1_values) > 1 else 0.0,
                "macro_f1_ci95_low": f1_low,
                "macro_f1_ci95_high": f1_high,
            }
        )
    path = ensure_dir(output_dir) / "publication_summary.csv"
    pd.DataFrame(summary_rows).to_csv(path, index=False)
    return path


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    if count == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, position in enumerate(order):
        value = min(1.0, float(p_values[position]) * (count - rank))
        running = max(running, value)
        adjusted[position] = running
    return adjusted


def paired_test_table(
    runs: list[dict[str, Any]], output_dir: str | Path
) -> Path:
    """Compare V4 with strong paired baselines for every dataset and seed."""
    comparisons = {
        "soft_vote": "Soft Vote",
        "logistic_stack": "Logistic Stack",
        "rf_soft": "Random Forest Stack",
        "dt_soft": "DT-Soft",
    }
    table_rows: list[dict[str, Any]] = []
    for run in runs:
        prediction_path = Path(run["run_dir"]) / "test_predictions.npz"
        with np.load(prediction_path) as stored:
            values = {key: stored[key] for key in stored.files}
        candidate = values["evolutionary_fusion_v4"]
        labels = values["labels"]
        for reference_key, reference_name in comparisons.items():
            result = exact_mcnemar(labels, candidate, values[reference_key])
            table_rows.append(
                {
                    "dataset": run["dataset"],
                    "seed": int(run["seed"]),
                    "candidate": "Evolutionary Fusion (V4)",
                    "reference": reference_name,
                    "reference_key": reference_key,
                    **result,
                }
            )

        cnn_keys = ["cnn_1", "cnn_2", "cnn_3"]
        strongest_key = max(
            cnn_keys, key=lambda key: float(np.mean(values[key] == labels))
        )
        result = exact_mcnemar(labels, candidate, values[strongest_key])
        table_rows.append(
            {
                "dataset": run["dataset"],
                "seed": int(run["seed"]),
                "candidate": "Evolutionary Fusion (V4)",
                "reference": "Strongest CNN",
                "reference_key": strongest_key,
                **result,
            }
        )

    frame = pd.DataFrame(table_rows)
    frame["mcnemar_holm_p"] = _holm_adjust(
        frame["mcnemar_exact_p"].to_numpy(dtype=np.float64)
    )
    path = ensure_dir(output_dir) / "paired_tests.csv"
    frame.to_csv(path, index=False)
    return path

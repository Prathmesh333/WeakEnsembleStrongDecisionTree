from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .config import TreeConfig


@dataclass(slots=True)
class FittedCombiner:
    estimator: Any
    predictions: np.ndarray
    best_parameters: dict[str, Any]
    fit_seconds: float
    predict_milliseconds_per_sample: float


@dataclass(slots=True)
class EnsembleResult:
    predictions: dict[str, np.ndarray]
    combiners: dict[str, FittedCombiner]
    depth_ablation: dict[str, tuple[np.ndarray, int, int]]
    best_two_indices: tuple[int, int]
    meta_feature_dimensions: dict[str, int]


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Probabilities must have shape (models, samples, classes)")
    if values.shape[0] < 2:
        raise ValueError("At least two base models are required")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain NaN or infinite values")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("Probabilities must be between zero and one")
    row_sums = values.sum(axis=2, keepdims=True, dtype=np.float32)
    if (row_sums <= 0.0).any():
        raise ValueError("Probability vectors must have a positive sum")
    max_deviation = float(np.max(np.abs(row_sums - 1.0)))
    if max_deviation > 5e-3:
        raise ValueError(
            "Each model probability vector must sum to one "
            f"(maximum deviation: {max_deviation:.6f})"
        )
    # Old mixed-precision Kaggle caches can contain float16-rounded softmax values.
    # Renormalizing them preserves argmax predictions and avoids retraining the CNNs.
    return values / row_sums


def soft_features(probabilities: np.ndarray) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    return values.transpose(1, 0, 2).reshape(values.shape[1], -1)


def hard_features(probabilities: np.ndarray) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    predictions = values.argmax(axis=2).T
    identity = np.eye(values.shape[2], dtype=np.float32)
    encoded = identity[predictions]
    return encoded.reshape(values.shape[1], -1)


def enhanced_features(probabilities: np.ndarray) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    base = soft_features(values)
    confidences = values.max(axis=2).T
    entropies = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=2).T
    predictions = values.argmax(axis=2)
    agreements = [
        (predictions[first] == predictions[second]).astype(np.float32)
        for first, second in itertools.combinations(range(values.shape[0]), 2)
    ]
    agreement_matrix = np.stack(agreements, axis=1)
    return np.concatenate([base, confidences, entropies, agreement_matrix], axis=1)


def majority_vote(probabilities: np.ndarray) -> np.ndarray:
    """Vote by class, resolving ties with the corresponding mean probabilities."""
    values = _validate_probabilities(probabilities)
    hard = values.argmax(axis=2)
    average = values.mean(axis=0)
    output = np.empty(values.shape[1], dtype=np.int64)
    for sample in range(values.shape[1]):
        counts = np.bincount(hard[:, sample], minlength=values.shape[2])
        candidates = np.flatnonzero(counts == counts.max())
        output[sample] = candidates[np.argmax(average[sample, candidates])]
    return output


def soft_vote(probabilities: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = _validate_probabilities(probabilities)
    if weights is None:
        average = values.mean(axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (values.shape[0],):
            raise ValueError("weights must contain one value per model")
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("weights must be non-negative and have a positive sum")
        average = np.average(values, axis=0, weights=weights)
    return average.argmax(axis=1)


def _timed_predict(estimator: Any, features: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    predictions = estimator.predict(features)
    elapsed = time.perf_counter() - started
    return np.asarray(predictions, dtype=np.int64), elapsed * 1000.0 / len(features)


def fit_logistic_stack(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> FittedCombiner:
    estimator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    started = time.perf_counter()
    estimator.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - started
    predictions, milliseconds = _timed_predict(estimator, test_features)
    return FittedCombiner(estimator, predictions, {}, fit_seconds, milliseconds)


def fit_nonlinear_stack(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> dict[str, FittedCombiner]:
    """Fit strong nonlinear baselines on the same soft-probability features."""
    estimators: dict[str, Any] = {
        "rf_soft": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "hgb_soft": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=True,
            random_state=seed,
        ),
    }
    results: dict[str, FittedCombiner] = {}
    for name, estimator in estimators.items():
        started = time.perf_counter()
        estimator.fit(train_features, train_labels)
        fit_seconds = time.perf_counter() - started
        predictions, milliseconds = _timed_predict(estimator, test_features)
        results[name] = FittedCombiner(
            estimator,
            predictions,
            dict(estimator.get_params(deep=False)),
            fit_seconds,
            milliseconds,
        )
    return results


def fit_tree_stack(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    config: TreeConfig,
    seed: int,
) -> FittedCombiner:
    """Tune a shallow tree using only cross-validation within the meta partition."""
    finite_depths = [depth for depth in config.depths if depth is not None]
    if not finite_depths:
        finite_depths = [5]
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        param_grid={
            "criterion": config.criteria,
            "max_depth": finite_depths,
            "min_samples_leaf": config.min_samples_leaf,
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - started
    predictions, milliseconds = _timed_predict(search.best_estimator_, test_features)
    return FittedCombiner(
        search.best_estimator_, predictions, dict(search.best_params_), fit_seconds, milliseconds
    )


def fit_tree_at_depth(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    config: TreeConfig,
    seed: int,
    depth: int | None,
) -> FittedCombiner:
    """Tune the criterion and leaf size while holding tree depth fixed."""
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        param_grid={
            "criterion": config.criteria,
            "max_depth": [depth],
            "min_samples_leaf": config.min_samples_leaf,
        },
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - started
    predictions, milliseconds = _timed_predict(search.best_estimator_, test_features)
    return FittedCombiner(
        search.best_estimator_, predictions, dict(search.best_params_), fit_seconds, milliseconds
    )


def run_ensembles(
    meta_probabilities: np.ndarray,
    meta_labels: np.ndarray,
    test_probabilities: np.ndarray,
    tree_config: TreeConfig,
    seed: int,
    validation_accuracies: np.ndarray,
) -> EnsembleResult:
    meta_probabilities = _validate_probabilities(meta_probabilities)
    test_probabilities = _validate_probabilities(test_probabilities)
    if meta_probabilities.shape[0] != test_probabilities.shape[0]:
        raise ValueError("Meta and test predictions have different model counts")
    if meta_probabilities.shape[2] != test_probabilities.shape[2]:
        raise ValueError("Meta and test predictions have different class counts")

    predictions: dict[str, np.ndarray] = {
        f"cnn_{index + 1}": test_probabilities[index].argmax(axis=1)
        for index in range(test_probabilities.shape[0])
    }
    predictions["majority_vote"] = majority_vote(test_probabilities)
    predictions["soft_vote"] = soft_vote(test_probabilities)
    predictions["weighted_soft_vote"] = soft_vote(
        test_probabilities, np.asarray(validation_accuracies)
    )

    feature_builders = {
        "hard": hard_features,
        "soft": soft_features,
        "enhanced": enhanced_features,
    }
    train_features = {
        name: builder(meta_probabilities) for name, builder in feature_builders.items()
    }
    test_features = {
        name: builder(test_probabilities) for name, builder in feature_builders.items()
    }

    combiners: dict[str, FittedCombiner] = {}
    combiners["logistic_stack"] = fit_logistic_stack(
        train_features["soft"], meta_labels, test_features["soft"], seed
    )
    combiners.update(
        fit_nonlinear_stack(train_features["soft"], meta_labels, test_features["soft"], seed)
    )
    for feature_name in ("hard", "soft", "enhanced"):
        method = f"dt_{feature_name}"
        combiners[method] = fit_tree_stack(
            train_features[feature_name],
            meta_labels,
            test_features[feature_name],
            tree_config,
            seed,
        )
    predictions.update({name: result.predictions for name, result in combiners.items()})

    ordering = np.argsort(np.asarray(validation_accuracies))[::-1]
    best_two = (int(ordering[0]), int(ordering[1]))
    best_two_train = soft_features(meta_probabilities[list(best_two)])
    best_two_test = soft_features(test_probabilities[list(best_two)])
    combiners["dt_soft_best_two"] = fit_tree_stack(
        best_two_train, meta_labels, best_two_test, tree_config, seed
    )
    predictions["dt_soft_best_two"] = combiners["dt_soft_best_two"].predictions

    depth_ablation: dict[str, tuple[np.ndarray, int, int]] = {}
    for depth in tree_config.depths:
        fitted = fit_tree_at_depth(
            train_features["soft"],
            meta_labels,
            test_features["soft"],
            tree_config,
            seed,
            depth,
        )
        estimator = fitted.estimator
        label = "unrestricted" if depth is None else str(depth)
        depth_ablation[label] = (
            fitted.predictions,
            int(estimator.get_depth()),
            int(estimator.get_n_leaves()),
        )

    dimensions = {name: features.shape[1] for name, features in train_features.items()}
    return EnsembleResult(predictions, combiners, depth_ablation, best_two, dimensions)


def disagreement_analysis(
    base_predictions: np.ndarray,
    majority_predictions: np.ndarray,
    tree_predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, int]:
    if base_predictions.ndim != 2 or base_predictions.shape[0] != 3:
        raise ValueError("Disagreement analysis currently expects exactly three CNNs")
    first, second, third = base_predictions
    all_agree = (first == second) & (second == third)
    all_disagree = (first != second) & (first != third) & (second != third)
    two_agree = ~(all_agree | all_disagree)
    any_base_correct = (base_predictions == labels[None, :]).any(axis=0)
    majority_correct = majority_predictions == labels
    tree_correct = tree_predictions == labels
    return {
        "samples": int(len(labels)),
        "all_three_agree": int(all_agree.sum()),
        "exactly_two_agree": int(two_agree.sum()),
        "all_three_disagree": int(all_disagree.sum()),
        "tree_corrects_majority": int((tree_correct & ~majority_correct).sum()),
        "tree_harms_correct_majority": int((~tree_correct & majority_correct).sum()),
        "tree_wrong_despite_a_correct_base_model": int((~tree_correct & any_base_correct).sum()),
    }


def pairwise_diversity_analysis(
    base_predictions: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Measure whether base models make different errors on the same samples."""
    if base_predictions.ndim != 2:
        raise ValueError("base_predictions must have shape (models, samples)")
    if base_predictions.shape[1] != len(labels):
        raise ValueError("Prediction and label sample counts differ")
    pairs: dict[str, dict[str, float]] = {}
    for first, second in itertools.combinations(range(base_predictions.shape[0]), 2):
        first_correct = base_predictions[first] == labels
        second_correct = base_predictions[second] == labels
        pairs[f"cnn_{first + 1}_vs_cnn_{second + 1}"] = {
            "prediction_disagreement_rate": float(
                np.mean(base_predictions[first] != base_predictions[second])
            ),
            "double_fault_rate": float(np.mean(~first_correct & ~second_correct)),
            "one_correct_one_wrong_rate": float(np.mean(first_correct != second_correct)),
        }
    return {
        "pairwise": pairs,
        "oracle_accuracy_any_cnn_correct": float(
            np.mean((base_predictions == labels[None, :]).any(axis=0))
        ),
    }

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import ExperimentConfig
from .data import build_dataset, build_loaders
from .evaluation import (
    classification_metrics,
    probability_feature_names,
    save_confusion_matrix,
    save_training_curves,
    save_tree_visualization,
)
from .models import MODEL_BUILDERS, MODEL_VERSION
from .profiles import MODEL_DIVERSITY_PROFILES, model_training_config
from .stacking import disagreement_analysis, pairwise_diversity_analysis, run_ensembles
from .training import load_checkpoint, predict_probabilities, train_model
from .utils import count_parameters, ensure_dir, resolve_device, set_seed, write_json


METHOD_LABELS = {
    "cnn_1": "CNN-1",
    "cnn_2": "CNN-2",
    "cnn_3": "CNN-3",
    "majority_vote": "Majority",
    "soft_vote": "Soft Vote",
    "weighted_soft_vote": "Weighted Soft Vote",
    "logistic_stack": "Logistic Stack",
    "rf_soft": "Random Forest Stack",
    "hgb_soft": "HGB Stack",
    "dt_hard": "DT-Hard",
    "dt_soft": "DT-Soft",
    "dt_enhanced": "DT-Enhanced",
    "dt_soft_best_two": "DT-Soft (Best 2)",
}


def _configuration_hash(config: ExperimentConfig, dataset_name: str, seed: int) -> str:
    relevant = {
        "dataset": {**asdict(config.dataset), "name": dataset_name},
        "training": asdict(config.training),
        "tree": asdict(config.tree),
        "model_version": MODEL_VERSION,
        "seed": seed,
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _save_split_indices(bundle: Any, path: Path) -> None:
    np.savez_compressed(
        path,
        base=bundle.splits.base,
        base_train=bundle.splits.base_train,
        base_validation=bundle.splits.base_validation,
        meta=bundle.splits.meta,
        test=bundle.splits.test,
    )


def _model_prediction_cache(
    model: torch.nn.Module,
    model_name: str,
    loaders: Any,
    device: torch.device,
    cache_path: Path,
    reuse: bool,
) -> dict[str, Any]:
    if cache_path.exists() and reuse:
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    validation = predict_probabilities(model, loaders.base_validation, device)
    meta = predict_probabilities(model, loaders.meta, device)
    test = predict_probabilities(model, loaders.test, device)
    values: dict[str, Any] = {
        "validation_probabilities": validation.probabilities,
        "validation_labels": validation.labels,
        "meta_probabilities": meta.probabilities,
        "meta_labels": meta.labels,
        "test_probabilities": test.probabilities,
        "test_labels": test.labels,
        "validation_accuracy": np.asarray(
            (validation.probabilities.argmax(axis=1) == validation.labels).mean()
        ),
        "meta_ms_per_sample": np.asarray(meta.milliseconds_per_sample),
        "test_ms_per_sample": np.asarray(test.milliseconds_per_sample),
    }
    np.savez_compressed(cache_path, **values)
    return values


def _method_complexity(
    method: str,
    model_parameters: list[int],
    combiners: dict[str, Any],
) -> dict[str, int | None]:
    if method.startswith("cnn_"):
        index = int(method.split("_")[1]) - 1
        return {
            "base_parameters": model_parameters[index],
            "combiner_parameters": 0,
            "tree_depth": None,
            "tree_leaves": None,
        }
    result: dict[str, int | None] = {
        "base_parameters": sum(model_parameters),
        "combiner_parameters": 0,
        "tree_depth": None,
        "tree_leaves": None,
    }
    if method in combiners:
        estimator = combiners[method].estimator
        if isinstance(estimator, DecisionTreeClassifier):
            result.update(
                {
                    "combiner_parameters": int(estimator.tree_.node_count),
                    "tree_depth": int(estimator.get_depth()),
                    "tree_leaves": int(estimator.get_n_leaves()),
                }
            )
        elif hasattr(estimator, "coef_"):
            result["combiner_parameters"] = int(
                estimator.coef_.size + estimator.intercept_.size
            )
        elif isinstance(estimator, RandomForestClassifier):
            result.update(
                {
                    "combiner_parameters": int(
                        sum(tree.tree_.node_count for tree in estimator.estimators_)
                    ),
                    "tree_depth": int(max(tree.get_depth() for tree in estimator.estimators_)),
                    "tree_leaves": int(
                        sum(tree.get_n_leaves() for tree in estimator.estimators_)
                    ),
                }
            )
    return result


def run_single_experiment(
    config: ExperimentConfig, dataset_name: str, seed: int, force: bool = False
) -> tuple[list[dict[str, Any]], Path]:
    dataset_config = replace(config.dataset, name=dataset_name)
    run_hash = _configuration_hash(config, dataset_name, seed)
    run_dir = ensure_dir(Path(config.output_dir) / dataset_name / f"seed_{seed}" / run_hash)
    write_json(
        {**config.as_dict(), "active_dataset": dataset_name, "active_seed": seed},
        run_dir / "config.json",
    )
    print(f"[{dataset_name} seed={seed}] preparing data")
    set_seed(seed)
    bundle = build_dataset(dataset_config, seed)
    _save_split_indices(bundle, run_dir / "split_indices.npz")
    device = resolve_device(config.training.device)

    model_names = list(MODEL_BUILDERS)
    model_parameters: list[int] = []
    validation_accuracies: list[float] = []
    meta_probabilities: list[np.ndarray] = []
    test_probabilities: list[np.ndarray] = []
    inference_times: list[float] = []
    meta_labels: np.ndarray | None = None
    test_labels: np.ndarray | None = None

    for model_index, (model_name, builder) in enumerate(MODEL_BUILDERS.items()):
        model_seed = seed + 1000 * model_index
        set_seed(model_seed)
        training_config = model_training_config(config.training, model_name)
        model = builder(bundle.in_channels, bundle.num_classes)
        model_parameters.append(count_parameters(model))
        loaders = build_loaders(bundle, dataset_config, training_config, model_seed)
        checkpoint_path = run_dir / "checkpoints" / f"{model_name}.pt"
        history_path = run_dir / "training" / f"{model_name}.csv"
        use_checkpoint = checkpoint_path.exists() and config.reuse_checkpoints and not force
        if use_checkpoint:
            print(f"[{dataset_name} seed={seed}] loading {model_name}")
            load_checkpoint(model, checkpoint_path, device)
        else:
            print(f"[{dataset_name} seed={seed}] training {model_name} on {device}")
            result = train_model(
                model,
                loaders.base_train,
                loaders.base_validation,
                training_config,
                device,
                checkpoint_path,
            )
            ensure_dir(history_path.parent)
            pd.DataFrame(result.history).to_csv(history_path, index=False)
            save_training_curves(
                result.history, run_dir / "figures" / f"training_{model_name}.png"
            )
            write_json(
                {
                    "best_epoch": result.best_epoch,
                    "best_validation_accuracy": result.best_validation_accuracy,
                    "training_seconds": result.training_seconds,
                },
                run_dir / "training" / f"{model_name}_summary.json",
            )

        cache_path = run_dir / "predictions" / f"{model_name}.npz"
        ensure_dir(cache_path.parent)
        prediction_values = _model_prediction_cache(
            model,
            model_name,
            loaders,
            device,
            cache_path,
            config.reuse_predictions and not force,
        )
        validation_accuracies.append(float(prediction_values["validation_accuracy"]))
        meta_probabilities.append(prediction_values["meta_probabilities"])
        test_probabilities.append(prediction_values["test_probabilities"])
        inference_times.append(float(prediction_values["test_ms_per_sample"]))
        current_meta_labels = prediction_values["meta_labels"]
        current_test_labels = prediction_values["test_labels"]
        if meta_labels is not None and not np.array_equal(meta_labels, current_meta_labels):
            raise RuntimeError("CNN meta predictions are not aligned to the same samples")
        if test_labels is not None and not np.array_equal(test_labels, current_test_labels):
            raise RuntimeError("CNN test predictions are not aligned to the same samples")
        meta_labels = current_meta_labels
        test_labels = current_test_labels
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert meta_labels is not None and test_labels is not None
    stacked_meta = np.stack(meta_probabilities)
    stacked_test = np.stack(test_probabilities)
    print(f"[{dataset_name} seed={seed}] fitting fusion models")
    ensembles = run_ensembles(
        stacked_meta,
        meta_labels,
        stacked_test,
        config.tree,
        seed,
        np.asarray(validation_accuracies),
    )

    model_dir = ensure_dir(run_dir / "combiners")
    for method, combiner in ensembles.combiners.items():
        joblib.dump(combiner.estimator, model_dir / f"{method}.joblib")

    detailed_metrics: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    base_ensemble_time = float(sum(inference_times))
    for method, predictions in ensembles.predictions.items():
        metrics = classification_metrics(test_labels, predictions, bundle.class_names)
        complexity = _method_complexity(method, model_parameters, ensembles.combiners)
        if method.startswith("cnn_"):
            model_index = int(method.split("_")[1]) - 1
            inference_ms = inference_times[model_index]
        elif method == "dt_soft_best_two":
            selected = list(ensembles.best_two_indices)
            complexity["base_parameters"] = sum(model_parameters[index] for index in selected)
            inference_ms = sum(inference_times[index] for index in selected)
            inference_ms += ensembles.combiners[method].predict_milliseconds_per_sample
        else:
            overhead = (
                ensembles.combiners[method].predict_milliseconds_per_sample
                if method in ensembles.combiners
                else 0.0
            )
            inference_ms = base_ensemble_time + overhead
        label = METHOD_LABELS.get(method, method)
        detailed_metrics[method] = {
            **metrics,
            **complexity,
            "display_name": label,
            "inference_ms_per_sample": inference_ms,
        }
        summary_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "run_hash": run_hash,
                "method": label,
                "method_key": method,
                "category": "main",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "inference_ms_per_sample": inference_ms,
                **complexity,
            }
        )
        save_confusion_matrix(
            test_labels,
            predictions,
            bundle.class_names,
            f"{dataset_name}: {label}",
            run_dir / "figures" / f"confusion_{method}.png",
        )

    for depth_label, (predictions, actual_depth, leaves) in ensembles.depth_ablation.items():
        metrics = classification_metrics(test_labels, predictions, bundle.class_names)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "run_hash": run_hash,
                "method": f"DT-Soft depth={depth_label}",
                "method_key": f"dt_soft_depth_{depth_label}",
                "category": "depth_ablation",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "inference_ms_per_sample": base_ensemble_time,
                "base_parameters": sum(model_parameters),
                "combiner_parameters": 2 * leaves - 1,
                "tree_depth": actual_depth,
                "tree_leaves": leaves,
            }
        )

    soft_tree = ensembles.combiners["dt_soft"].estimator
    feature_names = probability_feature_names(len(model_names), bundle.class_names)
    save_tree_visualization(
        soft_tree,
        feature_names,
        bundle.class_names,
        run_dir / "figures" / "decision_tree_soft.png",
    )
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": soft_tree.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        run_dir / "decision_tree_feature_importance.csv", index=False
    )

    base_test_predictions = stacked_test.argmax(axis=2)
    analysis = disagreement_analysis(
        base_test_predictions,
        ensembles.predictions["majority_vote"],
        ensembles.predictions["dt_soft"],
        test_labels,
    )
    diversity = pairwise_diversity_analysis(base_test_predictions, test_labels)
    write_json(
        {
            "model_names": model_names,
            "model_parameters": dict(zip(model_names, model_parameters)),
            "validation_accuracies": dict(zip(model_names, validation_accuracies)),
            "training_profiles": {
                name: {
                    **MODEL_DIVERSITY_PROFILES[name],
                    "learning_rate": model_training_config(config.training, name).learning_rate,
                }
                for name in model_names
            },
            "best_two_model_indices": ensembles.best_two_indices,
            "meta_feature_dimensions": ensembles.meta_feature_dimensions,
            "split_sizes": {
                "complete": int(len(bundle.splits.base) + len(bundle.splits.meta) + len(bundle.splits.test)),
                "base": int(len(bundle.splits.base)),
                "base_train": int(len(bundle.splits.base_train)),
                "base_validation": int(len(bundle.splits.base_validation)),
                "meta": int(len(bundle.splits.meta)),
                "test": int(len(bundle.splits.test)),
            },
            "combiner_best_parameters": {
                name: result.best_parameters for name, result in ensembles.combiners.items()
            },
            "disagreement_analysis": analysis,
            "pairwise_diversity_analysis": diversity,
            "metrics": detailed_metrics,
        },
        run_dir / "report.json",
    )
    pd.DataFrame(summary_rows).to_csv(run_dir / "results.csv", index=False)
    return summary_rows, run_dir


def aggregate_results(rows: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    output = ensure_dir(output_dir)
    frame = pd.DataFrame(rows)
    raw_path = output / "all_runs.csv"
    frame.to_csv(raw_path, index=False)
    grouped = (
        frame.groupby(["dataset", "method", "method_key", "category"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            inference_ms_mean=("inference_ms_per_sample", "mean"),
            inference_ms_std=("inference_ms_per_sample", "std"),
        )
        .fillna(0.0)
    )
    aggregate_path = output / "aggregate_results.csv"
    grouped.to_csv(aggregate_path, index=False)

    main = grouped[grouped["category"] == "main"].copy()
    main["accuracy_mean_std"] = main.apply(
        lambda row: f"{100 * row['accuracy_mean']:.2f} ± {100 * row['accuracy_std']:.2f}", axis=1
    )
    method_order = [
        "CNN-1",
        "CNN-2",
        "CNN-3",
        "Majority",
        "Soft Vote",
        "Weighted Soft Vote",
        "Logistic Stack",
        "DT-Hard",
        "DT-Soft",
        "DT-Enhanced",
        "DT-Soft (Best 2)",
    ]
    paper_table = main.pivot(index="dataset", columns="method", values="accuracy_mean_std")
    paper_table = paper_table.reindex(columns=[name for name in method_order if name in paper_table])
    paper_path = output / "paper_accuracy_table.csv"
    paper_table.to_csv(paper_path)
    return aggregate_path, paper_path

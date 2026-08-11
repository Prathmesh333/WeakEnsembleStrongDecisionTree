from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .config import DatasetConfig, ExperimentConfig, TrainingConfig, TreeConfig
from .data import build_dataset, build_loaders
from .elite_training import EliteWeightConfig, train_elite_model
from .evaluation import (
    classification_metrics,
    probability_feature_names,
    save_confusion_matrix,
    save_training_curves,
    save_tree_visualization,
)
from .experiment import (
    METHOD_LABELS,
    _configuration_hash,
    _method_complexity,
    aggregate_results,
)
from .models import MODEL_BUILDERS
from .profiles import MODEL_DIVERSITY_PROFILES, model_training_config
from .stacking import (
    disagreement_analysis,
    pairwise_diversity_analysis,
    run_ensembles,
)
from .training import load_checkpoint, predict_probabilities, train_model
from .utils import count_parameters, ensure_dir, set_seed, write_json


def _prediction_cache(
    model: torch.nn.Module,
    loaders: Any,
    device: torch.device,
    cache_path: Path,
    reuse: bool,
    mixed_precision: bool,
) -> dict[str, Any]:
    if cache_path.exists() and reuse:
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    validation = predict_probabilities(model, loaders.base_validation, device, mixed_precision)
    meta = predict_probabilities(model, loaders.meta, device, mixed_precision)
    test = predict_probabilities(model, loaders.test, device, mixed_precision)
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
        "test_ms_per_sample": np.asarray(test.milliseconds_per_sample),
    }
    ensure_dir(cache_path.parent)
    np.savez_compressed(cache_path, **values)
    return values


def _train_model_task(
    model_name: str,
    model_position: int,
    seed: int,
    dataset_values: dict[str, Any],
    training_values: dict[str, Any],
    run_dir_value: str,
    device_queue: Any,
    force: bool,
    elite_weight_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one independent CNN in a spawned process pinned to one CUDA device."""
    device_index = int(device_queue.get())
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        model_seed = seed + 1000 * model_position
        set_seed(model_seed)
        dataset_config = DatasetConfig(**dataset_values)
        training_config = model_training_config(TrainingConfig(**training_values), model_name)
        training_config.device = str(device)
        bundle = build_dataset(dataset_config, seed)
        loaders = build_loaders(bundle, dataset_config, training_config, model_seed)
        model = MODEL_BUILDERS[model_name](bundle.in_channels, bundle.num_classes)
        parameter_count = count_parameters(model)
        run_dir = Path(run_dir_value)
        checkpoint_path = run_dir / "checkpoints" / f"{model_name}.pt"
        cache_path = run_dir / "predictions" / f"{model_name}.npz"
        history_path = run_dir / "training" / f"{model_name}.csv"

        reuse_checkpoint = checkpoint_path.exists() and not force
        if reuse_checkpoint:
            print(f"[{model_name}] loading checkpoint on GPU {device_index}", flush=True)
            load_checkpoint(model, checkpoint_path, device)
        else:
            print(f"[{model_name}] training on GPU {device_index}", flush=True)
            if elite_weight_values is None:
                result = train_model(
                    model,
                    loaders.base_train,
                    loaders.base_validation,
                    training_config,
                    device,
                    checkpoint_path,
                )
            else:
                result = train_elite_model(
                    model,
                    loaders.base_train,
                    loaders.base_validation,
                    training_config,
                    EliteWeightConfig(**elite_weight_values),
                    device,
                    checkpoint_path,
                )
            ensure_dir(history_path.parent)
            pd.DataFrame(result.history).to_csv(history_path, index=False)
            save_training_curves(
                result.history, run_dir / "figures" / f"training_{model_name}.png"
            )
            training_summary = {
                "gpu": device_index,
                "best_epoch": result.best_epoch,
                "best_validation_accuracy": result.best_validation_accuracy,
                "training_seconds": result.training_seconds,
                "training_strategy": (
                    "raw_ema_greedy_soup"
                    if elite_weight_values is not None
                    else "standard_best_checkpoint"
                ),
            }
            if elite_weight_values is not None:
                training_summary.update(
                    {
                        "elite_kind": result.elite_kind,
                        "soup_checkpoint_count": result.soup_checkpoint_count,
                    }
                )
            write_json(
                training_summary,
                run_dir / "training" / f"{model_name}_summary.json",
            )

        cached = _prediction_cache(
            model,
            loaders,
            device,
            cache_path,
            reuse=not force,
            mixed_precision=training_config.mixed_precision,
        )
        return {
            "model_name": model_name,
            "model_position": model_position,
            "gpu": device_index,
            "parameter_count": parameter_count,
            "validation_accuracy": float(cached["validation_accuracy"]),
            "test_ms_per_sample": float(cached["test_ms_per_sample"]),
            "cache_path": str(cache_path),
            "training_strategy": (
                "raw_ema_greedy_soup"
                if elite_weight_values is not None
                else "standard_best_checkpoint"
            ),
            "training_profile": {
                **MODEL_DIVERSITY_PROFILES[model_name],
                "learning_rate": training_config.learning_rate,
            },
        }
    finally:
        torch.cuda.empty_cache()
        device_queue.put(device_index)


def _fit_and_report(
    config: ExperimentConfig,
    dataset_name: str,
    seed: int,
    run_dir: Path,
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dataset_config = replace(config.dataset, name=dataset_name, download=False)
    bundle = build_dataset(dataset_config, seed)
    ordered = sorted(summaries, key=lambda item: item["model_position"])
    prediction_sets: list[dict[str, np.ndarray]] = []
    for summary in ordered:
        with np.load(summary["cache_path"]) as cached:
            prediction_sets.append({key: cached[key] for key in cached.files})

    meta_labels = prediction_sets[0]["meta_labels"]
    test_labels = prediction_sets[0]["test_labels"]
    for cached in prediction_sets[1:]:
        if not np.array_equal(meta_labels, cached["meta_labels"]):
            raise RuntimeError("Meta predictions are not sample-aligned")
        if not np.array_equal(test_labels, cached["test_labels"]):
            raise RuntimeError("Test predictions are not sample-aligned")
    stacked_meta = np.stack([values["meta_probabilities"] for values in prediction_sets])
    stacked_test = np.stack([values["test_probabilities"] for values in prediction_sets])
    validation_accuracies = np.asarray([item["validation_accuracy"] for item in ordered])
    model_parameters = [int(item["parameter_count"]) for item in ordered]
    inference_times = [float(item["test_ms_per_sample"]) for item in ordered]

    ensembles = run_ensembles(
        stacked_meta,
        meta_labels,
        stacked_test,
        config.tree,
        seed,
        validation_accuracies,
    )
    combiner_dir = ensure_dir(run_dir / "combiners")
    for method, combiner in ensembles.combiners.items():
        joblib.dump(combiner.estimator, combiner_dir / f"{method}.joblib")

    # Preserve paired predictions so publication statistics can compare two
    # methods on exactly the same test samples without rerunning the CNNs.
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        labels=test_labels,
        **ensembles.predictions,
    )

    detailed_metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    base_ensemble_time = float(sum(inference_times))
    selected_confusions = {
        "cnn_1",
        "cnn_2",
        "cnn_3",
        "majority_vote",
        "soft_vote",
        "logistic_stack",
        "rf_soft",
        "hgb_soft",
        "dt_soft",
    }
    for method, predictions in ensembles.predictions.items():
        metrics = classification_metrics(test_labels, predictions, bundle.class_names)
        complexity = _method_complexity(method, model_parameters, ensembles.combiners)
        if method.startswith("cnn_"):
            position = int(method.split("_")[1]) - 1
            inference_ms = inference_times[position]
        elif method == "dt_soft_best_two":
            positions = list(ensembles.best_two_indices)
            complexity["base_parameters"] = sum(model_parameters[index] for index in positions)
            inference_ms = sum(inference_times[index] for index in positions)
            inference_ms += ensembles.combiners[method].predict_milliseconds_per_sample
        else:
            overhead = (
                ensembles.combiners[method].predict_milliseconds_per_sample
                if method in ensembles.combiners
                else 0.0
            )
            inference_ms = base_ensemble_time + overhead
        display_name = METHOD_LABELS.get(method, method)
        detailed_metrics[method] = {
            **metrics,
            **complexity,
            "display_name": display_name,
            "inference_ms_per_sample": inference_ms,
        }
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "method": display_name,
                "method_key": method,
                "category": "main",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "inference_ms_per_sample": inference_ms,
                **complexity,
            }
        )
        if method in selected_confusions:
            save_confusion_matrix(
                test_labels,
                predictions,
                bundle.class_names,
                f"{dataset_name}: {display_name}",
                run_dir / "figures" / f"confusion_{method}.png",
            )

    for depth_label, (predictions, actual_depth, leaves) in ensembles.depth_ablation.items():
        metrics = classification_metrics(test_labels, predictions, bundle.class_names)
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
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
    feature_names = probability_feature_names(len(ordered), bundle.class_names)
    save_tree_visualization(
        soft_tree,
        feature_names,
        bundle.class_names,
        run_dir / "figures" / "decision_tree_soft.png",
    )
    pd.DataFrame(
        {"feature": feature_names, "importance": soft_tree.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        run_dir / "decision_tree_feature_importance.csv", index=False
    )
    base_predictions = stacked_test.argmax(axis=2)
    write_json(
        {
            "parallel_cuda_training": True,
            "model_gpu_assignments": {
                item["model_name"]: item["gpu"] for item in ordered
            },
            "model_parameters": {
                item["model_name"]: item["parameter_count"] for item in ordered
            },
            "validation_accuracies": {
                item["model_name"]: item["validation_accuracy"] for item in ordered
            },
            "training_profiles": {
                item["model_name"]: item["training_profile"] for item in ordered
            },
            "best_two_model_indices": ensembles.best_two_indices,
            "meta_feature_dimensions": ensembles.meta_feature_dimensions,
            "combiner_best_parameters": {
                name: value.best_parameters for name, value in ensembles.combiners.items()
            },
            "disagreement_analysis": disagreement_analysis(
                base_predictions,
                ensembles.predictions["majority_vote"],
                ensembles.predictions["dt_soft"],
                test_labels,
            ),
            "pairwise_diversity_analysis": pairwise_diversity_analysis(
                base_predictions, test_labels
            ),
            "metrics": detailed_metrics,
        },
        run_dir / "report.json",
    )
    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
    return rows


def run_parallel_cuda(
    config: ExperimentConfig,
    dataset_name: str,
    seed: int,
    max_gpus: int,
    force: bool,
) -> tuple[list[dict[str, Any]], Path]:
    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError("No CUDA GPU detected. Enable a Kaggle or Colab GPU accelerator.")
    gpu_count = min(available_gpus, max_gpus, len(MODEL_BUILDERS))
    dataset_config = replace(config.dataset, name=dataset_name, download=False)
    run_hash = _configuration_hash(config, dataset_name, seed)
    run_dir = ensure_dir(Path(config.output_dir) / dataset_name / f"seed_{seed}" / run_hash)
    write_json(
        {**config.as_dict(), "active_dataset": dataset_name, "active_seed": seed},
        run_dir / "config.json",
    )

    context = mp.get_context("spawn")
    summaries: list[dict[str, Any]] = []
    with context.Manager() as manager:
        device_queue = manager.Queue()
        for device_index in range(gpu_count):
            device_queue.put(device_index)
        with ProcessPoolExecutor(max_workers=gpu_count, mp_context=context) as executor:
            futures = [
                executor.submit(
                    _train_model_task,
                    model_name,
                    position,
                    seed,
                    asdict(dataset_config),
                    asdict(config.training),
                    str(run_dir.resolve()),
                    device_queue,
                    force,
                )
                for position, model_name in enumerate(MODEL_BUILDERS)
            ]
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                print(
                    f"Completed {summary['model_name']} on GPU {summary['gpu']} "
                    f"(validation accuracy={summary['validation_accuracy']:.4f})",
                    flush=True,
                )
    rows = _fit_and_report(config, dataset_name, seed, run_dir, summaries)
    return rows, run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the three TreeStack CNNs concurrently across CUDA GPUs."
    )
    parser.add_argument("--dataset", choices=["fashion_mnist", "cifar10"], default="fashion_mnist")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="artifacts/kaggle")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_epochs = 40 if args.dataset == "fashion_mnist" else 150
    default_patience = 10 if args.dataset == "fashion_mnist" else 20
    dataset_config = DatasetConfig(
        name=args.dataset,
        root=str(Path(args.data_root).resolve()),
        num_workers=args.num_workers,
        download=True,
    )
    training_config = TrainingConfig(
        epochs=args.epochs or default_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        patience=args.patience or default_patience,
        label_smoothing=args.label_smoothing,
        device="auto",
        scheduler="cosine",
        mixed_precision=True,
    )
    config = ExperimentConfig(
        datasets=[args.dataset],
        seeds=args.seeds,
        output_dir=args.output_dir,
        dataset=dataset_config,
        training=training_config,
        tree=TreeConfig(),
    )
    config.validate()
    print(
        f"Detected {torch.cuda.device_count()} CUDA device(s); "
        f"using up to {args.max_gpus} in parallel.",
        flush=True,
    )
    print("Downloading/checking the dataset once before workers start.", flush=True)
    build_dataset(dataset_config, args.seeds[0])
    config.dataset.download = False

    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows, run_dir = run_parallel_cuda(
            config, args.dataset, seed, max_gpus=args.max_gpus, force=args.force
        )
        all_rows.extend(rows)
        print(f"Completed seed {seed}: {run_dir}", flush=True)
    aggregate_path, paper_path = aggregate_results(all_rows, args.output_dir)
    print(f"Aggregate results: {aggregate_path}", flush=True)
    print(f"Paper table: {paper_path}", flush=True)


if __name__ == "__main__":
    main()

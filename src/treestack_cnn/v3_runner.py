from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import DatasetConfig, ExperimentConfig, TrainingConfig, TreeConfig
from .cuda_runner import _fit_and_report, _train_model_task
from .data import build_dataset
from .elite_training import EliteWeightConfig
from .evaluation import classification_metrics, save_confusion_matrix
from .evolution import EvolutionConfig, evolve_fusion, predict_genome
from .experiment import METHOD_LABELS, aggregate_results
from .models import MODEL_BUILDERS, MODEL_VERSION
from .stacking import pairwise_diversity_analysis, soft_vote
from .utils import ensure_dir, write_json


V3_PROTOCOL_VERSION = "v3_elite_weights_evolutionary_fusion_1"


def _v3_configuration_hash(
    config: ExperimentConfig,
    dataset_name: str,
    seed: int,
    elite_config: EliteWeightConfig,
    evolution_config: EvolutionConfig,
) -> str:
    relevant = {
        "dataset": {**asdict(config.dataset), "name": dataset_name},
        "training": asdict(config.training),
        "tree": asdict(config.tree),
        "elite_weights": asdict(elite_config),
        "evolution": asdict(evolution_config),
        "model_version": MODEL_VERSION,
        "protocol_version": V3_PROTOCOL_VERSION,
        "seed": seed,
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _load_prediction_sets(
    summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
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
    return ordered, prediction_sets


def _append_evolutionary_fusion(
    config: ExperimentConfig,
    dataset_name: str,
    seed: int,
    run_dir: Path,
    summaries: list[dict[str, Any]],
    elite_config: EliteWeightConfig,
    evolution_config: EvolutionConfig,
) -> dict[str, Any]:
    ordered, prediction_sets = _load_prediction_sets(summaries)
    stacked_meta = np.stack(
        [values["meta_probabilities"] for values in prediction_sets]
    )
    stacked_test = np.stack(
        [values["test_probabilities"] for values in prediction_sets]
    )
    meta_labels = prediction_sets[0]["meta_labels"]
    test_labels = prediction_sets[0]["test_labels"]
    validation_accuracies = np.asarray(
        [item["validation_accuracy"] for item in ordered], dtype=np.float64
    )
    result = evolve_fusion(
        stacked_meta,
        meta_labels,
        validation_accuracies,
        evolution_config,
        seed,
    )

    started = time.perf_counter()
    predictions, overrides = predict_genome(stacked_test, result.genome)
    fusion_ms = (time.perf_counter() - started) * 1000.0 / len(test_labels)
    bundle = build_dataset(
        replace(config.dataset, name=dataset_name, download=False), seed
    )
    metrics = classification_metrics(test_labels, predictions, bundle.class_names)
    base_parameters = int(sum(item["parameter_count"] for item in ordered))
    base_inference_ms = float(sum(item["test_ms_per_sample"] for item in ordered))
    combiner_parameters = int(2 * len(ordered) + 1)
    display_name = METHOD_LABELS["evolutionary_fusion"]
    row = {
        "dataset": dataset_name,
        "seed": seed,
        "method": display_name,
        "method_key": "evolutionary_fusion",
        "category": "main",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "inference_ms_per_sample": base_inference_ms + fusion_ms,
        "base_parameters": base_parameters,
        "combiner_parameters": combiner_parameters,
        "tree_depth": None,
        "tree_leaves": None,
    }
    results_path = run_dir / "results.csv"
    result_rows = pd.read_csv(results_path)
    result_rows = pd.concat([result_rows, pd.DataFrame([row])], ignore_index=True)
    result_rows.to_csv(results_path, index=False)
    save_confusion_matrix(
        test_labels,
        predictions,
        bundle.class_names,
        f"{dataset_name}: {display_name}",
        run_dir / "figures" / "confusion_evolutionary_fusion.png",
    )

    fallback = soft_vote(stacked_test)
    fallback_correct = fallback == test_labels
    candidate_correct = predictions == test_labels
    corrections = int(np.sum(candidate_correct & ~fallback_correct))
    harms = int(np.sum(~candidate_correct & fallback_correct))
    meta_search_fallback_accuracy = float(
        np.mean(
            soft_vote(stacked_meta[:, result.search_indices])
            == meta_labels[result.search_indices]
        )
    )
    meta_validation_fallback_accuracy = float(
        np.mean(
            soft_vote(stacked_meta[:, result.validation_indices])
            == meta_labels[result.validation_indices]
        )
    )
    meta_dir = ensure_dir(run_dir / "evolution")
    np.savez_compressed(
        meta_dir / "meta_split_indices.npz",
        search=result.search_indices,
        validation=result.validation_indices,
    )
    pd.DataFrame(result.history).to_csv(
        meta_dir / "fusion_search_history.csv", index=False
    )
    write_json(
        {
            "genome": result.genome.as_dict(),
            "search_score": result.search_score.as_dict(),
            "validation_score": result.validation_score.as_dict(),
            "evolution_config": asdict(evolution_config),
        },
        meta_dir / "selected_genome.json",
    )

    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("metrics", {})["evolutionary_fusion"] = {
        **metrics,
        "display_name": display_name,
        "base_parameters": base_parameters,
        "combiner_parameters": combiner_parameters,
        "tree_depth": None,
        "tree_leaves": None,
        "inference_ms_per_sample": base_inference_ms + fusion_ms,
    }
    report["elite_weight_training"] = {
        "config": asdict(elite_config),
        "strategy": "raw, exponential moving average, and greedy checkpoint soup",
        "architectures_are_merged_only_within_their_own_training_trajectory": True,
    }
    report["evolutionary_fusion"] = {
        "protocol_version": V3_PROTOCOL_VERSION,
        "genome": result.genome.as_dict(),
        "search_score": result.search_score.as_dict(),
        "validation_score": result.validation_score.as_dict(),
        "meta_search_soft_vote_accuracy": meta_search_fallback_accuracy,
        "meta_validation_soft_vote_accuracy": meta_validation_fallback_accuracy,
        "meta_search_samples": int(len(result.search_indices)),
        "meta_validation_samples": int(len(result.validation_indices)),
        "test_override_samples": int(np.sum(overrides)),
        "test_override_rate": float(np.mean(overrides)),
        "test_corrections_over_soft_vote": corrections,
        "test_harms_to_soft_vote": harms,
        "test_net_corrections_over_soft_vote": corrections - harms,
        "evolution_config": asdict(evolution_config),
    }
    base_predictions = stacked_test.argmax(axis=2)
    report["pairwise_diversity_analysis"] = pairwise_diversity_analysis(
        base_predictions, test_labels
    )
    write_json(report, report_path)
    return row


def run_parallel_cuda_v3(
    config: ExperimentConfig,
    dataset_name: str,
    seed: int,
    max_gpus: int,
    force: bool,
    elite_config: EliteWeightConfig,
    evolution_config: EvolutionConfig,
) -> tuple[list[dict[str, Any]], Path]:
    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError("No CUDA GPU detected. Enable a Kaggle or Colab GPU accelerator.")
    gpu_count = min(available_gpus, max_gpus, len(MODEL_BUILDERS))
    dataset_config = replace(config.dataset, name=dataset_name, download=False)
    run_hash = _v3_configuration_hash(
        config, dataset_name, seed, elite_config, evolution_config
    )
    run_dir = ensure_dir(
        Path(config.output_dir) / dataset_name / f"seed_{seed}" / run_hash
    )
    write_json(
        {
            **config.as_dict(),
            "active_dataset": dataset_name,
            "active_seed": seed,
            "v3_protocol_version": V3_PROTOCOL_VERSION,
            "elite_weights": asdict(elite_config),
            "evolution": asdict(evolution_config),
        },
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
                    asdict(elite_config),
                )
                for position, model_name in enumerate(MODEL_BUILDERS)
            ]
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                print(
                    f"Completed {summary['model_name']} on GPU {summary['gpu']} "
                    f"(elite validation accuracy={summary['validation_accuracy']:.4f})",
                    flush=True,
                )
    rows = _fit_and_report(config, dataset_name, seed, run_dir, summaries)
    print("Evolving fusion operators on the isolated meta split.", flush=True)
    rows.append(
        _append_evolutionary_fusion(
            config,
            dataset_name,
            seed,
            run_dir,
            summaries,
            elite_config,
            evolution_config,
        )
    )
    return rows, run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V3 elite checkpoint training and evolutionary CNN fusion."
    )
    parser.add_argument(
        "--dataset", choices=["fashion_mnist", "cifar10"], default="fashion_mnist"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="artifacts/v3-kaggle")
    parser.add_argument("--ema-decay", type=float, default=0.98)
    parser.add_argument("--averaging-start-epoch", type=int)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=36)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--meta-validation-fraction", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_epochs = 50 if args.dataset == "fashion_mnist" else 180
    default_patience = 12 if args.dataset == "fashion_mnist" else 24
    epochs = args.epochs or default_epochs
    averaging_start = args.averaging_start_epoch or min(epochs, max(2, epochs // 4))
    dataset_config = DatasetConfig(
        name=args.dataset,
        root=str(Path(args.data_root).resolve()),
        num_workers=args.num_workers,
        download=True,
    )
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        patience=args.patience or default_patience,
        label_smoothing=0.0,
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
    elite_config = EliteWeightConfig(
        ema_decay=args.ema_decay,
        averaging_start_epoch=averaging_start,
    )
    evolution_config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        meta_validation_fraction=args.meta_validation_fraction,
    )
    config.validate()
    elite_config.validate(epochs)
    evolution_config.validate()
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
        rows, run_dir = run_parallel_cuda_v3(
            config,
            args.dataset,
            seed,
            max_gpus=args.max_gpus,
            force=args.force,
            elite_config=elite_config,
            evolution_config=evolution_config,
        )
        all_rows.extend(rows)
        print(f"Completed V3 seed {seed}: {run_dir}", flush=True)
    aggregate_path, paper_path = aggregate_results(all_rows, args.output_dir)
    print(f"Aggregate results: {aggregate_path}", flush=True)
    print(f"Paper table: {paper_path}", flush=True)


if __name__ == "__main__":
    main()

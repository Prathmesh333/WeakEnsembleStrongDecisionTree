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
from .experiment import aggregate_results
from .models import MODEL_BUILDERS, MODEL_VERSION
from .stacking import pairwise_diversity_analysis, soft_vote
from .statistics import exact_mcnemar, paired_test_table, publication_summary
from .utils import ensure_dir, write_json
from .v3_runner import _load_prediction_sets


SUPPORTED_DATASETS = ("fashion_mnist", "cifar10")
V4_PROTOCOL_VERSION = "v4_search_only_selection_audit_holdout_1"
V4_DISPLAY_NAME = "Evolutionary Fusion (V4)"

def _v4_configuration_hash(
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
        "protocol_version": V4_PROTOCOL_VERSION,
        "seed": seed,
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _upsert_result_row(results_path: Path, row: dict[str, Any]) -> None:
    result_rows = pd.read_csv(results_path)
    result_rows = result_rows.loc[
        result_rows["method_key"] != row["method_key"]
    ].reset_index(drop=True)
    for column in row:
        if column not in result_rows.columns:
            result_rows[column] = np.nan
    complete_row = {column: row.get(column, np.nan) for column in result_rows.columns}
    result_rows.loc[len(result_rows)] = complete_row
    result_rows.to_csv(results_path, index=False)


def _append_test_predictions(
    run_dir: Path, predictions: np.ndarray, key: str = "evolutionary_fusion_v4"
) -> None:
    path = run_dir / "test_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing paired prediction artifact: {path}")
    with np.load(path) as stored:
        values = {name: stored[name] for name in stored.files}
    values[key] = np.asarray(predictions, dtype=np.int64)
    np.savez_compressed(path, **values)


def _append_evolutionary_fusion_v4(
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
    row = {
        "dataset": dataset_name,
        "seed": seed,
        "method": V4_DISPLAY_NAME,
        "method_key": "evolutionary_fusion_v4",
        "category": "main",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "inference_ms_per_sample": base_inference_ms + fusion_ms,
        "base_parameters": base_parameters,
        "combiner_parameters": combiner_parameters,
        "tree_depth": None,
        "tree_leaves": None,
    }
    _upsert_result_row(run_dir / "results.csv", row)
    _append_test_predictions(run_dir, predictions)
    save_confusion_matrix(
        test_labels,
        predictions,
        bundle.class_names,
        f"{dataset_name}: {V4_DISPLAY_NAME}",
        run_dir / "figures" / "confusion_evolutionary_fusion_v4.png",
    )

    fallback = soft_vote(stacked_test)
    paired = exact_mcnemar(test_labels, predictions, fallback)
    meta_search_fallback_accuracy = float(
        np.mean(
            soft_vote(stacked_meta[:, result.search_indices])
            == meta_labels[result.search_indices]
        )
    )
    meta_audit_fallback_accuracy = float(
        np.mean(
            soft_vote(stacked_meta[:, result.validation_indices])
            == meta_labels[result.validation_indices]
        )
    )
    evolution_dir = ensure_dir(run_dir / "evolution")
    np.savez_compressed(
        evolution_dir / "meta_split_indices.npz",
        search=result.search_indices,
        audit=result.validation_indices,
    )
    pd.DataFrame(result.history).to_csv(
        evolution_dir / "fusion_search_history.csv", index=False
    )
    write_json(
        {
            "genome": result.genome.as_dict(),
            "search_score": result.search_score.as_dict(),
            "audit_score": result.validation_score.as_dict(),
            "evolution_config": asdict(evolution_config),
            "selection_partition": "search",
            "audit_partition_used_for_selection": False,
            "search_candidate_count": result.search_candidate_count,
        },
        evolution_dir / "selected_genome.json",
    )

    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("metrics", {})["evolutionary_fusion_v4"] = {
        **metrics,
        "display_name": V4_DISPLAY_NAME,
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
    report["evolutionary_fusion_v4"] = {
        "protocol_version": V4_PROTOCOL_VERSION,
        "genome": result.genome.as_dict(),
        "search_score": result.search_score.as_dict(),
        "audit_score": result.validation_score.as_dict(),
        "meta_search_soft_vote_accuracy": meta_search_fallback_accuracy,
        "meta_audit_soft_vote_accuracy": meta_audit_fallback_accuracy,
        "meta_search_samples": int(len(result.search_indices)),
        "meta_audit_samples": int(len(result.validation_indices)),
        "selection_partition": "search",
        "audit_partition_used_for_selection": False,
        "audit_genomes_evaluated": 1,
        "search_candidate_count": result.search_candidate_count,
        "test_override_samples": int(np.sum(overrides)),
        "test_override_rate": float(np.mean(overrides)),
        "test_comparison_to_soft_vote": paired,
        "evolution_config": asdict(evolution_config),
    }
    base_predictions = stacked_test.argmax(axis=2)
    report["pairwise_diversity_analysis"] = pairwise_diversity_analysis(
        base_predictions, test_labels
    )
    write_json(report, report_path)
    return row


def run_parallel_cuda_v4(
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
        raise RuntimeError(
            "No CUDA GPU detected. Enable Kaggle's GPU T4 x2 accelerator and rerun."
        )
    gpu_count = min(available_gpus, max_gpus, len(MODEL_BUILDERS))
    dataset_config = replace(config.dataset, name=dataset_name, download=False)
    run_hash = _v4_configuration_hash(
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
            "v4_protocol_version": V4_PROTOCOL_VERSION,
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
    print(
        "Selecting the fusion genome on meta-search; the audit split is score-only.",
        flush=True,
    )
    rows.append(
        _append_evolutionary_fusion_v4(
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
        description=(
            "Run the complete V4 benchmark on Fashion-MNIST and CIFAR-10 with "
            "search-only genome selection and an audit-only meta holdout."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(SUPPORTED_DATASETS),
        default=list(SUPPORTED_DATASETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--fashion-epochs", type=int, default=50)
    parser.add_argument("--cifar10-epochs", type=int, default=180)
    parser.add_argument("--fashion-patience", type=int, default=12)
    parser.add_argument("--cifar10-patience", type=int, default=24)
    parser.add_argument("--fashion-averaging-start", type=int, default=12)
    parser.add_argument("--cifar10-averaging-start", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="artifacts/v4-full")
    parser.add_argument("--ema-decay", type=float, default=0.98)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=36)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--meta-audit-fraction", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    return parser


def _dataset_runtime(args: argparse.Namespace, dataset_name: str) -> dict[str, int]:
    if dataset_name == "fashion_mnist":
        return {
            "epochs": args.fashion_epochs,
            "patience": args.fashion_patience,
            "averaging_start": args.fashion_averaging_start,
        }
    return {
        "epochs": args.cifar10_epochs,
        "patience": args.cifar10_patience,
        "averaging_start": args.cifar10_averaging_start,
    }


def main() -> None:
    args = build_parser().parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Each dataset may be listed only once")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Each random seed may be listed only once")
    if args.max_gpus < 1:
        raise ValueError("max-gpus must be at least 1")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if torch.cuda.device_count() < 1:
        raise RuntimeError(
            "No CUDA GPU detected. Enable Kaggle's GPU T4 x2 accelerator and rerun."
        )
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve()
    ensure_dir(output_dir)
    print(
        f"Detected {torch.cuda.device_count()} CUDA device(s); "
        f"using up to {args.max_gpus} for independent CNN training.",
        flush=True,
    )

    evolution_config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        meta_validation_fraction=args.meta_audit_fraction,
    )
    evolution_config.validate()
    all_rows: list[dict[str, Any]] = []
    completed_runs: list[dict[str, Any]] = []
    manifest_path = output_dir / "run_manifest.json"

    for dataset_name in args.datasets:
        runtime = _dataset_runtime(args, dataset_name)
        if runtime["patience"] < 1:
            raise ValueError(f"Patience must be positive for {dataset_name}")
        dataset_config = DatasetConfig(
            name=dataset_name,
            root=str(data_root),
            num_workers=args.num_workers,
            download=True,
        )
        training_config = TrainingConfig(
            epochs=runtime["epochs"],
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            patience=runtime["patience"],
            label_smoothing=0.0,
            device="auto",
            scheduler="cosine",
            mixed_precision=True,
        )
        config = ExperimentConfig(
            datasets=list(args.datasets),
            seeds=list(args.seeds),
            output_dir=str(output_dir),
            dataset=dataset_config,
            training=training_config,
            tree=TreeConfig(),
        )
        elite_config = EliteWeightConfig(
            ema_decay=args.ema_decay,
            averaging_start_epoch=runtime["averaging_start"],
        )
        config.validate()
        elite_config.validate(runtime["epochs"])
        print(f"Downloading/checking {dataset_name} once before workers start.", flush=True)
        build_dataset(dataset_config, args.seeds[0])
        config.dataset.download = False

        for seed in args.seeds:
            rows, run_dir = run_parallel_cuda_v4(
                config,
                dataset_name,
                seed,
                max_gpus=args.max_gpus,
                force=args.force,
                elite_config=elite_config,
                evolution_config=evolution_config,
            )
            all_rows.extend(rows)
            completed_runs.append(
                {"dataset": dataset_name, "seed": seed, "run_dir": str(run_dir)}
            )
            write_json(
                {
                    "protocol_version": V4_PROTOCOL_VERSION,
                    "requested_datasets": list(args.datasets),
                    "requested_seeds": list(args.seeds),
                    "completed_runs": completed_runs,
                },
                manifest_path,
            )
            print(f"Completed V4 {dataset_name} seed {seed}: {run_dir}", flush=True)

    aggregate_path, paper_path = aggregate_results(all_rows, output_dir)
    publication_path = publication_summary(all_rows, output_dir)
    paired_path = paired_test_table(completed_runs, output_dir)
    print(f"Aggregate results: {aggregate_path}", flush=True)
    print(f"Paper accuracy table: {paper_path}", flush=True)
    print(f"Publication summary with 95% CIs: {publication_path}", flush=True)
    print(f"Paired exact tests: {paired_path}", flush=True)


if __name__ == "__main__":
    main()

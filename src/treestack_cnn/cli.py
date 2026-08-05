from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .experiment import aggregate_results, run_single_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="treestack",
        description="Run leakage-free TreeStack-CNN experiments.",
    )
    parser.add_argument("--config", type=Path, help="YAML experiment configuration")
    parser.add_argument("--datasets", nargs="+", choices=["fashion_mnist", "cifar10"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain models and regenerate predictions even when cached artifacts exist",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.datasets:
        config.datasets = args.datasets
    if args.seeds:
        config.seeds = args.seeds
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device:
        config.training.device = args.device
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.no_download:
        config.dataset.download = False
    config.validate()

    all_rows = []
    for dataset_name in config.datasets:
        for seed in config.seeds:
            rows, run_dir = run_single_experiment(config, dataset_name, seed, force=args.force)
            all_rows.extend(rows)
            print(f"Completed {dataset_name} seed={seed}: {run_dir}")
    aggregate_path, paper_path = aggregate_results(all_rows, config.output_dir)
    print(f"Aggregate results: {aggregate_path}")
    print(f"Paper table: {paper_path}")


if __name__ == "__main__":
    main()

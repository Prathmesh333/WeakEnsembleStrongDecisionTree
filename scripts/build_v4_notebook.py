from __future__ import annotations

import json
import textwrap
from pathlib import Path


TARGET = Path("notebooks/treestack_v4_full_benchmark_kaggle.ipynb")


def _source(value: str) -> list[str]:
    text = textwrap.dedent(value).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(value: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


cells = [
    markdown(
        """
        # TreeStack V4 — full two-dataset, three-seed benchmark

        This notebook runs the complete repository benchmark on **Fashion-MNIST and CIFAR-10** with seeds **17, 42, and 73**. It trains the three heterogeneous CNNs on up to two T4 GPUs, evaluates raw/EMA/greedy-soup checkpoints, fits all voting and stacking baselines, and searches the evolutionary fusion rule.

        V4 corrects the V3 selection flaw: all genomes compete only on the meta-search partition. Exactly one frozen genome is then evaluated on the audit partition. The audit result never chooses or modifies the genome, and the official test set is evaluated only after freezing.
        """
    ),
    markdown(
        """
        ## 1. Choose the run mode

        `paper` is the full benchmark. `smoke` runs both datasets for two epochs with one seed so you can verify a new Kaggle environment before spending GPU hours. Completed checkpoints and predictions are reused when the same configuration is rerun.
        """
    ),
    code(
        """
        RUN_MODE = "paper"  # "paper" or "smoke"
        DATASETS = ["fashion_mnist", "cifar10"]

        if RUN_MODE not in {"paper", "smoke"}:
            raise ValueError("RUN_MODE must be 'paper' or 'smoke'")

        PAPER_SETTINGS = {
            "seeds": [17, 42, 73],
            "fashion_epochs": 50,
            "cifar10_epochs": 180,
            "fashion_patience": 12,
            "cifar10_patience": 24,
            "fashion_averaging_start": 12,
            "cifar10_averaging_start": 45,
            "generations": 30,
            "population_size": 36,
            "elite_count": 4,
        }
        SMOKE_SETTINGS = {
            "seeds": [42],
            "fashion_epochs": 2,
            "cifar10_epochs": 2,
            "fashion_patience": 2,
            "cifar10_patience": 2,
            "fashion_averaging_start": 1,
            "cifar10_averaging_start": 1,
            "generations": 2,
            "population_size": 8,
            "elite_count": 2,
        }
        SETTINGS = PAPER_SETTINGS if RUN_MODE == "paper" else SMOKE_SETTINGS

        BATCH_SIZE = 128
        MAX_GPUS = 2
        NUM_WORKERS = 1
        META_AUDIT_FRACTION = 0.20
        FORCE_RETRAIN = False
        RUN_REPOSITORY_TESTS = True

        GIT_REPOSITORY = "https://github.com/Prathmesh333/WeakEnsembleStrongDecisionTree.git"
        GIT_REF = "main"
        print({"mode": RUN_MODE, "datasets": DATASETS, **SETTINGS})
        """
    ),
    markdown("## 2. Create stable Kaggle/Colab paths"),
    code(
        """
        import os
        import sys
        from pathlib import Path

        if Path("/kaggle/working").exists():
            PLATFORM = "kaggle"
            WORK_ROOT = Path("/kaggle/working")
        elif Path("/content").exists():
            PLATFORM = "colab"
            WORK_ROOT = Path("/content")
        else:
            PLATFORM = "local"
            WORK_ROOT = Path.cwd()

        REPO_DIR = WORK_ROOT / "WeakEnsembleStrongDecisionTree"
        DATA_ROOT = WORK_ROOT / "treestack-v4-data"
        OUTPUT_DIR = WORK_ROOT / f"treestack-v4-{RUN_MODE}-results"
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print({"platform": PLATFORM, "repo": str(REPO_DIR), "output": str(OUTPUT_DIR)})
        """
    ),
    markdown(
        """
        ## 3. Clone or update the repository

        Internet access must be enabled in Kaggle for the first clone and dataset download. A rerun updates an existing clean clone with a fast-forward pull.
        """
    ),
    code(
        """
        import subprocess

        if (REPO_DIR / ".git").exists():
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", GIT_REF], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "checkout", GIT_REF], check=True)
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", GIT_REF],
                check=True,
            )
        elif REPO_DIR.exists():
            raise RuntimeError(f"{REPO_DIR} exists but is not a Git repository; rename or remove it")
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", GIT_REF, GIT_REPOSITORY, str(REPO_DIR)],
                check=True,
            )

        COMMIT = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
        print("Repository commit:", COMMIT)
        """
    ),
    markdown(
        """
        ## 4. Activate the source for this kernel and every child process

        This is the protection against `ModuleNotFoundError: No module named treestack_cnn`. The current kernel receives the source path, and the same `PYTHONPATH` is passed to the subprocess that launches multiprocessing workers.
        """
    ),
    code(
        """
        SRC_DIR = (REPO_DIR / "src").resolve()
        if not (SRC_DIR / "treestack_cnn" / "v4_runner.py").exists():
            raise FileNotFoundError(f"V4 source is missing from commit {COMMIT}: {SRC_DIR}")
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        RUN_ENV = os.environ.copy()
        previous_pythonpath = RUN_ENV.get("PYTHONPATH", "")
        RUN_ENV["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
        os.environ["PYTHONPATH"] = RUN_ENV["PYTHONPATH"]
        print("Source path activated:", SRC_DIR)
        """
    ),
    markdown("## 5. Verify Python dependencies"),
    code(
        """
        import importlib.util

        REQUIRED = {
            "joblib": "joblib",
            "matplotlib": "matplotlib",
            "numpy": "numpy",
            "pandas": "pandas",
            "yaml": "PyYAML",
            "sklearn": "scikit-learn",
            "scipy": "scipy",
            "seaborn": "seaborn",
            "torch": "torch",
            "torchvision": "torchvision",
            "pytest": "pytest",
        }
        missing_packages = [package for module, package in REQUIRED.items() if importlib.util.find_spec(module) is None]
        if missing_packages:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", *missing_packages],
                check=True,
            )
        import importlib

        importlib.invalidate_caches()
        import treestack_cnn
        from treestack_cnn.v4_runner import V4_PROTOCOL_VERSION

        child_check = subprocess.run(
            [sys.executable, "-c", "import treestack_cnn.v4_runner; print('child import: PASS')"],
            cwd=REPO_DIR,
            env=RUN_ENV,
            text=True,
            capture_output=True,
            check=True,
        )
        print(child_check.stdout.strip())
        print("Package:", treestack_cnn.__file__)
        print("Protocol:", V4_PROTOCOL_VERSION)
        print("Dependency check: PASS")
        """
    ),
    markdown("## 6. Verify the GPU accelerator"),
    code(
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected. In Kaggle select Settings > Accelerator > GPU T4 x2.")
        print(f"PyTorch {torch.__version__}; CUDA devices: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(f"cuda:{index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")
        if torch.cuda.device_count() < 2:
            print("Warning: only one GPU is available. The run remains correct but trains CNNs sequentially.")
        """
    ),
    markdown(
        """
        ## 7. Run repository tests before training

        These tests compile every notebook code cell and check probability normalization, split isolation, deterministic evolution, audit-only selection, prediction artifacts, and paired statistics.
        """
    ),
    code(
        """
        if RUN_REPOSITORY_TESTS:
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=REPO_DIR,
                env=RUN_ENV,
                check=True,
            )
        else:
            print("Repository tests skipped by configuration")
        """
    ),
    markdown("## 8. Inspect model and optimizer diversity"),
    code(
        """
        import pandas as pd
        from IPython.display import display

        from treestack_cnn.models import build_models
        from treestack_cnn.profiles import MODEL_DIVERSITY_PROFILES
        from treestack_cnn.utils import count_parameters

        profile_rows = []
        for dataset_name, channels, classes in [("fashion_mnist", 1, 10), ("cifar10", 3, 10)]:
            for model_name, model in build_models(channels, classes).items():
                profile_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "parameters": count_parameters(model),
                        **MODEL_DIVERSITY_PROFILES[model_name],
                    }
                )
        display(pd.DataFrame(profile_rows))
        """
    ),
    markdown(
        """
        ## 9. Download both datasets and audit every split

        The official test sets remain untouched. Only the official training set is divided into CNN base data and meta-level data. CNN early stopping uses a subset of the base partition, never the meta or test samples.
        """
    ),
    code(
        """
        import numpy as np

        from treestack_cnn.config import DatasetConfig
        from treestack_cnn.data import build_dataset

        split_rows = []
        for dataset_name in DATASETS:
            bundle = build_dataset(
                DatasetConfig(name=dataset_name, root=str(DATA_ROOT), download=True),
                SETTINGS["seeds"][0],
            )
            split_sets = {
                "base": set(bundle.splits.base.tolist()),
                "meta": set(bundle.splits.meta.tolist()),
                "test": set(bundle.splits.test.tolist()),
            }
            disjoint = not (
                split_sets["base"] & split_sets["meta"]
                or split_sets["base"] & split_sets["test"]
                or split_sets["meta"] & split_sets["test"]
            )
            if not disjoint:
                raise RuntimeError(f"Leakage detected in {dataset_name}")
            split_rows.append(
                {
                    "dataset": dataset_name,
                    "cnn_train": len(bundle.splits.base_train),
                    "cnn_validation": len(bundle.splits.base_validation),
                    "meta_total": len(bundle.splits.meta),
                    "meta_search": round(len(bundle.splits.meta) * (1 - META_AUDIT_FRACTION)),
                    "meta_audit": round(len(bundle.splits.meta) * META_AUDIT_FRACTION),
                    "official_test": len(bundle.splits.test),
                    "disjoint": disjoint,
                }
            )
        display(pd.DataFrame(split_rows))
        print("Leakage audit: PASS")
        """
    ),
    markdown(
        """
        ## 10. Build the complete resumable command

        The child process uses the same Python executable as this notebook. If Kaggle interrupts the run, rerun this cell with the same settings; matching checkpoints and probability caches are reused. Set `FORCE_RETRAIN=True` only when you intentionally want to discard reusable training work.
        """
    ),
    code(
        """
        command = [
            sys.executable,
            "-m",
            "treestack_cnn.v4_runner",
            "--datasets",
            *DATASETS,
            "--seeds",
            *[str(seed) for seed in SETTINGS["seeds"]],
            "--fashion-epochs",
            str(SETTINGS["fashion_epochs"]),
            "--cifar10-epochs",
            str(SETTINGS["cifar10_epochs"]),
            "--fashion-patience",
            str(SETTINGS["fashion_patience"]),
            "--cifar10-patience",
            str(SETTINGS["cifar10_patience"]),
            "--fashion-averaging-start",
            str(SETTINGS["fashion_averaging_start"]),
            "--cifar10-averaging-start",
            str(SETTINGS["cifar10_averaging_start"]),
            "--batch-size",
            str(BATCH_SIZE),
            "--max-gpus",
            str(MAX_GPUS),
            "--num-workers",
            str(NUM_WORKERS),
            "--generations",
            str(SETTINGS["generations"]),
            "--population-size",
            str(SETTINGS["population_size"]),
            "--elite-count",
            str(SETTINGS["elite_count"]),
            "--meta-audit-fraction",
            str(META_AUDIT_FRACTION),
            "--data-root",
            str(DATA_ROOT),
            "--output-dir",
            str(OUTPUT_DIR),
        ]
        if FORCE_RETRAIN:
            command.append("--force")
        print("Running:", " ".join(command))
        """
    ),
    markdown("## 11. Run all datasets and seeds"),
    code(
        """
        subprocess.run(command, cwd=REPO_DIR, env=RUN_ENV, check=True)
        """
    ),
    markdown("## 12. Verify that every requested run completed"),
    code(
        """
        import json

        manifest_path = OUTPUT_DIR / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("No run manifest was produced. Inspect the training cell output.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {(dataset, seed) for dataset in DATASETS for seed in SETTINGS["seeds"]}
        completed = {(item["dataset"], item["seed"]) for item in manifest["completed_runs"]}
        missing = sorted(expected - completed)
        if missing:
            raise RuntimeError(f"Incomplete runs: {missing}. Rerun the training cell to resume.")
        run_records = manifest["completed_runs"]
        print(f"Completeness audit: PASS ({len(completed)}/{len(expected)} runs)")
        """
    ),
    markdown("## 13. Main accuracy and macro-F1 results"),
    code(
        """
        aggregate = pd.read_csv(OUTPUT_DIR / "aggregate_results.csv")
        main_results = aggregate.loc[aggregate["category"] == "main"].copy()
        main_results["accuracy_percent"] = 100 * main_results["accuracy_mean"]
        main_results["accuracy_std_percent"] = 100 * main_results["accuracy_std"]
        main_results["macro_f1_percent"] = 100 * main_results["macro_f1_mean"]
        display(
            main_results.sort_values(["dataset", "accuracy_mean"], ascending=[True, False])[
                ["dataset", "method", "seeds", "accuracy_percent", "accuracy_std_percent", "macro_f1_percent"]
            ]
        )
        """
    ),
    markdown("## 14. Publication table with bootstrap 95% confidence intervals"),
    code(
        """
        publication = pd.read_csv(OUTPUT_DIR / "publication_summary.csv")
        for column in ["accuracy_mean", "accuracy_std", "accuracy_ci95_low", "accuracy_ci95_high"]:
            publication[column + "_percent"] = 100 * publication[column]
        display(
            publication.sort_values(["dataset", "accuracy_mean"], ascending=[True, False])[
                [
                    "dataset", "method", "seeds", "accuracy_mean_percent",
                    "accuracy_std_percent", "accuracy_ci95_low_percent", "accuracy_ci95_high_percent",
                ]
            ]
        )
        """
    ),
    markdown(
        """
        ## 15. Exact paired tests

        McNemar's exact test uses correction-versus-harm pairs on the same official test samples. Holm-adjusted values account for the family of comparisons printed here. Do not treat a one-image gain as evidence without these paired counts.
        """
    ),
    code(
        """
        paired = pd.read_csv(OUTPUT_DIR / "paired_tests.csv")
        paired["accuracy_delta_pp"] = 100 * paired["accuracy_delta"]
        display(
            paired[
                [
                    "dataset", "seed", "candidate", "reference", "accuracy_delta_pp",
                    "corrections", "harms", "net_corrections", "mcnemar_exact_p", "mcnemar_holm_p",
                ]
            ].sort_values(["dataset", "seed", "reference"])
        )
        """
    ),
    markdown("## 16. Plot mean accuracy with seed variation"),
    code(
        """
        import matplotlib.pyplot as plt

        selected_methods = [
            "CNN-3", "Soft Vote", "Logistic Stack", "Random Forest Stack",
            "DT-Soft", "Evolutionary Fusion (V4)",
        ]
        figure, axes = plt.subplots(1, len(DATASETS), figsize=(16, 5), squeeze=False)
        for axis, dataset_name in zip(axes[0], DATASETS):
            view = main_results[
                (main_results["dataset"] == dataset_name)
                & (main_results["method"].isin(selected_methods))
            ].copy().sort_values("accuracy_mean")
            axis.barh(
                view["method"], 100 * view["accuracy_mean"],
                xerr=100 * view["accuracy_std"], color="#0072B2", alpha=0.85,
            )
            axis.set(title=dataset_name, xlabel="Test accuracy (%)")
            axis.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown("## 17. Audit the frozen-genome protocol"),
    code(
        """
        protocol_rows = []
        reports = {}
        for item in run_records:
            run_dir = Path(item["run_dir"])
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            reports[(item["dataset"], item["seed"])] = (run_dir, report)
            audit = report["evolutionary_fusion_v4"]
            protocol_rows.append(
                {
                    "dataset": item["dataset"],
                    "seed": item["seed"],
                    "selection_partition": audit["selection_partition"],
                    "audit_used_for_selection": audit["audit_partition_used_for_selection"],
                    "audit_genomes_evaluated": audit["audit_genomes_evaluated"],
                    "search_candidates": audit["search_candidate_count"],
                    "search_samples": audit["meta_search_samples"],
                    "audit_samples": audit["meta_audit_samples"],
                }
            )
        protocol_table = pd.DataFrame(protocol_rows)
        if not (
            (protocol_table["selection_partition"] == "search").all()
            and (~protocol_table["audit_used_for_selection"]).all()
            and (protocol_table["audit_genomes_evaluated"] == 1).all()
        ):
            raise RuntimeError("Protocol audit failed")
        display(protocol_table)
        print("Frozen-genome audit: PASS")
        """
    ),
    markdown("## 18. Inspect which checkpoint strategy actually won"),
    code(
        """
        checkpoint_rows = []
        for (dataset_name, seed), (run_dir, report) in reports.items():
            for model_name in report["training_profiles"]:
                history = pd.read_csv(run_dir / "training" / f"{model_name}.csv")
                best = history.loc[history["elite_validation_accuracy"].idxmax()]
                checkpoint_rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": model_name,
                        "epochs_run": len(history),
                        "best_epoch": int(best["epoch"]),
                        "elite_kind": best["elite_kind"],
                        "elite_validation_accuracy": best["elite_validation_accuracy"],
                        "accepted_soup_checkpoints": int(best["soup_checkpoint_count"]),
                    }
                )
        checkpoint_table = pd.DataFrame(checkpoint_rows)
        display(checkpoint_table)
        display(checkpoint_table.groupby(["dataset", "elite_kind"]).size().rename("wins").reset_index())
        """
    ),
    markdown("## 19. Plot raw, EMA, soup, and retained-elite trajectories"),
    code(
        """
        example_seed = 42 if 42 in SETTINGS["seeds"] else SETTINGS["seeds"][0]
        for dataset_name in DATASETS:
            run_dir, report = reports[(dataset_name, example_seed)]
            figure, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
            for axis, model_name in zip(axes, report["training_profiles"]):
                history = pd.read_csv(run_dir / "training" / f"{model_name}.csv")
                axis.plot(history["epoch"], history["validation_accuracy"], label="raw")
                axis.plot(history["epoch"], history["ema_validation_accuracy"], label="EMA")
                axis.plot(history["epoch"], history["soup_best_validation_accuracy"], label="greedy soup")
                axis.plot(history["epoch"], history["elite_validation_accuracy"], linewidth=2.2, label="retained elite")
                axis.set(title=f"{dataset_name}: {model_name}", xlabel="Epoch", ylabel="Validation accuracy")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8)
            plt.tight_layout()
            plt.show()
        """
    ),
    markdown("## 20. Inspect selected operators, weights, and temperatures"),
    code(
        """
        genome_rows = []
        for (dataset_name, seed), (_, report) in reports.items():
            evolution = report["evolutionary_fusion_v4"]
            genome = evolution["genome"]
            model_names = list(report["training_profiles"])
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "operator": genome["operator"],
                "gate_threshold": genome["gate_threshold"],
                "require_disagreement": genome["require_disagreement"],
            }
            for index, model_name in enumerate(model_names):
                row[f"weight_{model_name}"] = genome["weights"][index]
                row[f"temperature_{model_name}"] = genome["temperatures"][index]
            genome_rows.append(row)
        display(pd.DataFrame(genome_rows))
        """
    ),
    markdown("## 21. Compare meta-search and audit performance"),
    code(
        """
        audit_rows = []
        for (dataset_name, seed), (_, report) in reports.items():
            evolution = report["evolutionary_fusion_v4"]
            audit_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "search_v4": evolution["search_score"]["accuracy"],
                    "search_soft": evolution["meta_search_soft_vote_accuracy"],
                    "audit_v4": evolution["audit_score"]["accuracy"],
                    "audit_soft": evolution["meta_audit_soft_vote_accuracy"],
                    "audit_delta_pp": 100 * (
                        evolution["audit_score"]["accuracy"]
                        - evolution["meta_audit_soft_vote_accuracy"]
                    ),
                }
            )
        display(pd.DataFrame(audit_rows))
        """
    ),
    markdown("## 22. Audit test corrections, harms, and override rates"),
    code(
        """
        correction_rows = []
        for (dataset_name, seed), (_, report) in reports.items():
            evolution = report["evolutionary_fusion_v4"]
            paired_soft = evolution["test_comparison_to_soft_vote"]
            correction_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "overrides": evolution["test_override_samples"],
                    "override_rate_percent": 100 * evolution["test_override_rate"],
                    "corrections": paired_soft["corrections"],
                    "harms": paired_soft["harms"],
                    "net": paired_soft["net_corrections"],
                    "exact_p": paired_soft["mcnemar_exact_p"],
                }
            )
        display(pd.DataFrame(correction_rows))
        """
    ),
    markdown("## 23. Confirm that CNN diversity did not collapse"),
    code(
        """
        diversity_rows = []
        for (dataset_name, seed), (_, report) in reports.items():
            diversity = report["pairwise_diversity_analysis"]
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "oracle_accuracy": diversity["oracle_accuracy_any_cnn_correct"],
            }
            for pair, values in diversity["pairwise"].items():
                row[f"{pair}_disagreement"] = values["prediction_disagreement_rate"]
                row[f"{pair}_double_fault"] = values["double_fault_rate"]
            diversity_rows.append(row)
        display(pd.DataFrame(diversity_rows))
        """
    ),
    markdown(
        """
        ## 24. Diagnose the Best-2 decision-tree ablation

        V3 produced an anomalous Best-2 result. This table preserves the selected model indices and fitted hyperparameters so a collapse cannot pass unnoticed.
        """
    ),
    code(
        """
        best_two_rows = []
        all_runs = pd.read_csv(OUTPUT_DIR / "all_runs.csv")
        for (dataset_name, seed), (_, report) in reports.items():
            run_slice = all_runs[(all_runs["dataset"] == dataset_name) & (all_runs["seed"] == seed)]
            scores = run_slice.set_index("method_key")["accuracy"]
            best_two_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "selected_indices": report["best_two_model_indices"],
                    "best_two_accuracy": scores.get("dt_soft_best_two", np.nan),
                    "all_three_dt_soft_accuracy": scores.get("dt_soft", np.nan),
                    "best_parameters": report["combiner_best_parameters"]["dt_soft_best_two"],
                }
            )
        display(pd.DataFrame(best_two_rows))
        """
    ),
    markdown("## 25. Display representative confusion matrices"),
    code(
        """
        from IPython.display import Image, display

        for dataset_name in DATASETS:
            run_dir, _ = reports[(dataset_name, example_seed)]
            print(f"\\n{dataset_name}, seed={example_seed}")
            for filename, title in [
                ("confusion_soft_vote.png", "Soft Vote"),
                ("confusion_rf_soft.png", "Random Forest Stack"),
                ("confusion_dt_soft.png", "DT-Soft"),
                ("confusion_evolutionary_fusion_v4.png", "Evolutionary Fusion V4"),
            ]:
                print(title)
                display(Image(filename=str(run_dir / "figures" / filename), width=600))
        """
    ),
    markdown("## 26. Publication-readiness gates"),
    code(
        """
        gate_rows = []
        for dataset_name in DATASETS:
            dataset_main = main_results[main_results["dataset"] == dataset_name].set_index("method")
            v4 = dataset_main.loc["Evolutionary Fusion (V4)", "accuracy_mean"]
            soft = dataset_main.loc["Soft Vote", "accuracy_mean"]
            logistic = dataset_main.loc["Logistic Stack", "accuracy_mean"]
            forest = dataset_main.loc["Random Forest Stack", "accuracy_mean"]
            dataset_protocol = protocol_table[protocol_table["dataset"] == dataset_name]
            gate_rows.extend(
                [
                    (dataset_name, "Three independent seeds", len(SETTINGS["seeds"]) >= 3, len(SETTINGS["seeds"])),
                    (dataset_name, "Audit is score-only", (~dataset_protocol["audit_used_for_selection"]).all(), "one frozen genome"),
                    (dataset_name, "V4 mean exceeds Soft Vote", v4 > soft, f"delta={100 * (v4-soft):+.3f} pp"),
                    (dataset_name, "V4 mean exceeds Logistic Stack", v4 > logistic, f"delta={100 * (v4-logistic):+.3f} pp"),
                    (dataset_name, "V4 mean exceeds Random Forest", v4 > forest, f"delta={100 * (v4-forest):+.3f} pp"),
                ]
            )
        gate_table = pd.DataFrame(gate_rows, columns=["dataset", "gate", "passed", "evidence"])
        display(gate_table)
        print("A passed accuracy gate is not a significance claim; use the paired-test table above.")
        """
    ),
    markdown("## 27. Verify all required artifacts"),
    code(
        """
        required_root_files = [
            "all_runs.csv", "aggregate_results.csv", "paper_accuracy_table.csv",
            "publication_summary.csv", "paired_tests.csv", "run_manifest.json",
        ]
        missing_artifacts = [name for name in required_root_files if not (OUTPUT_DIR / name).exists()]
        for item in run_records:
            run_dir = Path(item["run_dir"])
            for relative in [
                "report.json", "results.csv", "test_predictions.npz", "split_indices.npz",
                "evolution/selected_genome.json", "evolution/meta_split_indices.npz",
            ]:
                if not (run_dir / relative).exists():
                    missing_artifacts.append(str(run_dir / relative))
        if missing_artifacts:
            raise FileNotFoundError(f"Missing artifacts: {missing_artifacts}")
        print("Artifact audit: PASS")
        """
    ),
    markdown("## 28. Archive everything for download and later analysis"),
    code(
        """
        import shutil

        archive_base = WORK_ROOT / f"treestack-v4-{RUN_MODE}-all-datasets"
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=OUTPUT_DIR))
        print("Results archive:", archive_path)
        print(f"Size: {archive_path.stat().st_size / 2**20:.1f} MiB")
        """
    ),
    markdown(
        """
        ## 29. Optional browser download

        Kaggle: use the Files panel to download the ZIP or save the notebook version with outputs. Colab: the following cell opens a browser download.
        """
    ),
    code(
        """
        if PLATFORM == "colab":
            from google.colab import files

            files.download(str(archive_path))
        else:
            print("Download from the notebook Files panel:", archive_path)
        """
    ),
    markdown(
        """
        ## 30. How to interpret the final evidence

        The method is publication-ready only if its advantage repeats across both datasets and seeds, correction counts consistently exceed harms, paired tests support the effect, and strong baselines such as logistic and random-forest stacking do not explain the gain. Report null or negative results honestly. Do not modify V4 after reading these official test outcomes; any new V5 design requires a new untouched confirmation benchmark.
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"v4-{index:03d}"


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(TARGET)

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "treestack_v3_evolutionary_kaggle.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # TreeStack-CNN V3: elite weights and evolutionary fusion

        V3 tests two separate ideas without mixing incompatible tensors. Within each CNN, every epoch competes as a raw checkpoint, an exponential moving average (EMA), and a greedy checkpoint soup. Across the heterogeneous CNNs, a population evolves arithmetic, geometric/product, and confidence-selection fusion rules with conservative override gates.

        The official test set is never used for checkpoint or genome selection. Because Fashion-MNIST V2 test results have already been inspected, treat further Fashion-MNIST runs as exploratory and use an untouched dataset such as CIFAR-10 for confirmation.
        """
    ),
    markdown(
        """
        ## 1. Choose the pilot run

        Start with one seed. The full three-seed run should begin only after the training curves, meta-validation result, override rate, and correction-versus-harm audit look credible.
        """
    ),
    code(
        """
        REPOSITORY = "https://github.com/Prathmesh333/WeakEnsembleStrongDecisionTree.git"
        DATASET = "fashion_mnist"       # Confirm later on an untouched dataset.
        SEEDS = [42]                     # Publication run: [17, 42, 73]
        EPOCHS = 50                      # CIFAR-10: 180
        BATCH_SIZE = 128
        MAX_GPUS = 2
        NUM_WORKERS = 1

        EMA_DECAY = 0.98
        AVERAGING_START_EPOCH = 12       # CIFAR-10: about 45
        GENERATIONS = 30
        POPULATION_SIZE = 36
        ELITE_COUNT = 4
        META_VALIDATION_FRACTION = 0.20
        FORCE_RETRAIN = False

        print({
            "dataset": DATASET,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "fusion_population": POPULATION_SIZE,
            "fusion_generations": GENERATIONS,
        })
        """
    ),
    markdown(
        """
        ## 2. Fetch and activate the V3 source

        The repository uses a `src` layout. This cell activates that exact checkout for both the notebook kernel and the training subprocess, avoiding the `No module named treestack_cnn` failure from earlier notebooks.
        """
    ),
    code(
        """
        import importlib
        import os
        import subprocess
        import sys
        from pathlib import Path

        ON_KAGGLE = Path("/kaggle/working").exists()
        WORK_ROOT = Path("/kaggle/working" if ON_KAGGLE else "/content")
        REPO_DIR = WORK_ROOT / "WeakEnsembleStrongDecisionTree"

        if not (REPO_DIR / "pyproject.toml").exists():
            subprocess.run(["git", "clone", REPOSITORY, str(REPO_DIR)], check=True)
        else:
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)

        os.chdir(REPO_DIR)
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        SRC_DIR = (REPO_DIR / "src").resolve()
        runner_file = SRC_DIR / "treestack_cnn" / "v3_runner.py"
        assert runner_file.exists(), (
            f"Commit {commit} does not contain {runner_file.name}. "
            "Delete the checkout and rerun this cell."
        )

        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        pythonpath = [str(SRC_DIR)]
        pythonpath.extend(
            item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if item and item != str(SRC_DIR)
        )
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
        RUN_ENV = os.environ.copy()

        for module_name in list(sys.modules):
            if module_name == "treestack_cnn" or module_name.startswith("treestack_cnn."):
                del sys.modules[module_name]
        importlib.invalidate_caches()
        module = importlib.import_module("treestack_cnn.v3_runner")
        assert Path(module.__file__).resolve() == runner_file.resolve()
        subprocess.run(
            [sys.executable, "-c", "import treestack_cnn.v3_runner as m; print(m.__file__)"],
            cwd=REPO_DIR,
            env=RUN_ENV,
            check=True,
        )
        print(f"V3 source activated from commit {commit}: {runner_file}")
        """
    ),
    markdown(
        """
        ## 3. Inspect what is allowed to share weights

        Each architecture averages only checkpoints from its own trajectory. Direct averaging or multiplication between these three networks is invalid because their parameter names and shapes differ. Cross-model multiplication is implemented safely as a geometric mean of probabilities in log space.
        """
    ),
    code(
        """
        import pandas as pd
        from IPython.display import display

        from treestack_cnn.elite_training import EliteWeightConfig
        from treestack_cnn.evolution import EvolutionConfig, FUSION_OPERATORS
        from treestack_cnn.models import build_models
        from treestack_cnn.profiles import MODEL_DIVERSITY_PROFILES
        from treestack_cnn.utils import count_parameters

        in_channels = 1 if DATASET == "fashion_mnist" else 3
        models = build_models(in_channels, 10)
        rows = []
        for name, model in models.items():
            profile = MODEL_DIVERSITY_PROFILES[name]
            rows.append({
                "model": name,
                "parameters": count_parameters(model),
                "optimizer": profile["optimizer"],
                "learning_rate_multiplier": profile["learning_rate_multiplier"],
                "label_smoothing": profile["label_smoothing"],
                "checkpoint_candidates": "raw + EMA + greedy soup",
            })
        display(pd.DataFrame(rows))
        print("Fusion operators:", FUSION_OPERATORS)
        print("Elite config:", EliteWeightConfig(EMA_DECAY, AVERAGING_START_EPOCH))
        print("Evolution config:", EvolutionConfig(
            generations=GENERATIONS,
            population_size=POPULATION_SIZE,
            elite_count=ELITE_COUNT,
            meta_validation_fraction=META_VALIDATION_FRACTION,
        ))
        del models
        """
    ),
    markdown(
        """
        ## 4. Audit both leakage boundaries

        CNN fitting, CNN validation, fusion search, fusion validation, and final testing must remain distinct. The fusion-validation samples select the final genome but never update CNN parameters.
        """
    ),
    code(
        """
        import numpy as np

        from treestack_cnn.config import DatasetConfig
        from treestack_cnn.data import build_dataset
        from treestack_cnn.evolution import stratified_meta_split

        DATA_ROOT = WORK_ROOT / "treestack-v3-data"
        audit_config = DatasetConfig(name=DATASET, root=str(DATA_ROOT), download=True)
        audit_bundle = build_dataset(audit_config, SEEDS[0])
        base = set(audit_bundle.splits.base.tolist())
        meta = set(audit_bundle.splits.meta.tolist())
        test = set(audit_bundle.splits.test.tolist())
        assert not (base & meta or base & test or meta & test)

        official_training_targets = np.asarray(
            audit_bundle.meta.dataset.datasets[0].targets, dtype=np.int64
        )
        meta_labels = official_training_targets[audit_bundle.splits.meta]
        fusion_search, fusion_validation = stratified_meta_split(
            meta_labels, META_VALIDATION_FRACTION, SEEDS[0] + 7000
        )
        assert not (set(fusion_search) & set(fusion_validation))
        assert set(fusion_search) | set(fusion_validation) == set(range(len(meta_labels)))
        print({
            "cnn_train": len(audit_bundle.splits.base_train),
            "cnn_validation": len(audit_bundle.splits.base_validation),
            "fusion_search": len(fusion_search),
            "fusion_validation": len(fusion_validation),
            "official_test": len(audit_bundle.splits.test),
        })
        print("Leakage audit: PASS")
        del audit_bundle, meta_labels, base, meta, test
        """
    ),
    markdown("## 5. Verify the Kaggle accelerator"),
    code(
        """
        import torch

        subprocess.run(["nvidia-smi"], check=False)
        GPU_COUNT = torch.cuda.device_count()
        print(f"PyTorch {torch.__version__}; CUDA devices: {GPU_COUNT}")
        for index in range(GPU_COUNT):
            properties = torch.cuda.get_device_properties(index)
            print(f"cuda:{index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")
        assert GPU_COUNT > 0, "Enable a GPU accelerator and restart the session."
        print("Two models train concurrently when two T4 devices are available.")
        """
    ),
    markdown(
        """
        ## 6. Train elite CNN checkpoints and evolve fusion

        The V3 configuration hash differs from V2, so older checkpoints cannot be reused accidentally. Set `FORCE_RETRAIN=True` only when intentionally repeating the identical V3 configuration from scratch.
        """
    ),
    code(
        """
        OUTPUT_DIR = WORK_ROOT / "treestack-v3-results"
        command = [
            sys.executable, "-m", "treestack_cnn.v3_runner",
            "--dataset", DATASET,
            "--seeds", *map(str, SEEDS),
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--max-gpus", str(min(MAX_GPUS, GPU_COUNT)),
            "--num-workers", str(NUM_WORKERS),
            "--data-root", str(DATA_ROOT),
            "--output-dir", str(OUTPUT_DIR),
            "--ema-decay", str(EMA_DECAY),
            "--averaging-start-epoch", str(AVERAGING_START_EPOCH),
            "--generations", str(GENERATIONS),
            "--population-size", str(POPULATION_SIZE),
            "--elite-count", str(ELITE_COUNT),
            "--meta-validation-fraction", str(META_VALIDATION_FRACTION),
        ]
        if FORCE_RETRAIN:
            command.append("--force")
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=REPO_DIR, env=RUN_ENV, check=True)
        """
    ),
    markdown("## 7. Compare V3 with every baseline"),
    code(
        """
        aggregate = pd.read_csv(OUTPUT_DIR / "aggregate_results.csv")
        main_results = aggregate[aggregate["category"] == "main"].copy()
        main_results["accuracy_percent"] = 100 * main_results["accuracy_mean"]
        main_results["macro_f1_percent"] = 100 * main_results["macro_f1_mean"]
        display(main_results[[
            "method", "seeds", "accuracy_percent", "accuracy_std",
            "macro_f1_percent", "inference_ms_mean"
        ]].sort_values("accuracy_percent", ascending=False))
        display(pd.read_csv(OUTPUT_DIR / "paper_accuracy_table.csv"))
        """
    ),
    markdown("## 8. Load the most recently produced run"),
    code(
        """
        import json

        run_reports = list(OUTPUT_DIR.glob(f"{DATASET}/seed_*/**/report.json"))
        assert run_reports, "No report was produced. Inspect the training cell output."
        latest_report = max(run_reports, key=lambda path: path.stat().st_mtime)
        run_dir = latest_report.parent
        figure_dir = run_dir / "figures"
        report = json.loads(latest_report.read_text(encoding="utf-8"))
        print("Report:", latest_report)
        print("GPU assignments:", report["model_gpu_assignments"])
        print("Elite validation accuracies:", report["validation_accuracies"])
        """
    ),
    markdown(
        """
        ## 9. Inspect self-learning checkpoint evolution

        Raw validation accuracy shows the ordinary checkpoint. EMA and greedy soup are compatible weight combinations from the same model trajectory. The elite line records whichever candidate would be retained after each epoch.
        """
    ),
    code(
        """
        import matplotlib.pyplot as plt

        history_rows = []
        figure, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
        for axis, model_name in zip(axes, report["training_profiles"]):
            history = pd.read_csv(run_dir / "training" / f"{model_name}.csv")
            axis.plot(history["epoch"], history["validation_accuracy"], label="raw")
            axis.plot(history["epoch"], history["ema_validation_accuracy"], label="EMA")
            axis.plot(history["epoch"], history["soup_best_validation_accuracy"], label="greedy soup")
            axis.plot(history["epoch"], history["elite_validation_accuracy"], linewidth=2.5, label="retained elite")
            axis.set(title=model_name, xlabel="Epoch", ylabel="Validation accuracy")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            best = history.loc[history["elite_validation_accuracy"].idxmax()]
            history_rows.append({
                "model": model_name,
                "epochs_run": len(history),
                "best_epoch": int(best["epoch"]),
                "elite_kind": best["elite_kind"],
                "elite_validation_accuracy": best["elite_validation_accuracy"],
                "accepted_soup_checkpoints": int(best["soup_checkpoint_count"]),
            })
        plt.tight_layout()
        plt.show()
        display(pd.DataFrame(history_rows))
        """
    ),
    markdown("## 10. Inspect the evolutionary search trajectory"),
    code(
        """
        evolution_history = pd.read_csv(run_dir / "evolution" / "fusion_search_history.csv")
        display(evolution_history.tail(10))
        figure, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(evolution_history["generation"], evolution_history["accuracy"], label="search accuracy")
        axes[0].plot(evolution_history["generation"], evolution_history["fitness"], label="fitness")
        axes[1].plot(evolution_history["generation"], evolution_history["unique_correction_rate"], label="unique corrections")
        axes[1].plot(evolution_history["generation"], evolution_history["harm_rate"], label="harms")
        axes[1].plot(evolution_history["generation"], evolution_history["override_rate"], label="overrides", alpha=0.7)
        for axis in axes:
            axis.set(xlabel="Generation")
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0].set_ylabel("Score")
        axes[1].set_ylabel("Rate")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 11. Inspect the selected fusion genome

        A geometric operator is the requested probability multiplication, implemented as a weighted sum of log probabilities and normalized afterward. The gate threshold controls how much evidence is required before overriding ordinary soft voting.
        """
    ),
    code(
        """
        selected = json.loads((run_dir / "evolution" / "selected_genome.json").read_text())
        display(pd.DataFrame({
            "model": list(report["training_profiles"]),
            "weight": selected["genome"]["weights"],
            "temperature": selected["genome"]["temperatures"],
        }))
        print("Operator:", selected["genome"]["operator"])
        print("Gate threshold:", selected["genome"]["gate_threshold"])
        print("Requires base-model disagreement:", selected["genome"]["require_disagreement"])
        print("Search score:", selected["search_score"])
        print("Held-out meta-validation score:", selected["validation_score"])
        """
    ),
    markdown("## 12. Audit corrections and harms before looking at test accuracy"),
    code(
        """
        evolution_report = report["evolutionary_fusion"]
        correction_table = pd.DataFrame({
            "quantity": [
                "meta search samples", "meta validation samples",
                "test overrides", "soft-vote errors corrected",
                "correct soft-vote decisions harmed", "net corrections",
            ],
            "value": [
                evolution_report["meta_search_samples"],
                evolution_report["meta_validation_samples"],
                evolution_report["test_override_samples"],
                evolution_report["test_corrections_over_soft_vote"],
                evolution_report["test_harms_to_soft_vote"],
                evolution_report["test_net_corrections_over_soft_vote"],
            ],
        })
        display(correction_table)
        print("Meta-validation evolutionary accuracy:", evolution_report["validation_score"]["accuracy"])
        print("Meta-validation soft-vote accuracy:", evolution_report["meta_validation_soft_vote_accuracy"])
        """
    ),
    markdown("## 13. Confirm that diversity did not collapse"),
    code(
        """
        pairwise = report["pairwise_diversity_analysis"]
        display(pd.DataFrame.from_dict(pairwise["pairwise"], orient="index"))
        print("Oracle accuracy:", pairwise["oracle_accuracy_any_cnn_correct"])
        """
    ),
    markdown("## 14. Compare final confusion matrices"),
    code(
        """
        from IPython.display import Image, display

        for method, title in [
            ("soft_vote", "Soft Vote"),
            ("logistic_stack", "Logistic Stack"),
            ("dt_soft", "DT-Soft"),
            ("evolutionary_fusion", "Evolutionary Fusion V3"),
        ]:
            print(title)
            display(Image(filename=str(figure_dir / f"confusion_{method}.png"), width=650))
        """
    ),
    markdown(
        """
        ## 15. Publication-readiness checks

        These checks diagnose the experiment; they are not permission to tune against test labels. The most important pre-test evidence is whether V3 beats soft voting on the isolated meta-validation subset while harming fewer correct decisions than it repairs.
        """
    ),
    code(
        """
        metrics = report["metrics"]
        v3_accuracy = metrics["evolutionary_fusion"]["accuracy"]
        soft_accuracy = metrics["soft_vote"]["accuracy"]
        logistic_accuracy = metrics["logistic_stack"]["accuracy"]
        strongest_cnn = max(metrics[name]["accuracy"] for name in ("cnn_1", "cnn_2", "cnn_3"))
        minimum_cnn = min(metrics[name]["accuracy"] for name in ("cnn_1", "cnn_2", "cnn_3"))
        floor = 0.90 if DATASET == "fashion_mnist" else 0.75
        gates = [
            ("Every CNN clears the diagnostic floor", minimum_cnn >= floor, f"minimum={minimum_cnn:.4f}"),
            ("V3 beats Soft Vote on held-out meta-validation", evolution_report["validation_score"]["accuracy"] > evolution_report["meta_validation_soft_vote_accuracy"], f"delta={evolution_report['validation_score']['accuracy'] - evolution_report['meta_validation_soft_vote_accuracy']:+.4f}"),
            ("V3 corrections exceed harms on test", evolution_report["test_net_corrections_over_soft_vote"] > 0, f"net={evolution_report['test_net_corrections_over_soft_vote']:+d}"),
            ("V3 beats Soft Vote on test", v3_accuracy > soft_accuracy, f"delta={v3_accuracy - soft_accuracy:+.4f}"),
            ("V3 beats Logistic Stack", v3_accuracy > logistic_accuracy, f"delta={v3_accuracy - logistic_accuracy:+.4f}"),
            ("V3 beats the strongest CNN", v3_accuracy > strongest_cnn, f"delta={v3_accuracy - strongest_cnn:+.4f}"),
            ("Three independent seeds are complete", len(SEEDS) >= 3, f"seeds={len(SEEDS)}"),
        ]
        gate_table = pd.DataFrame(gates, columns=["gate", "passed", "evidence"])
        display(gate_table)
        print("Do not modify V3 from Fashion-MNIST test outcomes; confirm the frozen design elsewhere.")
        """
    ),
    markdown("## 16. Preserve all artifacts"),
    code(
        """
        import shutil

        archive_base = WORK_ROOT / f"treestack-v3-{DATASET}-results"
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=OUTPUT_DIR)
        print("Download or preserve:", archive_path)
        """
    ),
    markdown(
        """
        ## 17. Full experiment settings

        After the pilot succeeds, freeze the configuration and run `SEEDS = [17, 42, 73]`. For confirmatory evidence, run CIFAR-10 in a new notebook session with `EPOCHS = 180` and `AVERAGING_START_EPOCH = 45`. Report mean, standard deviation, paired significance tests, total GPU hours, selected operators, override rates, and correction-versus-harm counts.

        V3 is related to population-based training, stochastic weight averaging, model soups, and model alignment. Its paper claim must focus on the diversity-preserving fitness and conservative fusion gate rather than claiming that weight averaging itself is new.
        """
    ),
]

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
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(TARGET)

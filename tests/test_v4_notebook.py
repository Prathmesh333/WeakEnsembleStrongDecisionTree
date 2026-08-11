import json
from pathlib import Path

import numpy as np
import pandas as pd

from treestack_cnn.config import ExperimentConfig
from treestack_cnn.elite_training import EliteWeightConfig
from treestack_cnn.evolution import EvolutionConfig
from treestack_cnn.v4_runner import SUPPORTED_DATASETS, build_parser
from treestack_cnn.v4_runner import _append_evolutionary_fusion_v4
from treestack_cnn.utils import write_json


def test_v4_notebook_is_complete_and_compiles() -> None:
    path = Path("notebooks/treestack_v4_full_benchmark_kaggle.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) >= 50
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "treestack_cnn.v4_runner" in source
    assert 'DATASETS = ["fashion_mnist", "cifar10"]' in source
    assert '"seeds": [17, 42, 73]' in source
    assert "audit_partition_used_for_selection" in source
    assert "publication_summary.csv" in source
    assert "paired_tests.csv" in source
    assert "sys.path.insert(0, str(SRC_DIR))" in source
    assert "env=RUN_ENV" in source
    assert "sys.executable" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"v4-notebook-cell-{index}", "exec")


def test_v4_runner_defaults_cover_the_full_benchmark() -> None:
    args = build_parser().parse_args([])
    assert tuple(args.datasets) == SUPPORTED_DATASETS
    assert args.seeds == [17, 42, 73]
    assert args.fashion_epochs == 50
    assert args.cifar10_epochs == 180
    assert args.max_gpus == 2
    assert args.num_workers == 1
    assert args.generations == 30
    assert args.population_size == 36


def test_v4_append_preserves_audit_and_paired_predictions(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(14)
    labels = np.tile(np.arange(3), 40)
    probabilities = []
    for model_index in range(3):
        values = rng.uniform(0.01, 0.15, size=(len(labels), 3))
        values[np.arange(len(labels)), labels] += 0.65 - 0.03 * model_index
        values /= values.sum(axis=1, keepdims=True)
        probabilities.append(values.astype(np.float32))
    stacked = np.asarray(probabilities)
    summaries = []
    for position in range(3):
        cache_path = tmp_path / f"model_{position}.npz"
        np.savez_compressed(
            cache_path,
            meta_probabilities=stacked[position],
            meta_labels=labels,
            test_probabilities=stacked[position],
            test_labels=labels,
        )
        summaries.append(
            {
                "model_position": position,
                "cache_path": str(cache_path),
                "validation_accuracy": 0.92 - 0.01 * position,
                "parameter_count": 100 + position,
                "test_ms_per_sample": 0.1,
            }
        )

    pd.DataFrame(
        [
            {
                "dataset": "fashion_mnist",
                "seed": 42,
                "method": "Soft Vote",
                "method_key": "soft_vote",
                "category": "main",
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "inference_ms_per_sample": 0.3,
                "base_parameters": 303,
                "combiner_parameters": 0,
                "tree_depth": np.nan,
                "tree_leaves": np.nan,
            }
        ]
    ).to_csv(tmp_path / "results.csv", index=False)
    np.savez_compressed(tmp_path / "test_predictions.npz", labels=labels, soft_vote=labels)
    write_json({"metrics": {}, "training_profiles": {}}, tmp_path / "report.json")
    monkeypatch.setattr(
        "treestack_cnn.v4_runner.build_dataset",
        lambda config, seed: type("Bundle", (), {"class_names": ["a", "b", "c"]})(),
    )
    monkeypatch.setattr(
        "treestack_cnn.v4_runner.save_confusion_matrix", lambda *args, **kwargs: None
    )

    row = _append_evolutionary_fusion_v4(
        ExperimentConfig(),
        "fashion_mnist",
        42,
        tmp_path,
        summaries,
        EliteWeightConfig(averaging_start_epoch=2),
        EvolutionConfig(
            generations=2,
            population_size=8,
            elite_count=2,
            tournament_size=2,
        ),
    )

    assert row["method_key"] == "evolutionary_fusion_v4"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    audit = report["evolutionary_fusion_v4"]
    assert audit["selection_partition"] == "search"
    assert audit["audit_partition_used_for_selection"] is False
    assert audit["audit_genomes_evaluated"] == 1
    with np.load(tmp_path / "test_predictions.npz") as stored:
        assert "evolutionary_fusion_v4" in stored.files
        assert np.array_equal(stored["labels"], labels)

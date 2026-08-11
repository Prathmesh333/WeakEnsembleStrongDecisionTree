import numpy as np
import pandas as pd

from treestack_cnn.evolution import (
    EvolutionConfig,
    FusionGenome,
    evolve_fusion,
    fusion_probabilities,
    predict_genome,
)
from treestack_cnn.config import ExperimentConfig
from treestack_cnn.elite_training import EliteWeightConfig
from treestack_cnn.v3_runner import _append_evolutionary_fusion
from treestack_cnn.utils import write_json


def _synthetic_probabilities(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(3), 80)
    probabilities = []
    for model_index in range(3):
        values = rng.uniform(0.02, 0.12, size=(len(labels), 3))
        values[np.arange(len(labels)), labels] += 0.70 - 0.05 * model_index
        error_positions = np.arange(model_index * 5, len(labels), 19 + model_index)
        wrong = (labels[error_positions] + model_index + 1) % 3
        values[error_positions, labels[error_positions]] = 0.10
        values[error_positions, wrong] = 0.78
        values /= values.sum(axis=1, keepdims=True)
        probabilities.append(values)
    return np.asarray(probabilities, dtype=np.float32), labels.astype(np.int64)


def test_geometric_probability_fusion_is_normalized() -> None:
    probabilities, _ = _synthetic_probabilities()
    genome = FusionGenome(
        operator="geometric",
        weights=np.array([0.2, 0.3, 0.5]),
        temperatures=np.array([0.8, 1.0, 1.2]),
        gate_threshold=0.02,
        require_disagreement=True,
    )
    fused = fusion_probabilities(probabilities, genome)
    assert fused.shape == (240, 3)
    assert np.allclose(fused.sum(axis=1), 1.0, atol=1e-6)


def test_evolution_is_deterministic_and_preserves_meta_holdout() -> None:
    probabilities, labels = _synthetic_probabilities()
    config = EvolutionConfig(
        generations=3,
        population_size=8,
        elite_count=2,
        tournament_size=2,
        meta_validation_fraction=0.25,
    )
    first = evolve_fusion(probabilities, labels, np.array([0.9, 0.89, 0.88]), config, 42)
    second = evolve_fusion(probabilities, labels, np.array([0.9, 0.89, 0.88]), config, 42)

    assert first.genome.as_dict() == second.genome.as_dict()
    assert len(first.history) == 3
    assert not (set(first.search_indices) & set(first.validation_indices))
    assert set(first.search_indices) | set(first.validation_indices) == set(range(len(labels)))
    predictions, overrides = predict_genome(probabilities, first.genome)
    assert predictions.shape == labels.shape
    assert overrides.dtype == np.bool_
    assert 0.0 <= first.validation_score.accuracy <= 1.0
    assert first.search_candidate_count == 5 + config.generations * config.elite_count


def test_audit_labels_cannot_change_selected_genome(monkeypatch) -> None:
    probabilities, labels = _synthetic_probabilities()
    search_indices = np.arange(180)
    audit_indices = np.arange(180, 240)
    monkeypatch.setattr(
        "treestack_cnn.evolution.stratified_meta_split",
        lambda labels, validation_fraction, seed: (search_indices, audit_indices),
    )
    config = EvolutionConfig(
        generations=3,
        population_size=8,
        elite_count=2,
        tournament_size=2,
        meta_validation_fraction=0.25,
    )
    first = evolve_fusion(probabilities, labels, np.array([0.9, 0.89, 0.88]), config, 42)
    changed_labels = labels.copy()
    changed_labels[audit_indices] = (changed_labels[audit_indices] + 1) % 3
    second = evolve_fusion(
        probabilities, changed_labels, np.array([0.9, 0.89, 0.88]), config, 42
    )

    assert first.genome.as_dict() == second.genome.as_dict()
    assert first.search_score == second.search_score
    assert first.validation_score != second.validation_score


def test_v3_report_append_writes_reproducible_artifacts(tmp_path, monkeypatch) -> None:
    probabilities, labels = _synthetic_probabilities()
    test_probabilities = probabilities[:, :60]
    test_labels = labels[:60]
    summaries = []
    for position in range(3):
        cache_path = tmp_path / f"model_{position}.npz"
        np.savez_compressed(
            cache_path,
            meta_probabilities=probabilities[position],
            meta_labels=labels,
            test_probabilities=test_probabilities[position],
            test_labels=test_labels,
        )
        summaries.append(
            {
                "model_position": position,
                "cache_path": str(cache_path),
                "validation_accuracy": 0.90 - 0.01 * position,
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
                "accuracy": 0.9,
                "macro_f1": 0.9,
                "inference_ms_per_sample": 0.3,
            }
        ]
    ).to_csv(tmp_path / "results.csv", index=False)
    write_json({"metrics": {}, "training_profiles": {}}, tmp_path / "report.json")
    monkeypatch.setattr(
        "treestack_cnn.v3_runner.build_dataset",
        lambda config, seed: type("Bundle", (), {"class_names": ["a", "b", "c"]})(),
    )

    row = _append_evolutionary_fusion(
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

    assert row["method_key"] == "evolutionary_fusion"
    assert (tmp_path / "evolution" / "selected_genome.json").exists()
    assert (tmp_path / "evolution" / "fusion_search_history.csv").exists()
    report = __import__("json").loads((tmp_path / "report.json").read_text())
    assert "evolutionary_fusion" in report["metrics"]
    assert report["evolutionary_fusion"]["meta_validation_samples"] == 48

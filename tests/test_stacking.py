import numpy as np

from treestack_cnn.config import TreeConfig
from treestack_cnn.stacking import (
    enhanced_features,
    hard_features,
    majority_vote,
    pairwise_diversity_analysis,
    run_ensembles,
    soft_features,
)


def _random_probabilities(models: int, samples: int, classes: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.uniform(0.01, 1.0, size=(models, samples, classes))
    return values / values.sum(axis=2, keepdims=True)


def test_meta_feature_dimensions_and_hard_encoding() -> None:
    probabilities = _random_probabilities(3, 8, 4, seed=1)
    assert soft_features(probabilities).shape == (8, 12)
    hard = hard_features(probabilities)
    assert hard.shape == (8, 12)
    assert np.all(hard.reshape(8, 3, 4).sum(axis=2) == 1)
    assert enhanced_features(probabilities).shape == (8, 21)


def test_majority_vote_uses_mean_probability_for_three_way_tie() -> None:
    probabilities = np.asarray(
        [
            [[0.60, 0.25, 0.15]],
            [[0.10, 0.65, 0.25]],
            [[0.15, 0.20, 0.65]],
        ]
    )
    # Each model selects a different class. Class 1 has the largest mean score.
    assert majority_vote(probabilities).tolist() == [1]


def test_complete_ensemble_layer_runs_on_synthetic_predictions() -> None:
    meta_probabilities = _random_probabilities(3, 180, 3, seed=2)
    test_probabilities = _random_probabilities(3, 40, 3, seed=3)
    meta_labels = meta_probabilities.mean(axis=0).argmax(axis=1)
    config = TreeConfig(depths=[3, None], min_samples_leaf=[2], criteria=["gini"], cv_folds=2)
    result = run_ensembles(
        meta_probabilities,
        meta_labels,
        test_probabilities,
        config,
        seed=42,
        validation_accuracies=np.asarray([0.7, 0.8, 0.75]),
    )
    expected = {
        "majority_vote",
        "soft_vote",
        "weighted_soft_vote",
        "logistic_stack",
        "dt_hard",
        "dt_soft",
        "dt_enhanced",
        "dt_soft_best_two",
    }
    assert expected <= set(result.predictions)
    assert all(len(result.predictions[name]) == 40 for name in expected)
    assert set(result.depth_ablation) == {"3", "unrestricted"}
    assert result.meta_feature_dimensions == {"hard": 9, "soft": 9, "enhanced": 18}


def test_pairwise_diversity_reports_disagreement_and_oracle_accuracy() -> None:
    labels = np.asarray([0, 0, 1, 1])
    predictions = np.asarray(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
        ]
    )
    result = pairwise_diversity_analysis(predictions, labels)
    assert result["oracle_accuracy_any_cnn_correct"] == 1.0
    first_pair = result["pairwise"]["cnn_1_vs_cnn_2"]
    assert first_pair["prediction_disagreement_rate"] == 0.5
    assert first_pair["one_correct_one_wrong_rate"] == 0.5

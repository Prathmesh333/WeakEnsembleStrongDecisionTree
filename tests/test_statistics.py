import numpy as np
import pandas as pd

from treestack_cnn.statistics import exact_mcnemar, publication_summary


def test_exact_mcnemar_counts_corrections_and_harms() -> None:
    labels = np.array([0, 0, 0, 0, 0, 0])
    candidate = np.array([0, 0, 0, 1, 1, 0])
    reference = np.array([1, 1, 0, 0, 1, 0])

    result = exact_mcnemar(labels, candidate, reference)

    assert result["corrections"] == 2
    assert result["harms"] == 1
    assert result["net_corrections"] == 1
    assert result["discordant_pairs"] == 3
    assert 0.0 <= result["mcnemar_exact_p"] <= 1.0


def test_publication_summary_uses_sample_standard_deviation(tmp_path) -> None:
    rows = [
        {
            "dataset": "fashion_mnist",
            "seed": seed,
            "method": "Soft Vote",
            "method_key": "soft_vote",
            "category": "main",
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.01,
        }
        for seed, accuracy in zip([17, 42, 73], [0.90, 0.92, 0.94])
    ]

    path = publication_summary(rows, tmp_path)
    table = pd.read_csv(path)

    assert table.loc[0, "seeds"] == 3
    assert np.isclose(table.loc[0, "accuracy_mean"], 0.92)
    assert np.isclose(table.loc[0, "accuracy_std"], 0.02)
    assert table.loc[0, "accuracy_ci95_low"] <= 0.92
    assert table.loc[0, "accuracy_ci95_high"] >= 0.92

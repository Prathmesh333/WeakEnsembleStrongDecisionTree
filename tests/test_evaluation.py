from pathlib import Path

import pandas as pd

from treestack_cnn.experiment import aggregate_results


def test_aggregate_results_writes_mean_and_paper_table(tmp_path: Path) -> None:
    rows = [
        {
            "dataset": "fashion_mnist",
            "seed": seed,
            "run_hash": str(seed),
            "method": "DT-Soft",
            "method_key": "dt_soft",
            "category": "main",
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.01,
            "inference_ms_per_sample": 0.2,
        }
        for seed, accuracy in [(1, 0.90), (2, 0.92), (3, 0.91)]
    ]
    aggregate_path, paper_path = aggregate_results(rows, tmp_path)
    aggregate = pd.read_csv(aggregate_path)
    assert aggregate.loc[0, "accuracy_mean"] == 0.91
    assert aggregate.loc[0, "seeds"] == 3
    paper = pd.read_csv(paper_path)
    assert paper.loc[0, "DT-Soft"] == "91.00 ± 1.00"

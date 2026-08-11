import json
from pathlib import Path

from treestack_cnn.v3_runner import build_parser


def test_v3_notebook_is_complete_and_compiles() -> None:
    path = Path("notebooks/treestack_v3_evolutionary_kaggle.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) >= 30
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "treestack_cnn.v3_runner" in source
    assert "raw + EMA + greedy soup" in source
    assert "meta_validation_soft_vote_accuracy" in source
    assert "Evolutionary Fusion V3" in source
    assert "sys.path.insert(0, str(SRC_DIR))" in source
    assert "env=RUN_ENV" in source
    assert 'pip", "install", "-e"' not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"v3-notebook-cell-{index}", "exec")


def test_v3_runner_defaults_are_kaggle_safe() -> None:
    args = build_parser().parse_args([])
    assert args.dataset == "fashion_mnist"
    assert args.batch_size == 128
    assert args.max_gpus == 2
    assert args.num_workers == 1
    assert args.generations == 30
    assert args.population_size == 36
    assert args.elite_count == 4

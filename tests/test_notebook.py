import json
from pathlib import Path

from treestack_cnn.config import TrainingConfig
from treestack_cnn.cuda_runner import (
    MODEL_DIVERSITY_PROFILES,
    build_parser,
    model_training_config,
)


def test_kaggle_notebook_is_valid_and_code_cells_compile() -> None:
    path = Path("notebooks/treestack_kaggle_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) >= 30
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "import treestack_cnn.cuda_runner" in notebook_source
    assert "pairwise_diversity_analysis" in notebook_source
    assert "Publication-readiness checks" in notebook_source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_cuda_runner_defaults_target_two_gpus() -> None:
    args = build_parser().parse_args([])
    assert args.dataset == "fashion_mnist"
    assert args.max_gpus == 2
    assert args.num_workers == 1


def test_model_training_profiles_are_deliberately_different() -> None:
    base = TrainingConfig(learning_rate=2e-3)
    configurations = {
        name: model_training_config(base, name) for name in MODEL_DIVERSITY_PROFILES
    }
    assert configurations["cnn1_shallow"].optimizer == "adamw"
    assert configurations["cnn3_tiny_residual"].optimizer == "sgd"
    assert len({config.learning_rate for config in configurations.values()}) == 3
    assert len({config.label_smoothing for config in configurations.values()}) == 3

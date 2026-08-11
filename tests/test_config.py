from pathlib import Path

import pytest

from treestack_cnn.config import load_config


def test_default_config_is_valid() -> None:
    config = load_config()
    assert config.datasets == ["fashion_mnist", "cifar10"]
    assert config.dataset.use_official_test is True
    assert sum(
        [
            config.dataset.base_fraction,
            config.dataset.meta_fraction,
            config.dataset.test_fraction,
        ]
    ) == pytest.approx(1.0)


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("unknown_setting: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        load_config(path)


def test_unknown_scheduler_is_rejected() -> None:
    config = load_config()
    config.training.scheduler = "not-a-scheduler"
    with pytest.raises(ValueError, match="scheduler"):
        config.validate()


def test_unknown_optimizer_is_rejected() -> None:
    config = load_config()
    config.training.optimizer = "not-an-optimizer"
    with pytest.raises(ValueError, match="optimizer"):
        config.validate()

import numpy as np

from treestack_cnn.data import stratified_official_test_split, stratified_three_way_split


def test_stratified_split_is_disjoint_complete_and_deterministic() -> None:
    targets = np.repeat(np.arange(10), 100)
    first = stratified_three_way_split(targets, 0.6, 0.2, 0.2, seed=42)
    second = stratified_three_way_split(targets, 0.6, 0.2, 0.2, seed=42)

    assert np.array_equal(first.base, second.base)
    assert len(first.base) == 600
    assert len(first.meta) == 200
    assert len(first.test) == 200
    assert len(first.base_validation) == 60
    assert not (set(first.base) & set(first.meta))
    assert not (set(first.base) & set(first.test))
    assert not (set(first.meta) & set(first.test))
    assert set(first.base) | set(first.meta) | set(first.test) == set(range(1000))
    assert np.all(np.bincount(targets[first.meta], minlength=10) == 20)


def test_official_test_split_preserves_test_indices() -> None:
    training_targets = np.repeat(np.arange(10), 60)
    test_targets = np.repeat(np.arange(10), 10)
    splits = stratified_official_test_split(
        training_targets,
        test_targets,
        base_fraction=0.6,
        meta_fraction=0.2,
        seed=42,
    )

    assert len(splits.base) == 450
    assert len(splits.base_train) == 405
    assert len(splits.base_validation) == 45
    assert len(splits.meta) == 150
    assert np.array_equal(splits.test, np.arange(600, 700))
    assert not (set(splits.base) & set(splits.meta))

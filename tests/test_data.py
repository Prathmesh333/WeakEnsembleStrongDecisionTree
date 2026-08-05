import numpy as np

from treestack_cnn.data import stratified_three_way_split


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

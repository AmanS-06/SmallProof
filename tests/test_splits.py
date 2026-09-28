import random

import pytest

from eval.splits import _allocate, load_split, save_split, stratified_split


def make_items():
    """150 ids in three groups of 60, 60 and 30, like a small eval set."""
    ids = [f"q{i:03d}" for i in range(150)]
    groups = ["metrics"] * 60 + ["domain"] * 60 + ["novel"] * 30
    return ids, groups


def test_split_has_exact_size_and_no_overlap():
    ids, groups = make_items()
    dev, test = stratified_split(ids, groups, dev_size=50, seed=0)
    assert len(dev) == 50 and len(test) == 100
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == set(ids)


def test_split_keeps_group_proportions():
    ids, groups = make_items()
    group_of = dict(zip(ids, groups))
    dev, _ = stratified_split(ids, groups, dev_size=50, seed=0)
    counts = {group: sum(1 for item in dev if group_of[item] == group) for group in set(groups)}
    assert counts == {"metrics": 20, "domain": 20, "novel": 10}


def test_split_ignores_input_order_but_follows_seed():
    ids, groups = make_items()
    pairs = list(zip(ids, groups))
    random.Random(42).shuffle(pairs)
    shuffled_ids, shuffled_groups = zip(*pairs)
    first = stratified_split(ids, groups, 50, seed=7)
    assert stratified_split(list(shuffled_ids), list(shuffled_groups), 50, seed=7) == first
    assert stratified_split(ids, groups, 50, seed=8) != first


def test_allocate_uses_largest_remainder():
    assert _allocate(2, {"a": 1, "b": 1, "c": 1}) == {"a": 1, "b": 1, "c": 0}
    assert sum(_allocate(7, {"x": 5, "y": 3, "z": 2}).values()) == 7


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        stratified_split(["a", "a"], ["g", "g"], 1, seed=0)
    with pytest.raises(ValueError):
        stratified_split(["a"], ["g"], 2, seed=0)
    with pytest.raises(ValueError):
        stratified_split(["a"], [], 0, seed=0)
    assert stratified_split([], [], 0, seed=0) == ([], [])


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "splits" / "financebench.json"
    save_split(path, ["q1"], ["q2", "q3"], seed=0, note="test")
    assert load_split(path) == (["q1"], ["q2", "q3"])
    with pytest.raises(FileExistsError):
        save_split(path, ["q1"], ["q2"], seed=0)
    save_split(path, ["q2"], ["q1"], seed=1, overwrite=True)
    assert load_split(path) == (["q2"], ["q1"])

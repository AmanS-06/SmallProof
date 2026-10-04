"""Fixed, seeded dev and test splits of an eval set.

The split is stratified: each group (for example FinanceBench question type)
is split in the same proportion, so dev and test have a similar mix.
Save a split to JSON once and load it from then on, so every run in every
phase uses exactly the same questions. Tuning uses dev only; test is reported.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path


def _allocate(total: int, group_sizes: dict[str, int]) -> dict[str, int]:
    """Split total across groups in proportion to their sizes (largest remainder method).

    Each group first gets the whole-number part of its exact share. The few
    places left over go to the groups with the biggest fractional parts, so the
    shares always add up to total. Fractions avoid floating point rounding.
    """
    n_items = sum(group_sizes.values())
    exact = {group: Fraction(total * size, n_items) for group, size in group_sizes.items()}
    shares = {group: int(share) for group, share in exact.items()}  # int() floors positive fractions
    leftover = total - sum(shares.values())
    by_remainder = sorted(group_sizes, key=lambda group: (-(exact[group] - shares[group]), group))
    for group in by_remainder[:leftover]:
        shares[group] += 1
    return shares


def stratified_split(
    ids: Sequence[str],
    groups: Sequence[str],
    dev_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split ids into (dev_ids, test_ids), with exactly dev_size ids in dev.

    groups[i] is the group of ids[i]. The result depends only on the ids,
    groups and seed, not on the order they are passed in.
    """
    if len(ids) != len(groups):
        raise ValueError(f"Got {len(ids)} ids but {len(groups)} groups")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    if not 0 <= dev_size <= len(ids):
        raise ValueError(f"dev_size must be between 0 and {len(ids)}, got {dev_size}")
    if not ids:
        return [], []

    members: dict[str, list[str]] = {}
    for item_id, group in zip(ids, groups):
        members.setdefault(group, []).append(item_id)

    quotas = _allocate(dev_size, {group: len(items) for group, items in members.items()})
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    for group in sorted(members):
        # Sort first, so the shuffle always starts from the same order.
        shuffled = sorted(members[group])
        rng.shuffle(shuffled)
        dev.extend(shuffled[: quotas[group]])
        test.extend(shuffled[quotas[group] :])
    return sorted(dev), sorted(test)


def save_split(
    path: str | Path,
    dev_ids: Sequence[str],
    test_ids: Sequence[str],
    seed: int,
    note: str = "",
    overwrite: bool = False,
) -> None:
    """Write a split to JSON. Refuses to replace an existing split unless overwrite=True."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. A fixed split should not be replaced by accident.")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"seed": seed, "note": note, "dev": list(dev_ids), "test": list(test_ids)}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a split written by save_split and check that dev and test do not overlap."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dev, test = data["dev"], data["test"]
    overlap = set(dev) & set(test)
    if overlap:
        raise ValueError(f"Split file has {len(overlap)} ids in both dev and test")
    return dev, test

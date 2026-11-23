"""Tests for the split-v2 rule on a tiny synthetic group table."""

import pandas as pd
import pytest

from weapon_detection.ml.split_v2 import make_split_v2, negative_sources


def synthetic_groups() -> tuple[list[str], list[int]]:
    """3 big negative groups, 60 small negative groups, 1 too-big group, 200 positive groups."""
    groups, labels = [], []
    for name, size in [("bigA", 50), ("bigB", 40), ("bigC", 30), ("large23", 23)]:
        groups += [name] * size
        labels += [0] * size
    for i in range(60):  # 60 groups x 3 negatives = 180 negatives
        groups += [f"neg{i}"] * 3
        labels += [0] * 3
    for i in range(200):  # 200 groups x 5 positives = 1000 positives
        groups += [f"pos{i}"] * 5
        labels += [1] * 5
    return groups, labels


def test_split_v2_rule() -> None:
    groups, labels = synthetic_groups()
    splits = make_split_v2(groups, labels)
    table = pd.DataFrame({"group": groups, "y": labels, "split": splits})

    # Every image has exactly one split and no group is cut in two.
    assert len(splits) == len(groups)
    assert (table.groupby("group")["split"].nunique() == 1).all()
    split_of = table.groupby("group")["split"].first()

    # The 3 largest negative groups go to train; a group > 25% of 90 is never drawn.
    assert split_of[["bigA", "bigB", "bigC", "large23"]].tolist() == ["train"] * 4

    sources = negative_sources(groups, labels, splits)
    for split in ["val", "test"]:
        assert sources[split]["negatives"] >= 90
        assert sources[split]["distinct_negative_groups"] >= 8
        assert sources[split]["largest_negative_group_share"] <= 0.25
        # about 10% of the 1000 positives each, whole groups of 5 -> 100..104
        assert 100 <= sources[split]["positives"] < 105

    # Same seed -> same split.
    assert make_split_v2(groups, labels) == splits


def test_split_v2_stops_instead_of_relaxing() -> None:
    # Only 3 big negative groups + 5 small ones: test cannot reach 90 negatives from 8 groups.
    groups = ["a"] * 50 + ["b"] * 40 + ["c"] * 30
    labels = [0] * 120
    for i in range(5):
        groups += [f"n{i}"] * 2
        labels += [0, 0]
    for i in range(50):
        groups += [f"p{i}"] * 2
        labels += [1, 1]
    with pytest.raises(SystemExit, match="constraints not met"):
        make_split_v2(groups, labels)

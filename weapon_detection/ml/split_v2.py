"""Split v2: make sure val and test negatives come from many different sources.

Why: in split v1 every validation negative came from one video and almost every test
negative came from one banknote family. This rule was written down AFTER seeing the v1
test results and is applied once. Rule:

  1. A group is NEGATIVE if none of its images contains a firearm; otherwise POSITIVE.
  2. The 3 largest negative groups go to train (single-source blocks).
  3. Remaining negative groups are drawn at random (seed 42): first for test until it has
     >= 90 negative images, then for val until it has >= 90. A group bigger than 25% of 90
     is never drawn (it alone could break the 25% cap) and goes to train.
  4. Check: each split's negatives come from >= 8 groups and no group gives > 25% of them.
     If not, stop - the constraints are never relaxed.
  5. Positive groups are drawn at random (seed 42): test until it holds >= 10% of all
     positive images, then val, rest train. Whole groups only.
"""

import numpy as np
import pandas as pd

from weapon_detection.ml.common import SEED

N_LARGEST_NEGATIVE_TO_TRAIN = 3
MIN_SPLIT_NEGATIVES = 90
MIN_NEGATIVE_GROUPS = 8
MAX_GROUP_SHARE = 0.25
POSITIVE_FRACTION = 0.10


def group_table(groups: list[str], labels: list[int]) -> pd.DataFrame:
    """One row per group: number of images, positives, negatives, and the group type."""
    table = pd.DataFrame({"group": groups, "y": labels})
    summary = table.groupby("group").agg(images=("y", "size"), positives=("y", "sum"))
    summary["negatives"] = summary["images"] - summary["positives"]
    summary["is_negative_group"] = summary["positives"] == 0
    return summary


def negative_sources(groups: list[str], labels: list[int], splits: list[str]) -> dict:
    """Per split: negatives, how many groups they come from, and the biggest group's share."""
    table = pd.DataFrame({"group": groups, "y": labels, "split": splits})
    result = {}
    for split in ["train", "val", "test"]:
        negatives = table[(table["split"] == split) & (table["y"] == 0)]
        counts = negatives["group"].value_counts()
        part = table[table["split"] == split]
        result[split] = {
            "images": len(part),
            "positives": int(part["y"].sum()),
            "negatives": len(negatives),
            "distinct_negative_groups": int(counts.size),
            "largest_negative_group_share": round(float(counts.max() / len(negatives)), 4)
            if len(negatives)
            else 0.0,
        }
    return result


def make_split_v2(
    groups: list[str],
    labels: list[int],
    min_split_negatives: int = MIN_SPLIT_NEGATIVES,
    min_negative_groups: int = MIN_NEGATIVE_GROUPS,
) -> list[str]:
    """Return 'train' / 'val' / 'test' for every image, following the rule above."""
    table = group_table(groups, labels)
    split_of_group = {}

    # --- Negative groups --------------------------------------------------------------
    negative = table[table["is_negative_group"]].sort_values(
        ["images"], ascending=False, kind="stable"
    )
    for group in negative.index[:N_LARGEST_NEGATIVE_TO_TRAIN]:
        split_of_group[group] = "train"

    rng = np.random.default_rng(SEED)
    remaining = sorted(negative.index[N_LARGEST_NEGATIVE_TO_TRAIN:])  # sorted = reproducible
    order = [remaining[i] for i in rng.permutation(len(remaining))]
    max_drawable = int(MAX_GROUP_SHARE * min_split_negatives)  # 22 images for the defaults
    negatives_in = {"test": 0, "val": 0}
    for group in order:
        size = int(table.loc[group, "images"])
        target = next((s for s in ["test", "val"] if negatives_in[s] < min_split_negatives), None)
        if target is None or size > max_drawable:
            split_of_group[group] = "train"
        else:
            split_of_group[group] = target
            negatives_in[target] += size

    # --- Positive groups (including mixed groups) ---------------------------------------
    positive = sorted(table[~table["is_negative_group"]].index)
    order = [positive[i] for i in np.random.default_rng(SEED).permutation(len(positive))]
    target_positives = POSITIVE_FRACTION * int(table["positives"].sum())
    positives_in = {"test": 0, "val": 0}
    for group in order:
        target = next((s for s in ["test", "val"] if positives_in[s] < target_positives), None)
        if target is None:
            split_of_group[group] = "train"
        else:
            split_of_group[group] = target
            positives_in[target] += int(table.loc[group, "positives"])

    splits = [split_of_group[g] for g in groups]

    # --- Check the constraints; stop instead of relaxing them ---------------------------
    sources = negative_sources(groups, labels, splits)
    problems = []
    for split in ["val", "test"]:
        info = sources[split]
        if info["negatives"] < min_split_negatives:
            problems.append(f"{split}: only {info['negatives']} negatives")
        if info["distinct_negative_groups"] < min_negative_groups:
            problems.append(f"{split}: negatives from {info['distinct_negative_groups']} groups")
        if info["largest_negative_group_share"] > MAX_GROUP_SHARE:
            problems.append(f"{split}: one group gives {info['largest_negative_group_share']:.0%}")
    if problems:
        print(table.sort_values("negatives", ascending=False).head(30).to_string())
        raise SystemExit("Split v2 constraints not met: " + "; ".join(problems))
    return splits

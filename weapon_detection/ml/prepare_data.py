"""Step 1: extract the dataset, label each image, audit leakage and make a grouped split.

Run with:  uv run python -m weapon_detection.ml.prepare_data
No model is trained here.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from weapon_detection.ml.audit import (
    group_ids,
    image_info,
    near_duplicate_pairs,
    pairs_within_groups,
)
from weapon_detection.ml.common import (
    EXTRACT_DIR,
    FAMILY_MIN_CHARS,
    FAMILY_MIN_IMAGES,
    FIREARM_CLASS_NAMES,
    IMAGE_EXTENSIONS,
    ORIGINAL_SPLITS,
    OUTPUT_DIR,
    RAW_ZIP,
    SEED,
    box_area,
    family_keys,
    find_data_yaml,
    load_class_names,
    parse_label_line,
    relative_path,
    safe_extract,
    source_name,
)

MIN_NEGATIVES = 500  # course requirement: stop if we cannot build a fair negative class
MAX_HAMMING = 3  # dHash distance at or below which two images count as near duplicates
N_FOLDS = 10  # fold 0 = test, fold 1 = val, folds 2..9 = train
MIN_NEGATIVES_PER_SPLIT = 80  # report a split with fewer negatives than this


def print_table(headers: list[str], rows: list[list]) -> None:
    """Print a simple fixed-width text table."""
    widths = [max(len(str(x)) for x in column) for column in zip(headers, *rows)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    print()


def scan_dataset(dataset_root: Path, class_names: list[str]) -> tuple[list[dict], dict]:
    """Read every image path and its YOLO label file, split by split.

    Returns one record per image and a per-split report of counts.
    """
    records = []
    report = {}
    for split in ORIGINAL_SPLITS:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        if not image_dir.is_dir():
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        label_files = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        image_stems = {p.stem for p in images}
        counts = {
            "images": len(images),
            "label_files": len(label_files),
            "images_without_label_file": 0,
            "label_files_without_image": sum(p.stem not in image_stems for p in label_files),
            "empty_label_files": 0,
            "malformed_label_lines": 0,
            "corrupt_images": 0,  # filled in later, after decoding
            "objects_per_class": {name: 0 for name in class_names},
        }
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            objects = []
            if not label_path.exists():
                counts["images_without_label_file"] += 1
            else:
                lines = [line for line in label_path.read_text().splitlines() if line.strip()]
                if not lines:
                    counts["empty_label_files"] += 1
                for line in lines:
                    try:
                        class_id, box = parse_label_line(line)
                    except ValueError:
                        counts["malformed_label_lines"] += 1
                        continue
                    if not 0 <= class_id < len(class_names):
                        counts["malformed_label_lines"] += 1
                        continue
                    objects.append((class_names[class_id], box_area(box)))
                    counts["objects_per_class"][class_names[class_id]] += 1
            records.append(
                {
                    "path": image_path,
                    "original_split": split,
                    "source": source_name(image_path.name),
                    "classes": [name for name, _ in objects],
                    "box_areas": [area for _, area in objects],
                }
            )
        report[split] = counts
    return records, report


def count_crossing(pairs: list[tuple[int, int]], splits: list[str]) -> int:
    """How many pairs have their two images in different splits."""
    return sum(splits[a] != splits[b] for a, b in pairs)


def count_label_conflicts(pairs: list[tuple[int, int]], labels: list[int]) -> int:
    """How many pairs have different firearm labels (possible label noise)."""
    return sum(labels[a] != labels[b] for a, b in pairs)


def make_split(labels: list[int], groups: list[str]) -> list[str]:
    """StratifiedGroupKFold: fold 0 -> test, fold 1 -> val, the other folds -> train."""
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of_image = np.full(len(labels), -1)
    dummy_x = np.zeros(len(labels))  # the splitter only needs labels and groups
    for fold, (_, fold_indices) in enumerate(splitter.split(dummy_x, labels, groups)):
        fold_of_image[fold_indices] = fold
    names = {0: "test", 1: "val"}
    return [names.get(int(fold), "train") for fold in fold_of_image]


def check_no_group_overlap(groups: list[str], splits: list[str]) -> None:
    """Fail loudly if any group has images in two different splits."""
    split_of_group: dict[str, str] = {}
    for group, split in zip(groups, splits):
        if split_of_group.setdefault(group, split) != split:
            raise AssertionError(f"Group {group} appears in two splits")


def split_summary(
    groups: list[str],
    splits: list[str],
    labels: list[int],
    near: list[tuple[int, int]],
    families: list[str],
) -> dict:
    """Numbers that describe how well a split keeps related images apart."""
    sizes = pd.Series(groups).value_counts()
    table = pd.DataFrame({"group": groups, "split": splits, "family": families, "y": labels})
    family_rows = table[table["family"] != ""]
    split_counts = {}
    for split in ["train", "val", "test"]:
        part = table[table["split"] == split]
        split_counts[split] = {
            "images": len(part),
            "positives": int(part["y"].sum()),
            "negatives": int((part["y"] == 0).sum()),
            "groups": int(part["group"].nunique()),
        }
    return {
        "merged_groups": int(sizes.size),
        "groups_with_more_than_one_image": int((sizes > 1).sum()),
        "images_in_multi_image_groups": int(sizes[sizes > 1].sum()),
        "largest_group_size": int(sizes.max()),
        "groups_in_more_than_one_split": int((table.groupby("group")["split"].nunique() > 1).sum()),
        "near_duplicate_pairs_across_splits": count_crossing(near, splits),
        "families_spanning_splits": int(
            (family_rows.groupby("family")["split"].nunique() > 1).sum()
        ),
        "split": split_counts,
    }


def warn_large_groups(groups: list[str], families: list[str]) -> None:
    """Warn if one group holds more than 10% of all images, and name its families."""
    sizes = pd.Series(groups).value_counts()
    for group, size in sizes[sizes > 0.10 * len(groups)].items():
        names = sorted({f for g, f in zip(groups, families) if g == group and f})
        print(f"WARNING: group {group} holds {size} images (>10%); families: {names}")


def prepare(
    dataset_root: Path, output_dir: Path, min_negatives: int = MIN_NEGATIVES, n_jobs: int = -1
) -> tuple[pd.DataFrame, dict]:
    """Label, audit and split the dataset; write manifest.csv and audit.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Class names --------------------------------------------------------------
    class_names = load_class_names(find_data_yaml(dataset_root))
    print("Class names from data.yaml:")
    print_table(
        ["id", "name", "firearm?"],
        [[i, name, name.lower() in FIREARM_CLASS_NAMES] for i, name in enumerate(class_names)],
    )
    missing = FIREARM_CLASS_NAMES - {name.lower() for name in class_names}
    if missing:
        raise ValueError(f"FIREARM_CLASS_NAMES not in data.yaml: {sorted(missing)}")

    # --- Read labels and decode every image (in parallel) --------------------------
    records, split_report = scan_dataset(dataset_root, class_names)
    infos = Parallel(n_jobs=n_jobs)(delayed(image_info)(r["path"]) for r in records)
    for record, info in zip(records, infos):
        record.update(info)
        if not info["ok"]:
            split_report[record["original_split"]]["corrupt_images"] += 1
    records = [r for r in records if r["ok"]]  # corrupt images are dropped

    print("Per original split:")
    count_keys = [k for k in next(iter(split_report.values())) if k != "objects_per_class"]
    rows = [[split, *[c[k] for k in count_keys]] for split, c in split_report.items()]
    print_table(["split", *count_keys], rows)
    print("Objects per class per original split:")
    print_table(
        ["split", *class_names],
        [[s, *c["objects_per_class"].values()] for s, c in split_report.items()],
    )

    # --- Binary label ---------------------------------------------------------------
    labels = [int(any(c.lower() in FIREARM_CLASS_NAMES for c in r["classes"])) for r in records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Images used: {len(labels)}  positives: {n_pos}  negatives: {n_neg}\n")
    if n_neg < min_negatives:
        raise SystemExit(f"Only {n_neg} negatives (< {min_negatives}); not inventing labels.")

    # --- Leakage audit: which images must stay together? -----------------------------
    original_splits = [r["original_split"] for r in records]
    shas = [r["sha256"] for r in records]
    same_source = pairs_within_groups([r["source"] for r in records])
    exact = pairs_within_groups(shas)
    near = [
        (a, b)
        for a, b in near_duplicate_pairs([r["dhash"] for r in records], MAX_HAMMING)
        if shas[a] != shas[b]  # identical files are already counted as exact duplicates
    ]
    families = family_keys([r["path"].name for r in records])
    family_pairs = pairs_within_groups(families)

    family_counts = pd.Series([f for f in families if f]).value_counts()
    print(
        f"Filename families used ({len(family_counts)} families, "
        f"{int(family_counts.sum())} images):"
    )
    print_table(["family", "images"], [[f, int(n)] for f, n in family_counts.items()])

    # Grouping WITHOUT families (the old rule) is kept only for the before/after report.
    old_groups = group_ids(len(records), same_source + exact + near)
    old_splits = make_split(labels, old_groups)
    before = split_summary(old_groups, old_splits, labels, near, families)

    # Grouping WITH families: this is the split we use.
    groups = group_ids(len(records), same_source + exact + near + family_pairs)
    warn_large_groups(groups, families)

    # --- New split, decided before any modelling ------------------------------------
    new_splits = make_split(labels, groups)
    after = split_summary(groups, new_splits, labels, near, families)
    check_no_group_overlap(groups, new_splits)
    assert after["groups_in_more_than_one_split"] == 0
    assert after["near_duplicate_pairs_across_splits"] == 0
    assert after["families_spanning_splits"] == 0

    # --- Manifest -------------------------------------------------------------------
    manifest = pd.DataFrame(
        {
            "image_path": [relative_path(r["path"]) for r in records],
            "original_split": original_splits,
            "new_split": new_splits,
            "group": groups,
            "family": families,
            "firearm_present": labels,
            "classes": [";".join(r["classes"]) for r in records],
            "box_areas": [";".join(f"{a:.6f}" for a in r["box_areas"]) for r in records],
            "n_objects": [len(r["classes"]) for r in records],
            "width": [r["width"] for r in records],
            "height": [r["height"] for r in records],
        }
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    # Print the 10 largest groups and which families they contain.
    sizes = manifest.groupby("group").agg(
        images=("image_path", "size"),
        positives=("firearm_present", "sum"),
        families=("family", lambda f: ",".join(sorted(set(f) - {""}))),
    )
    print("10 largest groups after merging:")
    top = sizes.sort_values("images", ascending=False).head(10)
    print_table(["group", "images", "positives", "families"], [[g, *r] for g, r in top.iterrows()])

    # Check that stratification kept the class balance (A5).
    low_negative_splits = []
    print("Positive share per split:")
    for split, counts in after["split"].items():
        share = counts["positives"] / counts["images"]
        print(f"  {split}: {share:.1%} positives, {counts['negatives']} negatives")
        if counts["negatives"] < MIN_NEGATIVES_PER_SPLIT:
            low_negative_splits.append(split)
            print(f"  WARNING: {split} has fewer than {MIN_NEGATIVES_PER_SPLIT} negatives")
    print()

    negatives = manifest[manifest["firearm_present"] == 0]
    audit = {
        "class_names": class_names,
        "firearm_classes": sorted(n for n in class_names if n.lower() in FIREARM_CLASS_NAMES),
        "per_original_split": split_report,
        "images_used": len(records),
        "positives": n_pos,
        "negatives": n_neg,
        "hard_negatives_with_other_objects": int((negatives["n_objects"] > 0).sum()),
        "no_box_images": int((manifest["n_objects"] == 0).sum()),
        "same_source_pairs": len(same_source),
        "same_source_pairs_crossing_original_split": count_crossing(same_source, original_splits),
        "exact_duplicate_pairs": len(exact),
        "exact_duplicate_pairs_crossing_original_split": count_crossing(exact, original_splits),
        "exact_duplicate_label_conflicts": count_label_conflicts(exact, labels),
        "near_duplicate_max_hamming": MAX_HAMMING,
        "near_duplicate_pairs": len(near),
        "near_duplicate_pairs_crossing_original_split": count_crossing(near, original_splits),
        "near_duplicate_label_conflicts": count_label_conflicts(near, labels),
        "family_rule": f"prefix >= {FAMILY_MIN_CHARS} chars and >= {FAMILY_MIN_IMAGES} images",
        "families": {f: int(n) for f, n in family_counts.items()},
        "family_pairs_crossing_original_split": count_crossing(family_pairs, original_splits),
        "new_split_method": f"StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, seed={SEED})",
        "splits_with_fewer_than_80_negatives": low_negative_splits,
        "before_family_fix": before,
        "after_family_fix": after,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2))

    print("Leakage audit:")
    print_table(
        ["check", "value"],
        [[k, v] for k, v in audit.items() if isinstance(v, int) and not isinstance(v, bool)],
    )
    print("Before / after the filename-family fix:")
    keys = [k for k in before if k != "split"]
    print_table(["check", "before", "after"], [[k, before[k], after[k]] for k in keys])
    print("New split (after family fix):")
    print_table(
        ["split", "images", "positives", "negatives", "groups"],
        [[s, *c.values()] for s, c in after["split"].items()],
    )
    return manifest, audit


def main() -> None:
    """Check the ZIP, extract it safely once, then run prepare()."""
    with zipfile.ZipFile(RAW_ZIP) as archive:
        bad_member = archive.testzip()  # None means every CRC matched
    print(f"ZIP CRC check: {'OK' if bad_member is None else 'FAILED at ' + bad_member}")
    if bad_member is not None:
        raise SystemExit("ZIP is corrupt")
    marker = EXTRACT_DIR / ".extracted_ok"
    if not marker.exists():
        print(f"Extracting to {EXTRACT_DIR} ...")
        safe_extract(RAW_ZIP, EXTRACT_DIR)
        marker.write_text("ok\n")
    prepare(EXTRACT_DIR, OUTPUT_DIR)
    print(f"Wrote {OUTPUT_DIR / 'manifest.csv'} and {OUTPUT_DIR / 'audit.json'}")


if __name__ == "__main__":
    main()

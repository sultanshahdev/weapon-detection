"""Inspect local datasets and export reproducible, grouped YOLO splits."""

import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from xml.etree.ElementTree import ParseError

import yaml
from PIL import Image

from weapon_detection.data.annotations import IMAGE_SUFFIXES, Labels, read_labels, read_names
from weapon_detection.detection.model import normalize_label

SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class Sample:
    path: Path
    digest: str
    labels: Labels
    label_path: Path


def _key(path: Path, *, annotation: bool) -> Path:
    folder = {"labels", "annotations", "annotation"} if annotation else {"images", "jpegimages"}
    return Path(
        *(part for part in path.with_suffix("").parts[:-1] if part.lower() not in folder), path.stem
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_dataset(
    root: Path,
    firearm_labels: set[str],
    *,
    names_file: Path | None = None,
    voc_one_based: bool = False,
) -> tuple[list[Sample], dict]:
    if not root.is_dir():
        raise ValueError(f"Dataset directory is missing: {root}")
    paths = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
    images = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]
    annotations = defaultdict(list)
    for path in paths:
        suffix = path.suffix.lower()
        # Ignore conventional export metadata text files.
        if suffix == ".xml" or (
            suffix == ".txt" and not path.stem.lower().startswith(("readme", "license", "classes"))
        ):
            annotations[_key(path, annotation=True)].append(path)
    if names_file is None and (root / "data.yaml").is_file():
        names_file = root / "data.yaml"
    names = read_names(names_file) if names_file else {}
    image_keys = Counter(_key(path, annotation=False) for path in images)
    duplicates = defaultdict(list)
    classes = Counter()
    problems = {}
    samples = []
    for path in images:
        key = _key(path, annotation=False)
        candidates = annotations.get(key, [])
        try:
            with Image.open(root / path) as image:
                image.load()
                width, height = image.size
            digest = _sha256(root / path)
            duplicates[digest].append(path.as_posix())
            if image_keys[key] > 1 or len(candidates) > 1:
                raise ValueError("Ambiguous image or label match.")
            if not candidates:
                raise ValueError("Missing annotation.")
            labels = read_labels(
                root / candidates[0], width, height, names, voc_one_based=voc_one_based
            )
        except (OSError, ValueError, ParseError, Image.DecompressionBombError) as exc:
            problems[path.as_posix()] = str(exc)
            continue
        samples.append(Sample(path, digest, labels, candidates[0]))
        classes.update(name for name, _ in labels)
    firearm_count = sum(classes[label] for label in firearm_labels)
    report = {
        "images": len(images),
        "annotation_files": sum(map(len, annotations.values())),
        "formats": sorted(
            {path.suffix.lower() for items in annotations.values() for path in items}
        ),
        "valid_pairs": len(samples),
        "class_counts": dict(classes),
        "annotations": sum(classes.values()),
        "firearm_annotations": firearm_count,
        "non_firearm_annotations": sum(classes.values()) - firearm_count,
        "image_errors": problems,
        "orphan_annotations": [
            path.as_posix()
            for key, items in annotations.items()
            if key not in image_keys
            for path in items
        ],
        "duplicates": [files for files in duplicates.values() if len(files) > 1],
    }
    return samples, report


def split_samples(
    samples: list[Sample],
    groups: dict[str, str],
    ratios: tuple[float, float, float],
    seed: int,
    *,
    preserve_test: bool = False,
    selected_paths: set[Path] | None = None,
) -> dict[str, list[Sample]]:
    if len(ratios) != 3 or min(ratios) <= 0 or not math.isclose(sum(ratios), 1):
        raise ValueError("Provide three positive split ratios that sum to 1.")
    # Merge source groups and exact duplicates transitively.
    parents = list(range(len(samples)))
    owners = {}
    for index, sample in enumerate(samples):
        keys = [("hash", sample.digest)]
        if group := groups.get(sample.path.as_posix()):
            keys.append(("source", group))
        for key in keys:
            previous = owners.setdefault(key, index)
            parents[_root(parents, index)] = _root(parents, previous)
    components = defaultdict(list)
    for index, sample in enumerate(samples):
        components[_root(parents, index)].append(sample)

    shuffled = list(components.values())
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    splits: dict[str, list[Sample]] = {key: [] for key in SPLITS}
    free = []
    for component in shuffled:
        official = [_official_test(sample) for sample in component]
        retained = [s for s in component if selected_paths is None or s.path in selected_paths]
        if preserve_test and any(official):
            if not all(official):
                raise ValueError(
                    "Official test split shares a source or duplicate with another split."
                )
            splits["test"].extend(retained)
        elif retained:
            free.append(retained)
    if preserve_test and not splits["test"]:
        raise ValueError("No official test paths found.")
    size = len(samples) if selected_paths is None else len(selected_paths)
    targets = dict(zip(SPLITS, (size * ratio for ratio in ratios), strict=True))
    eligible = SPLITS[:2] if preserve_test else SPLITS
    for index, component in enumerate(free):
        empty = tuple(name for name in eligible if not splits[name])
        choices = empty if empty and len(free) - index <= len(empty) else eligible
        name = max(choices, key=lambda key: targets[key] - len(splits[key]))
        splits[name].extend(component)
    if any(not items for items in splits.values()):
        raise ValueError("Need enough independent source groups for nonempty train, val and test.")
    return splits


def _root(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _official_test(sample: Sample) -> bool:
    return "test" in (part.lower() for part in sample.path.parts[:-1])


def prepare_dataset(
    root: Path,
    output: Path,
    *,
    mapping: dict[str, str],
    negatives: set[str],
    groups: dict[str, str],
    source: dict[str, str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    names_file: Path | None = None,
    voc_one_based: bool = False,
    preserve_test: bool = False,
    overwrite: bool = False,
) -> dict:
    for key in ("name", "url", "version", "license", "rights_note"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"Provide source provenance field {key}.")
    if source.get("rights_reviewed") is not True:
        raise ValueError("Confirm rights review in the source metadata before preparing an export.")
    mapping = {normalize_label(key): normalize_label(value) for key, value in mapping.items()}
    targets = {key for key, value in mapping.items() if value == "firearm"}
    if not targets or set(mapping.values()) - {"firearm", "exclude"}:
        raise ValueError("Mapping must explicitly assign source labels to firearm or exclude.")
    if output.is_symlink():
        raise ValueError("Dataset output cannot be a symbolic link.")
    output = output.resolve()
    root = root.resolve()
    if output == root or output in root.parents or root in output.parents:
        raise ValueError("Dataset output must be outside the input directory.")
    if output.exists() and not (
        overwrite and output.is_dir() and (output / "preparation_report.json").is_file()
    ):
        raise ValueError("Output exists. Overwrite requires a previously prepared dataset.")
    samples, report = inspect_dataset(
        root, targets, names_file=names_file, voc_one_based=voc_one_based
    )
    if report["image_errors"] or report["orphan_annotations"]:
        raise ValueError(f"Resolve annotation issues before export: {json.dumps(report)}")
    unknown_groups = groups.keys() - {sample.path.as_posix() for sample in samples}
    if unknown_groups:
        raise ValueError(f"Source groups refer to unknown images: {sorted(unknown_groups)}")
    unknown = set(report["class_counts"]) - mapping.keys()
    if unknown:
        raise ValueError(f"Missing explicit mappings for: {sorted(unknown)}")
    selected = [
        sample
        for sample in samples
        if sample.path.as_posix() in negatives or any(name in targets for name, _ in sample.labels)
    ]
    invalid_negatives = negatives - {
        sample.path.as_posix()
        for sample in samples
        if not any(name in targets for name, _ in sample.labels)
    }
    if invalid_negatives:
        raise ValueError(f"Negative list includes missing or positive images: {invalid_negatives}")
    if not any(name in targets for sample in selected for name, _ in sample.labels):
        raise ValueError("Dataset has no mapped firearm annotations.")
    if preserve_test and any(_official_test(s) for s in samples if s not in selected):
        raise ValueError("Verify all official test negatives before preserving that split.")
    # Split before skipping unverified negatives so they cannot sever a duplicate/source link.
    splits = split_samples(
        samples,
        groups,
        ratios,
        seed,
        preserve_test=preserve_test,
        selected_paths={sample.path for sample in selected},
    )
    positive_counts = {
        split: sum(any(name in targets for name, _ in sample.labels) for sample in items)
        for split, items in splits.items()
    }
    if not positive_counts["train"]:
        raise ValueError("Training split has no firearm annotations. Review seed and grouping.")
    report.update(
        positive_counts=positive_counts,
        source=source,
        mapping=mapping,
        seed=seed,
        ratios=ratios,
        preserve_test=preserve_test,
        voc_one_based=voc_one_based,
        excluded_annotations=report["non_firearm_annotations"],
        skipped_unverified_negatives=len(samples) - len(selected),
        ungrouped_samples=sum(sample.path.as_posix() not in groups for sample in samples),
        split_counts={key: len(items) for key, items in splits.items()},
        splits={
            key: [
                {
                    "image": s.path.as_posix(),
                    "label": s.label_path.as_posix(),
                    "sha256": s.digest,
                    "group": groups.get(s.path.as_posix()),
                }
                for s in items
            ]
            for key, items in splits.items()
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        _write_export(stage, root, output, splits, targets, report)
        _publish_export(stage, output, overwrite)
    return report


def _publish_export(stage: Path, output: Path, overwrite: bool) -> None:
    if not output.exists():
        stage.rename(output)
        return
    if not overwrite or output.is_symlink() or not (output / "preparation_report.json").is_file():
        raise ValueError("Output already exists.")
    backup_dir = Path(mkdtemp(prefix=".dataset-backup-", dir=output.parent))
    backup = backup_dir / "previous"
    try:
        output.rename(backup)
        try:
            stage.rename(output)
        except OSError:
            try:
                backup.rename(output)
            except OSError as exc:
                raise OSError(f"Could not restore the previous export. Backup at {backup}") from exc
            raise
    except OSError:
        if not backup.exists():
            backup_dir.rmdir()
        raise
    else:
        shutil.rmtree(backup_dir)


def _write_export(
    stage: Path,
    root: Path,
    output: Path,
    splits: dict[str, list[Sample]],
    targets: set[str],
    report: dict,
) -> None:
    for split, items in splits.items():
        images = stage / "images" / split
        labels = stage / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index, sample in enumerate(items):
            stem = f"{index:07d}"
            shutil.copyfile(root / sample.path, images / (stem + sample.path.suffix.lower()))
            rows = [
                f"0 {' '.join(format(value, '.17g') for value in box)}"
                for name, box in sample.labels
                if name in targets
            ]
            (labels / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
            )
    config = {
        "path": str(output),
        **{key: f"images/{key}" for key in SPLITS},
        "names": {0: "firearm"},
    }
    (stage / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (stage / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

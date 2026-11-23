"""Tests for weapon_detection.ml using a tiny synthetic Roboflow-style dataset."""

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weapon_detection.ml.audit import dhash, group_ids
from weapon_detection.ml.common import parse_label_line, safe_extract, source_name
from weapon_detection.ml.features import HANDCRAFTED_NAMES, compute_features
from weapon_detection.ml.prepare_data import prepare

CLASS_NAMES = ["Card", "Gun", "Knife", "Pistol"]


def save_noise_image(path: Path, seed: int) -> None:
    """Random-noise image: different seeds give very different dHashes."""
    pixels = np.random.default_rng(seed).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def make_dataset(root: Path) -> None:
    """32 images in train/valid/test with box, polygon, empty and missing labels."""
    root.mkdir()
    (root / "data.yaml").write_text(f"nc: 4\nnames: {CLASS_NAMES}\n")
    for split in ["train", "valid", "test"]:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    def add(split: str, name: str, seed: int, label: str | None) -> None:
        save_noise_image(root / split / "images" / f"{name}.jpg", seed)
        if label is not None:
            (root / split / "labels" / f"{name}.txt").write_text(label)

    splits = ["train", "valid", "test"]
    for i in range(30):
        if i % 2 == 0:
            label = "1 0.5 0.5 0.2 0.2\n"  # Gun box -> positive
        elif i % 3 == 0:
            label = "2 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"  # Knife polygon -> hard negative
        else:
            label = ""  # empty label file -> negative with no boxes
        add(splits[i % 3], f"img{i:02d}_jpg.rf.{i:04x}", seed=i, label=label)
    # Two Roboflow copies of the same source, placed in different original splits.
    add("train", "shared_jpg.rf.aaaa", seed=100, label="3 0.5 0.5 0.4 0.4\n")
    add("test", "shared_jpg.rf.bbbb", seed=101, label="3 0.5 0.5 0.4 0.4\n")
    # One image with no label file at all.
    add("valid", "nolabel", seed=200, label=None)


def test_parse_box_and_polygon_lines() -> None:
    class_id, box = parse_label_line("5 0.5 0.5 0.2 0.4")
    assert class_id == 5
    assert box == pytest.approx([0.4, 0.3, 0.6, 0.7])
    class_id, box = parse_label_line("2 0.1 0.2 0.3 0.2 0.3 0.6")
    assert class_id == 2
    assert box == pytest.approx([0.1, 0.2, 0.3, 0.6])
    with pytest.raises(ValueError):
        parse_label_line("1 0.5 0.5")


def test_source_name_and_union_find() -> None:
    assert source_name("photo_jpg.rf.abc123.jpg") == "photo_jpg"
    assert source_name("plain.jpg") == "plain"
    # 0-1 and 1-2 are linked, so 0, 1, 2 must share a group; 3 stays alone.
    groups = group_ids(4, [(0, 1), (1, 2)])
    assert groups[0] == groups[1] == groups[2] != groups[3]


def test_dhash_is_64_bits_and_stable() -> None:
    gray = np.random.default_rng(0).integers(0, 256, size=(40, 40), dtype=np.uint8)
    assert dhash(gray) == dhash(gray.copy())
    assert 0 <= dhash(gray) < 2**64


def test_safe_extract_rejects_escaping_paths(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(zip_path, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_prepare_labels_groups_and_split(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    manifest, audit = prepare(root, tmp_path / "out", min_negatives=5, n_jobs=1)

    assert len(manifest) == 33
    assert audit["per_original_split"]["valid"]["images_without_label_file"] == 1
    assert sum(c["empty_label_files"] for c in audit["per_original_split"].values()) == 10
    assert audit["firearm_classes"] == ["Gun", "Pistol"]
    # Knife-only images are negatives (hard negatives), not positives.
    assert audit["hard_negatives_with_other_objects"] == 5
    assert audit["same_source_pairs"] == 1
    assert audit["same_source_pairs_crossing_original_split"] == 1

    # The two copies of "shared" must end up in the same group and the same new split.
    shared = manifest[manifest["image_path"].str.contains("shared_jpg")]
    assert shared["group"].nunique() == 1
    assert shared["new_split"].nunique() == 1

    # No group may appear in more than one new split.
    assert (manifest.groupby("group")["new_split"].nunique() == 1).all()
    assert set(manifest["new_split"]) == {"train", "val", "test"}
    assert (tmp_path / "out" / "manifest.csv").exists()
    assert (tmp_path / "out" / "audit.json").exists()


def test_prepare_stops_when_too_few_negatives(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    with pytest.raises(SystemExit, match="negatives"):
        prepare(root, tmp_path / "out", min_negatives=500, n_jobs=1)


def test_feature_shapes(tmp_path: Path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"{seed}.jpg"
        save_noise_image(path, seed)
        paths.append(path)
    handcrafted, hog_matrix = compute_features(paths, n_jobs=1)
    assert handcrafted.shape == (3, len(HANDCRAFTED_NAMES))
    assert len(HANDCRAFTED_NAMES) >= 6
    assert hog_matrix.shape == (3, 1764)
    assert np.isfinite(handcrafted).all()

from pathlib import Path

import pytest
import yaml
from PIL import Image

from weapon_detection.data.dataset import Sample, prepare_dataset, split_samples


def test_preparation_outputs_normalized_splits(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "data.yaml").write_text("names: [gun, rifle, knife]")
    for index in range(12):
        Image.new("RGB", (32, 32), color=(index, 20, 20)).save(root / f"{index}.png")
        row = "2 0.5 0.5 0.25 0.25\n"
        if index > 1:
            row += f"{index % 2} 0.5 0.5 0.25 0.25\n"
        (root / f"{index}.txt").write_text(row)
    output = tmp_path / "output"
    args = dict(
        mapping={"gun": "firearm", "rifle": "firearm", "knife": "exclude"},
        negatives={"0.png"},
        groups={},
        source={
            "name": "fixture",
            "url": "fixture",
            "version": "1",
            "license": "test",
            "rights_note": "Generated fixture",
            "rights_reviewed": True,
        },
    )
    report = prepare_dataset(root, output, **args)
    assert report["excluded_annotations"] == 12
    assert report["skipped_unverified_negatives"] == 1
    labels = [path.read_text() for path in (output / "labels").rglob("*.txt")]
    assert labels.count("") == 1
    assert labels.count("0 0.5 0.5 0.25 0.25\n") == 10
    assert yaml.safe_load((output / "data.yaml").read_text()) == {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "firearm"},
    }
    with pytest.raises(ValueError, match="Output exists"):
        prepare_dataset(root, output, **args)
    assert prepare_dataset(root, output, **args, overwrite=True) == report


def test_split_transitively_groups_exact_duplicates_and_sources() -> None:
    paths = ["a", "b", "c", "d", "test/e", "f", "g"]
    samples = [
        Sample(Path(path), "shared" if index < 2 else path, [], Path("unused"))
        for index, path in enumerate(paths)
    ]
    groups = {"b": "clip", "c": "clip"}
    kwargs = {"ratios": (0.7, 0.15, 0.15), "seed": 17, "preserve_test": True}
    splits = split_samples(samples, groups, **kwargs)
    assert splits == split_samples(samples, groups, **kwargs)
    owners = {sample.path.as_posix(): name for name, items in splits.items() for sample in items}
    assert owners["a"] == owners["b"] == owners["c"]
    assert [sample.path.as_posix() for sample in splits["test"]] == ["test/e"]
    with pytest.raises(ValueError, match="Official test split"):
        split_samples(samples, {**groups, "test/e": "clip"}, **kwargs)

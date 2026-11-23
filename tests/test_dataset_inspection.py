from pathlib import Path

import pytest
from PIL import Image

from weapon_detection.data.annotations import read_labels
from weapon_detection.data.dataset import inspect_dataset


def test_voc_and_yolo_conversion(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text(
        "<annotation><size><width>100</width><height>50</height></size>"
        "<object><name> Gun </name><bndbox><xmin>1</xmin><ymin>1</ymin>"
        "<xmax>100</xmax><ymax>50</ymax></bndbox></object></annotation>"
    )
    assert read_labels(xml, 100, 50, {}, voc_one_based=True) == [("gun", (0.5, 0.5, 1, 1))]
    with pytest.raises(ValueError, match="dimensions"):
        read_labels(xml, 100, 51, {})
    txt = tmp_path / "sample.txt"
    txt.write_text("9 0.5 0.5 0.2 0.4\n")
    assert read_labels(txt, 100, 50, {9: "rifle"}) == [("rifle", (0.5, 0.5, 0.2, 0.4))]


@pytest.mark.parametrize("line", ["0 0.1 0.1 1 1", "0 0.5 0.5 0 0.1", "0 nan 0.5 0.1 0.1", "bad"])
def test_invalid_yolo_boxes(tmp_path: Path, line: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match="line 1"):
        read_labels(path, 32, 32, {0: "gun"})


def test_audit_errors_duplicates_and_orphans(tmp_path: Path) -> None:
    for name in ("good", "missing", "malformed"):
        Image.new("RGB", (32, 32)).save(tmp_path / f"{name}.png")
    (tmp_path / "good.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.5 0.5 0.2 0.3")
    (tmp_path / "bad.png").write_bytes(b"invalid image")
    (tmp_path / "malformed.xml").write_text("<annotation>")
    (tmp_path / "orphan.txt").touch()
    (tmp_path / "data.yaml").write_text("names: [gun, knife]")
    _, report = inspect_dataset(tmp_path, {"gun"})
    assert report["firearm_annotations"] == report["non_firearm_annotations"] == 1
    assert set(report["image_errors"]) == {"missing.png", "bad.png", "malformed.png"}
    assert report["orphan_annotations"] == ["orphan.txt"]
    assert report["duplicates"] == [["good.png", "malformed.png", "missing.png"]]

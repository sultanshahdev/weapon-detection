"""Tests for the YOLO export and evaluation helpers (no model, no network)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from weapon_detection.ml import classical
from weapon_detection.ml.yolo_data import check_label_text, rewrite_label_text
from weapon_detection.ml.yolo_eval import firearm_class_ids, freeze_threshold, max_firearm_score

NAMES = ["Card", "Currency Notes", "Gun", "Knife", "Mobile", "Pistol"]


def test_label_rewrite_keeps_only_firearms_as_class_0() -> None:
    text = (
        "2 0.5 0.5 0.2 0.2\n"  # Gun -> kept
        "3 0.1 0.1 0.1 0.1\n"  # Knife -> dropped
        "5 0.2 0.2 0.4 0.2 0.4 0.6 0.2 0.6\n"  # Pistol polygon -> kept as a box
        "4 0.7 0.7 0.1 0.1\n"  # Mobile -> dropped
    )
    new = rewrite_label_text(text, NAMES)
    lines = new.splitlines()
    assert lines[0] == "0 0.500000 0.500000 0.200000 0.200000"
    assert lines[1] == "0 0.300000 0.400000 0.200000 0.400000"
    assert len(lines) == 2
    check_label_text(new)


def test_background_images_get_empty_label() -> None:
    assert rewrite_label_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n", NAMES) == ""  # card, notes
    assert rewrite_label_text("", NAMES) == ""


def test_check_label_text_rejects_bad_boxes() -> None:
    with pytest.raises(AssertionError):
        check_label_text("0 0.5 0.5 0 0.2")  # zero width
    with pytest.raises(AssertionError):
        check_label_text("1 0.5 0.5 0.2 0.2")  # wrong class


def fake_result(classes: list[int], confidences: list[float]) -> SimpleNamespace:
    """Looks like an Ultralytics result: result.boxes.cls and result.boxes.conf."""
    boxes = SimpleNamespace(cls=np.array(classes), conf=np.array(confidences))
    return SimpleNamespace(boxes=boxes)


def test_max_firearm_score() -> None:
    firearm_ids = [238, 423, 457]
    result = fake_result([238, 1, 457, 238], [0.4, 0.99, 0.7, 0.1])
    assert max_firearm_score(result, firearm_ids) == pytest.approx(0.7)  # 0.99 is not a firearm
    assert max_firearm_score(fake_result([1, 2], [0.9, 0.8]), firearm_ids) == 0.0
    assert max_firearm_score(fake_result([], []), firearm_ids) == 0.0


def test_firearm_class_ids_by_name() -> None:
    assert firearm_class_ids({0: "firearm"}) == [0]
    assert firearm_class_ids({0: "Person", 238: "Handgun", 423: "Rifle", 457: "Shotgun"}) == [
        238,
        423,
        457,
    ]
    with pytest.raises(ValueError):
        firearm_class_ids({0: "person", 1: "car"})


def test_threshold_is_frozen_from_validation_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(classical, "SETTINGS_PATH", tmp_path / "chosen_settings.json")
    y_val = np.array([0, 0, 1, 1])
    val_scores = np.array([0.1, 0.2, 0.8, 0.9])
    # freeze_threshold receives only validation labels and scores - no test data exists here.
    threshold = freeze_threshold("fake_yolo", y_val, val_scores, {"imgsz": 640})
    assert 0.2 < threshold <= 0.8
    saved = json.loads((tmp_path / "chosen_settings.json").read_text())
    assert saved["fake_yolo"]["threshold"] == threshold
    assert saved["fake_yolo"]["imgsz"] == 640

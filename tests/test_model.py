from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weapon_detection.detection import model
from weapon_detection.detection.model import YoloDetector


def test_missing_and_invalid_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="existing local"):
        YoloDetector(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="existing local"):
        YoloDetector(tmp_path)
    checkpoint = tmp_path / "invalid.pt"
    checkpoint.write_bytes(b"bad file")

    def broken_checkpoint(path):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(model, "load_yolo", broken_checkpoint)
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        YoloDetector(checkpoint)


def test_runtime_class_map_and_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "trusted.pt"
    path.touch()
    inner = SimpleNamespace(task="detect")
    fake = SimpleNamespace(model=inner, names={4: "  HaNdGuN ", 6: "Rifle", 7: "Water gun"})
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    boxes = SimpleNamespace(
        cls=np.array([4, 6, 6, 7]),
        conf=np.array([0.9, 0.8, 0.1, 0.9]),
        xyxy=np.array([[1, 2, 11, 12]] * 4),
    )

    def predict(**kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["classes"] == [4, 6]
        assert kwargs["source"] is frame
        return [SimpleNamespace(boxes=boxes)]

    fake.predict = predict
    monkeypatch.setattr(model, "load_yolo", lambda path: fake)
    detector = YoloDetector(path)
    results = detector.predict(frame)
    assert [d.class_id for d in results] == [4, 6]
    assert (results[0].x1, results[0].y2, results[0].normalized_class) == (1, 12, "firearm")
    boxes.cls, boxes.conf, boxes.xyxy = np.array([]), np.array([]), np.zeros((0, 4))
    assert detector.predict(frame) == ()
    fake.names = {0: "water gun", 1: "nail gun"}
    with pytest.raises(ValueError, match="no configured firearm labels"):
        YoloDetector(path)


def test_native_map_is_held_out_and_requires_compatible_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trusted.pt"
    path.touch()

    def validate(**args):
        assert args["split"] == "test" and args["device"] == "cpu"
        return SimpleNamespace(box=SimpleNamespace(map50=0.5, map=0.25))

    fake = SimpleNamespace(model=SimpleNamespace(task="detect"), names={0: "firearm"}, val=validate)
    monkeypatch.setattr(model, "load_yolo", lambda path: fake)
    detector = YoloDetector(path)
    assert detector.native_map(tmp_path / "data.yaml", tmp_path)["map50"] == 0.5
    fake.names = {238: "Handgun", 423: "Rifle"}
    with pytest.raises(ValueError, match="ID 0"):
        YoloDetector(path).native_map(tmp_path / "data.yaml", tmp_path)


def fake_checkpoint(tmp_path, monkeypatch, names):
    """A checkpoint file plus a mocked YOLO object with the given class names."""
    path = tmp_path / "model.pt"
    path.touch()
    fake = SimpleNamespace(model=SimpleNamespace(task="detect"), names=names)
    monkeypatch.setattr(model, "load_yolo", lambda path: fake)
    return path, fake


def test_finetuned_single_class_checkpoint_is_accepted(tmp_path, monkeypatch):
    path, _ = fake_checkpoint(tmp_path, monkeypatch, {0: "firearm"})
    assert YoloDetector(path).firearm_classes == {0: "firearm"}


def test_open_images_checkpoint_maps_only_firearm_ids(tmp_path, monkeypatch):
    # Real OIV7 ids for the three firearm classes plus two unrelated classes.
    names = {0: "Accordion", 238: "Handgun", 423: "Rifle", 457: "Shotgun", 600: "Zucchini"}
    path, _ = fake_checkpoint(tmp_path, monkeypatch, names)
    assert YoloDetector(path).firearm_classes == {238: "Handgun", 423: "Rifle", 457: "Shotgun"}


def test_coco_checkpoint_is_rejected(tmp_path, monkeypatch):
    path, _ = fake_checkpoint(tmp_path, monkeypatch, {0: "person", 1: "bicycle", 43: "knife"})
    with pytest.raises(ValueError, match="no configured firearm labels"):
        YoloDetector(path)


def test_predict_batch_matches_predict(tmp_path, monkeypatch):
    path, fake = fake_checkpoint(tmp_path, monkeypatch, {0: "firearm"})
    boxes = SimpleNamespace(
        cls=np.array([0]), conf=np.array([0.9]), xyxy=np.array([[1.0, 2.0, 11.0, 12.0]])
    )
    fake.predict = lambda **kwargs: [SimpleNamespace(boxes=boxes)] * (
        len(kwargs["source"]) if isinstance(kwargs["source"], list) else 1
    )
    detector = YoloDetector(path)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert detector.predict_batch([frame, frame]) == [detector.predict(frame)] * 2


def test_default_checkpoint_prefers_finetuned(tmp_path):
    assert model.default_checkpoint(tmp_path) is None
    (tmp_path / "models").mkdir()
    (tmp_path / model.PRETRAINED_CHECKPOINT).touch()
    assert model.default_checkpoint(tmp_path) == tmp_path / model.PRETRAINED_CHECKPOINT
    (tmp_path / model.FINETUNED_CHECKPOINT).touch()
    assert model.default_checkpoint(tmp_path) == tmp_path / model.FINETUNED_CHECKPOINT

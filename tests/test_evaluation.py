from pathlib import Path

import pytest
from PIL import Image

from weapon_detection.detection.evaluation import (
    evaluate_dataset,
    intersection_over_union,
    match_image,
)
from weapon_detection.detection.model import Detection


def test_iou_and_one_to_one_matching() -> None:
    assert intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    predictions = (
        Detection(2, "Gun", 0.9, 0, 0, 10, 10),
        Detection(2, "Gun", 0.8, 0, 0, 10, 10),
    )
    counts = match_image(predictions, [(0, 0, 10, 10), (20, 20, 30, 30)], 0.5)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    report = counts.report()
    assert report["precision"] == report["recall"] == report["f1"] == 0.5
    negatives = match_image(predictions, [], 0.5).report()
    assert negatives["false_detections_on_negatives"] == 2
    assert negatives["negative_images_with_detections"] == 1
    assert negatives["recall"] is None
    assert match_image((), [], 0.5).report()["f1"] is None


def test_held_out_and_missing_ground_truth(tmp_path: Path) -> None:
    class Detector:
        def predict(self, image):
            return (Detection(2, "Gun", 0.9, 8, 8, 24, 24),)

    (tmp_path / "data.yaml").write_text("names: [firearm]")
    for group in ("images", "labels"):
        (tmp_path / group / "test").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(tmp_path / "images/test/demo.png")
    with pytest.raises(ValueError, match="Missing annotation"):
        evaluate_dataset(tmp_path, Detector())
    (tmp_path / "labels/test/demo.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    result = evaluate_dataset(tmp_path, Detector())
    assert result["true_positives"] == 1
    assert result["map50"] is None

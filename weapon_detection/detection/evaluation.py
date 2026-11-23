"""Evaluate normalized firearm predictions at a fixed operating point."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from weapon_detection.data.annotations import Box, read_names, yolo_to_xyxy
from weapon_detection.data.dataset import SPLITS, inspect_dataset
from weapon_detection.detection.model import Detection, Detector


def intersection_over_union(first: Box, second: Box) -> float:
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    negative_images: int = 0
    negative_images_with_detections: int = 0
    false_detections_on_negatives: int = 0

    def report(self) -> dict:
        return {
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "precision": _ratio(self.tp, self.tp + self.fp),
            "recall": _ratio(self.tp, self.tp + self.fn),
            "f1": _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "negative_images": self.negative_images,
            "negative_images_with_detections": self.negative_images_with_detections,
            "false_detections_on_negatives": self.false_detections_on_negatives,
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def match_image(
    predictions: tuple[Detection, ...], ground_truth: list[Box], threshold: float
) -> Counts:
    if not 0 < threshold <= 1:
        raise ValueError("Match IoU threshold must be in (0, 1].")
    unmatched = dict(enumerate(ground_truth))
    tp = 0
    for prediction in sorted(predictions, key=lambda detection: -detection.confidence):
        predicted_box = (prediction.x1, prediction.y1, prediction.x2, prediction.y2)
        best_id, overlap = max(
            ((key, intersection_over_union(predicted_box, box)) for key, box in unmatched.items()),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        if overlap >= threshold:
            tp += 1
            del unmatched[best_id]
    negative = not ground_truth
    return Counts(
        tp=tp,
        fp=len(predictions) - tp,
        fn=len(unmatched),
        negative_images=int(negative),
        negative_images_with_detections=int(negative and bool(predictions)),
        false_detections_on_negatives=len(predictions) if negative else 0,
    )


def evaluate_dataset(root: Path, detector: Detector, *, threshold: float = 0.5) -> dict:
    """Evaluate an export's test split. Empty labels represent reviewed negatives."""
    # Inspect the complete export to detect exact duplicates spanning train and test.
    if not (root / "data.yaml").is_file():
        raise ValueError("Expected data.yaml in a prepared dataset directory.")
    if read_names(root / "data.yaml") != {0: "firearm"}:
        raise ValueError("Expected the normalized single firearm class.")
    samples, inspection = inspect_dataset(root, {"firearm"})
    if inspection["image_errors"] or inspection["orphan_annotations"]:
        raise ValueError(f"Invalid ground truth: {inspection}")
    for duplicate in inspection["duplicates"]:
        if len({Path(path).parts[:2] for path in duplicate}) > 1:
            raise ValueError(f"Duplicate images cross dataset splits: {duplicate}")
    test = [sample for sample in samples if sample.path.parts[:2] == ("images", "test")]
    if not test:
        raise ValueError("No labeled held-out test images found under images/test.")
    counts = Counts()
    review = []
    for sample in test:
        image = cv2.imread(
            str(root / sample.path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if image is None:
            raise ValueError(f"Could not decode evaluation image {sample.path}")
        height, width = image.shape[:2]
        truth = [yolo_to_xyxy(box, width, height) for _, box in sample.labels]
        result = match_image(detector.predict(image), truth, threshold)
        counts.tp += result.tp
        counts.fp += result.fp
        counts.fn += result.fn
        counts.negative_images += result.negative_images
        counts.negative_images_with_detections += result.negative_images_with_detections
        counts.false_detections_on_negatives += result.false_detections_on_negatives
        if result.fp or result.fn:
            review.append({"image": sample.path.as_posix(), "fp": result.fp, "fn": result.fn})
    return {
        "test_images": len(test),
        "match_iou": threshold,
        **counts.report(),
        "review": review,
        "map50": None,
        "map50_95": None,
        "map_status": "Unavailable. This evaluator measures a fixed confidence operating point.",
    }


def validate_native_map_config(root: Path) -> Path:
    """Use only a local generated YAML with no redirects or download directives."""
    root = root.resolve()
    data = root / "data.yaml"
    document = yaml.safe_load(data.read_text(encoding="utf-8"))
    expected = {
        "path": str(root),
        **{key: f"images/{key}" for key in SPLITS},
        "names": {0: "firearm"},
    }
    if document != expected or any(
        not (root / folder / split).is_dir() for folder in ("images", "labels") for split in SPLITS
    ):
        raise ValueError("Native mAP requires the generated local data.yaml and split layout.")
    return data

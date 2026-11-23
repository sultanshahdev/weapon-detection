"""Map detections from a trusted local YOLO checkpoint to the firearm class."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

FIREARM_ALIASES = ("firearm", "gun", "handgun", "pistol", "revolver", "rifle", "shotgun")

# Default checkpoint rule shared by every app: the fine-tuned single-class model if it
# exists, otherwise the pretrained Open Images V7 model. Both are checked at load time by
# YoloDetector (class names must include a firearm alias, otherwise it is rejected).
FINETUNED_CHECKPOINT = Path("models/firearm_yolov8n_best.pt")
PRETRAINED_CHECKPOINT = Path("models/yolov8n-oiv7.pt")


def default_checkpoint(root: Path) -> Path | None:
    """First existing checkpoint under root: fine-tuned, then OIV7; None if neither."""
    for relative in (FINETUNED_CHECKPOINT, PRETRAINED_CHECKPOINT):
        if (root / relative).is_file():
            return root / relative
    return None


def normalize_label(label: str) -> str:
    return " ".join(label.casefold().split())


@dataclass(frozen=True, slots=True)
class Detection:
    """Original checkpoint label and XYXY pixel coordinates in the input image."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    normalized_class: str = "firearm"


class Detector(Protocol):
    def predict(self, frame: np.ndarray) -> tuple[Detection, ...]: ...


def load_yolo(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path), task="detect")


class YoloDetector:
    """Load only an explicitly supplied local .pt file and run on CPU."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cpu",
        confidence: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 640,
        aliases: Iterable[str] = FIREARM_ALIASES,
    ) -> None:
        self.checkpoint = checkpoint.expanduser().resolve()
        if self.checkpoint.suffix.lower() != ".pt" or not self.checkpoint.is_file():
            raise ValueError(f"Expected an existing local .pt checkpoint: {self.checkpoint}")
        if device != "cpu":
            raise ValueError("This inference adapter supports cpu only.")
        if not (0 <= confidence <= 1 and 0 < iou <= 1 and imgsz > 0):
            raise ValueError(
                "Confidence must be in [0, 1], IoU in (0, 1], and image size positive."
            )
        self.device = device
        allowed = {normalize_label(label) for label in aliases}
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        # Resolve and inspect the explicitly selected model before opening any input video.
        try:
            self._model = load_yolo(self.checkpoint)
            if self._model.model.task != "detect":
                raise ValueError("Checkpoint task must be object detection.")
            names = self._model.names
            entries = names.items() if isinstance(names, Mapping) else enumerate(names)
            self.class_names = {int(key): str(name) for key, name in entries}
        except Exception as exc:
            raise ValueError(f"Could not read detector {self.checkpoint}: {exc}") from exc
        self.firearm_classes = {
            key: name for key, name in self.class_names.items() if normalize_label(name) in allowed
        }
        if not self.firearm_classes:
            raise ValueError(
                "Checkpoint has no configured firearm labels. Inspect its class names."
            )

    @staticmethod
    def _check_frame(frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size or frame.dtype != np.uint8:
            raise ValueError("Expected a nonempty uint8 BGR image.")

    def _boxes(self, source, batch: int = 1) -> list:
        """Run YOLO and return the boxes object of each result."""
        try:
            results = self._model.predict(
                source=source,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                classes=list(self.firearm_classes),
                batch=batch,
                verbose=False,
            )
            return [result.boxes for result in results]
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

    def predict(self, frame: np.ndarray) -> tuple[Detection, ...]:
        self._check_frame(frame)
        return self._to_detections(self._boxes(frame)[0], frame.shape[:2])

    def predict_batch(self, frames: list[np.ndarray]) -> list[tuple[Detection, ...]]:
        """Same as predict, for several frames in one model call (one result per frame)."""
        for frame in frames:
            self._check_frame(frame)
        boxes = self._boxes(list(frames), batch=len(frames))
        return [
            self._to_detections(item, frame.shape[:2])
            for item, frame in zip(boxes, frames, strict=True)
        ]

    def _to_detections(self, boxes, shape: tuple[int, int]) -> tuple[Detection, ...]:
        """Keep configured firearm classes, clip to the image and drop empty boxes."""
        height, width = shape
        detections = []
        if boxes is None:
            return ()
        for class_id, confidence, coords in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
        ):
            key = int(class_id)
            if key not in self.firearm_classes or not self.confidence <= confidence <= 1:
                continue
            if not all(math.isfinite(value) for value in coords):
                raise RuntimeError("YOLO returned a nonfinite bounding box.")
            x1, y1, x2, y2 = coords
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 > x1 and y2 > y1:
                detections.append(
                    Detection(key, self.firearm_classes[key], confidence, x1, y1, x2, y2)
                )
        return tuple(detections)

    def native_map(self, dataset: Path, output_dir: Path) -> dict[str, float]:
        """Run Ultralytics validation after the caller checks the dataset YAML."""
        if self.firearm_classes.keys() != {0} or self.class_names.keys() != {0}:
            raise ValueError("Native mAP needs a single firearm class at checkpoint ID 0.")
        metrics = self._model.val(
            data=str(dataset),
            split="test",
            device=self.device,
            imgsz=self.imgsz,
            conf=0.001,
            iou=self.iou,
            plots=False,
            workers=0,
            verbose=False,
            project=str(output_dir.resolve()),
            name="ultralytics_validation",
        )
        return {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}

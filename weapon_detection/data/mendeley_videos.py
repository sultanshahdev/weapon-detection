"""Read the Mendeley firearm-action videos and their COCO-style box labels.

Dataset: "Action recognition and object detection dataset for firearm-related actions",
Ruiz-Santaquiteria et al., Mendeley Data, DOI 10.17632/bbzpxhd22j.2.
It is a DEMO SOURCE ONLY: never use it for training, validation, thresholds or metrics.

Layout found on disk (checked against the real download):
    Gun_Action_Recognition_Dataset/<Handgun|Machine_Gun|No_Gun>/<video folder>/
        video.mp4
        label.json   (Handgun and Machine_Gun only; No_Gun has no label file)
One Handgun folder (PCH5_C1_P1_V1_HB_1) ships "label_antiguo.json" instead of
label.json; we read label.json only, so that video is treated as unlabelled.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from weapon_detection.utils.config import PROJECT_ROOT

TOP_FOLDER = "Gun_Action_Recognition_Dataset"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "interim" / "mendeley_firearm_actions" / TOP_FOLDER

# One ground-truth box: x1, y1, x2, y2 in pixels plus the dataset category name.
GtBox = tuple[float, float, float, float, str]


@dataclass(frozen=True)
class VideoRecord:
    class_name: str  # class folder, e.g. "Handgun"
    name: str  # video folder name, e.g. "PAH1_C1_P1_V1_HB_3"
    video_path: Path
    label_path: Path | None  # None when the folder has no label.json


def find_root(root: Path) -> Path:
    """Accept either the Gun_Action_Recognition_Dataset folder or its parent."""
    return root / TOP_FOLDER if (root / TOP_FOLDER).is_dir() else root


def list_videos(root: Path) -> list[VideoRecord]:
    """Every <class>/<video>/video.mp4 under root, sorted by class then video name."""
    root = find_root(root)
    records = []
    for video in sorted(root.glob("*/*/video.mp4")):
        folder = video.parent
        label = folder / "label.json"
        records.append(
            VideoRecord(
                class_name=folder.parent.name,
                name=folder.name,
                video_path=video,
                label_path=label if label.is_file() else None,
            )
        )
    # Sort explicitly so the order never depends on the file system.
    return sorted(records, key=lambda r: (r.class_name, r.name))


def load_boxes(label_path: Path | None) -> dict[int, list[GtBox]]:
    """Return {0-based frame index: [(x1, y1, x2, y2, category name), ...]}.

    Assumptions, verified on the real files in Step 0:
    - bbox is COCO [x, y, width, height] in pixels with the origin at the top-left corner
      (area == width * height, and every box lies inside the 640x480 frame).
    - images[].id runs 1..N where N is the number of decoded frames, so image_id is
      1-based. We subtract 1 to get the 0-based index used by OpenCV.
    A missing, empty or box-free label file (e.g. No_Gun) gives {}.
    """
    if label_path is None or not label_path.is_file() or label_path.stat().st_size == 0:
        return {}
    data = json.loads(label_path.read_text())
    names = {c["id"]: c["name"] for c in data.get("categories", [])}
    boxes: dict[int, list[GtBox]] = {}
    for ann in data.get("annotations", []):
        x, y, w, h = (float(v) for v in ann["bbox"])
        frame_index = int(ann["image_id"]) - 1  # 1-based image id -> 0-based frame
        name = names.get(ann["category_id"], str(ann["category_id"]))
        boxes.setdefault(frame_index, []).append((x, y, x + w, y + h, name))
    return boxes


def video_info(video_path: Path) -> dict:
    """Resolution, fps, frame count and duration read with OpenCV."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    duration = frames / fps if fps else 0.0
    return {"width": width, "height": height, "fps": fps, "frames": frames, "duration": duration}


def read_frame(video_path: Path, frame_index: int):
    """Return one 0-based frame as a BGR array, or None if it cannot be read."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    return frame if ok else None

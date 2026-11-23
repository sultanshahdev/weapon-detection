"""Facts about the Mendeley demo footage -> outputs/demo_cache/diagnostics.md (tables only).

Run with:  uv run python scripts/demo_diagnostics.py
D1. GT box size on all Handgun and Machine_Gun label.json files vs the firearm boxes in the
    TRAIN split of the Roboflow YOLO export (outputs/yolo_firearm/labels/train, read-only).
D2. For the three demo videos (top-ranked of each class in demo_selection.json): model
    boxes at inference size 640 vs 960, confidence 0.25. Informational only; the app stays
    at 640 because the model was trained at 640. Nothing here is written to the app cache.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weapon_detection.data.mendeley_videos import DATASET_ROOT, list_videos, load_boxes  # noqa: E402
from weapon_detection.detection.demo_cache import (  # noqa: E402
    CACHE_DIR,
    detect_video,
    markdown_table,
    matched_frames,
)
from weapon_detection.detection.model import YoloDetector  # noqa: E402

TRAIN_LABELS = ROOT / "outputs" / "yolo_firearm" / "labels" / "train"
CHECKPOINT = ROOT / "models" / "firearm_yolov8n_best.pt"
CONF = 0.25
SIZES = (640, 960)
PERCENTILES = (10, 50, 90)


def mendeley_sizes() -> dict[str, list[tuple[float, float, float]]]:
    """(width % of frame width, height % of frame height, area % of frame area) per class."""
    sizes: dict[str, list] = {}
    for label in sorted(DATASET_ROOT.glob("*/*/label.json")):
        data = json.loads(label.read_text())
        frame = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
        for ann in data["annotations"]:
            fw, fh = frame[ann["image_id"]]
            _, _, w, h = ann["bbox"]
            sizes.setdefault(label.parent.parent.name, []).append(
                (100 * w / fw, 100 * h / fh, 100 * w * h / (fw * fh))
            )
    return sizes


def roboflow_sizes() -> list[tuple[float, float, float]]:
    """YOLO labels are normalised (class cx cy w h), so w and h are already fractions."""
    sizes = []
    for label in sorted(TRAIN_LABELS.glob("*.txt")):
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[0] == "0":  # class 0 = firearm (data.yaml)
                w, h = float(parts[3]), float(parts[4])
                sizes.append((100 * w, 100 * h, 100 * w * h))
    return sizes


def size_rows() -> list[dict]:
    groups = {f"Mendeley {k} (GT)": v for k, v in mendeley_sizes().items()}
    groups["Roboflow train split (firearm)"] = roboflow_sizes()
    rows = []
    for name, values in groups.items():
        array = np.array(values)
        row = {"source": name, "boxes": len(values)}
        for col, label in enumerate(("width %", "height %", "area %")):
            for p in PERCENTILES:
                row[f"{label} p{p}"] = round(float(np.percentile(array[:, col], p)), 2)
        rows.append(row)
    return rows


def demo_videos() -> list:
    """Top-ranked video of each class from demo_selection.json."""
    selection = json.loads((CACHE_DIR / "demo_selection.json").read_text())["videos"]
    first = {}
    for v in selection:
        first.setdefault(v["class_name"], v["name"])
    records = {(r.class_name, r.name): r for r in list_videos(DATASET_ROOT)}
    return [records[(c, n)] for c, n in first.items()]


def imgsz_rows() -> list[dict]:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    rows = []
    videos = demo_videos()
    for size in SIZES:
        detector = YoloDetector(CHECKPOINT, confidence=CONF, imgsz=size)
        for record in videos:
            result = detect_video(detector, record.video_path, batch=16)
            gt = load_boxes(record.label_path)
            rows.append(
                {
                    "video": f"{record.class_name}/{record.name}",
                    "imgsz": size,
                    "frames": result["n_frames"],
                    "gt_frames": len(gt),
                    "frames_with_model_box": len(result["boxes"]),
                    "gt_frames_matched_iou50": matched_frames(gt, result["boxes"]),
                }
            )
            print(rows[-1])
    return rows


def main() -> None:
    lines = [
        "# Demo footage diagnostics (Mendeley videos; demo only, not an evaluation)",
        "",
        "## D1. Box size",
        "",
        "Columns: width % = box width / image width x 100; height % = box height / image "
        "height x 100; area % = box area / image area x 100; pN = N-th percentile over all "
        "boxes. Mendeley: every box in every Handgun and Machine_Gun label.json. Roboflow: "
        "every class-0 (firearm) box in outputs/yolo_firearm/labels/train.",
        "",
        markdown_table(size_rows()),
        "",
        "## D2. Inference size 640 vs 960 (three demo videos, confidence 0.25)",
        "",
        f"Checkpoint {CHECKPOINT.name}, CPU, every frame. Columns: frames = decoded frames; "
        "gt_frames = frames with a dataset box; frames_with_model_box = frames with at least "
        "one model box; gt_frames_matched_iou50 = GT frames where a model box overlaps a GT "
        "box with IoU >= 0.5. The app uses 640 (the training size); these numbers are not "
        "stored in the app cache.",
        "",
        markdown_table(imgsz_rows()),
    ]
    out = CACHE_DIR / "diagnostics.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"written {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

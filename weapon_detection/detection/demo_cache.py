"""Precomputed model detections for the Mendeley demo videos (demo only, not an evaluation).

The detector runs ONCE per video at a low confidence floor and the raw boxes are stored
in outputs/demo_cache/<class>/<video>.json. Every confidence threshold in the app is then
a filter on these stored boxes, so moving the slider never calls the model again.
"""

import json
import os
from pathlib import Path

import cv2

from weapon_detection.utils.config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "outputs" / "demo_cache"
# Lowest confidence we store; every threshold in the app is filtered from this.
DETECTION_FLOOR = 0.05
# The fine-tuned model was trained at 640, so the cache uses the same input size.
IMGSZ = 640
# Thresholds listed in the fact tables (display only; none was selected on this footage).
THRESHOLDS = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)
IOU_MATCH = 0.5


def signature(checkpoint: Path, imgsz: int = IMGSZ, floor: float = DETECTION_FLOOR) -> dict:
    """What a cache depends on. A cache with a different signature is stale."""
    stat = checkpoint.stat()
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "imgsz": imgsz,
        "floor": floor,
    }


def cache_path(class_name: str, video_name: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / class_name / f"{video_name}.json"


def write_cache(path: Path, data: dict) -> None:
    """Write atomically: a temp file in the same folder, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(data))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)  # only left over if the write failed or was interrupted


def read_cache(path: Path, expected: dict | None = None) -> dict | None:
    """Return the cache, or None if missing, unreadable or (if expected given) stale."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if expected is not None and any(data.get(k) != v for k, v in expected.items()):
        return None
    # JSON keys are strings; turn them back into 0-based frame indices.
    data["boxes"] = {int(k): v for k, v in data["boxes"].items()}
    return data


def pending(records: list, expected: dict, cache_dir: Path = CACHE_DIR) -> list:
    """Videos without a valid cache; these are the only ones a resumed run must compute."""
    return [
        r
        for r in records
        if read_cache(cache_path(r.class_name, r.name, cache_dir), expected) is None
    ]


def filter_boxes(boxes: dict[int, list], conf: float) -> dict[int, list]:
    """Keep boxes with confidence >= conf (the 5th value); drop frames left empty."""
    kept = {i: [b for b in found if b[4] >= conf] for i, found in boxes.items()}
    return {i: found for i, found in kept.items() if found}


def iou(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2, ...) boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, width) * max(0.0, height)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def matched_frames(gt: dict[int, list], model: dict[int, list], threshold=IOU_MATCH) -> int:
    """Frames where at least one model box overlaps a GT box with IoU >= threshold."""
    return sum(
        any(iou(g, m) >= threshold for g in gt_boxes for m in model.get(i, []))
        for i, gt_boxes in gt.items()
    )


def threshold_facts(gt: dict[int, list], boxes: dict[int, list], n_frames: int) -> list[dict]:
    """One row per threshold, computed from the cached boxes only."""
    rows = []
    for t in THRESHOLDS:
        model = filter_boxes(boxes, t)
        rows.append(
            {
                "threshold": t,
                "frames": n_frames,
                "frames_with_model_box": len(model),
                "model_boxes": sum(len(v) for v in model.values()),
                "gt_frames": len(gt),
                "gt_frames_matched_iou50": matched_frames(gt, model),
            }
        )
    return rows


def order_videos(records: list, first: list[tuple[str, str]]) -> list:
    """The `first` (class, name) videos in that order, then one per class in turn."""
    by_key = {(r.class_name, r.name): r for r in records}
    ordered = [by_key[k] for k in first if k in by_key]
    queues: dict[str, list] = {}
    for r in sorted(records, key=lambda r: (r.class_name, r.name)):
        if r not in ordered:
            queues.setdefault(r.class_name, []).append(r)
    while any(queues.values()):
        for class_name in sorted(queues):
            if queues[class_name]:
                ordered.append(queues[class_name].pop(0))
    return ordered


def detect_video(detector, video: Path, batch: int = 1, progress=None) -> dict:
    """Every frame of one video -> {frame index: [[x1, y1, x2, y2, conf], ...]}.

    detector is the repo's YoloDetector built with confidence=DETECTION_FLOOR.
    progress, if given, is called with the fraction of frames done.
    """
    capture = cv2.VideoCapture(str(video))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    boxes, frames, index = {}, [], 0

    def flush() -> None:
        nonlocal index
        found = detector.predict_batch(frames) if batch > 1 else [detector.predict(frames[0])]
        for detections in found:
            if detections:
                boxes[index] = [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections]
            index += 1
        frames.clear()
        if progress is not None:
            progress(min(1.0, index / total))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) == batch:
                flush()
        if frames:
            flush()
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        capture.release()
    return {"fps": fps, "n_frames": index, "boxes": boxes}


SUMMARY_HEADER = (
    "Demo footage statistics, not an evaluation. This footage is outside every split "
    "and no threshold was selected on it."
)


def summary_rows(entries: list[tuple[str, dict, dict]]) -> list[dict]:
    """Per class and threshold, averaged over videos.

    entries: (class name, GT boxes by frame, cache dict) for every precomputed video.
    frac_frames_with_model_box = frames with >= 1 model box / frames in the video.
    frac_gt_frames_matched_iou50 = GT frames with a model box at IoU >= 0.5 / GT frames
    (Handgun and Machine_Gun only; empty for No_Gun).
    """
    rows = []
    for class_name in sorted({c for c, _, _ in entries}):
        videos = [(gt, cache) for c, gt, cache in entries if c == class_name]
        for t in THRESHOLDS:
            frame_fracs, match_fracs = [], []
            for gt, cache in videos:
                model = filter_boxes(cache["boxes"], t)
                frame_fracs.append(len(model) / max(1, cache["n_frames"]))
                if gt:
                    match_fracs.append(matched_frames(gt, model) / len(gt))
            rows.append(
                {
                    "class": class_name,
                    "threshold": t,
                    "videos": len(videos),
                    "frac_frames_with_model_box": round(sum(frame_fracs) / len(videos), 3),
                    "frac_gt_frames_matched_iou50": (
                        round(sum(match_fracs) / len(match_fracs), 3) if match_fracs else None
                    ),
                }
            )
    return rows


def markdown_table(rows: list[dict]) -> str:
    """A plain Markdown table (None shown as an empty cell)."""
    if not rows:
        return "(no rows)"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        cells = ["" if row[k] is None else str(row[k]) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_summary(rows: list[dict], cache_dir: Path = CACHE_DIR) -> None:
    """Write threshold_summary.csv and threshold_summary.md (same numbers)."""
    import csv

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "threshold_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["class"])
        writer.writeheader()
        writer.writerows(rows)
    count = sum(r["videos"] for r in rows if r["threshold"] == THRESHOLDS[0])
    lines = [
        SUMMARY_HEADER,
        "",
        f"Videos included: {count}",
        "",
        "Columns: frac_frames_with_model_box = mean over videos of (frames with at least one "
        "model box / frames); frac_gt_frames_matched_iou50 = mean over videos of (GT frames "
        "with a model box at IoU >= 0.5 / GT frames), Handgun and Machine_Gun only.",
        "",
        markdown_table(rows),
    ]
    (cache_dir / "threshold_summary.md").write_text("\n".join(lines) + "\n")

"""Detection-cache tests with tiny synthetic data (no real footage, no model)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from weapon_detection.data.mendeley_videos import VideoRecord
from weapon_detection.detection.annotated_video import ffmpeg_path, render_annotated_video
from weapon_detection.detection.demo_cache import (
    THRESHOLDS,
    cache_path,
    detect_video,
    filter_boxes,
    iou,
    matched_frames,
    order_videos,
    pending,
    read_cache,
    signature,
    summary_rows,
    threshold_facts,
    write_cache,
)
from weapon_detection.detection.model import Detection

BOXES = {0: [[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.1]], 3: [[5, 5, 15, 15, 0.3]]}


def make_checkpoint(tmp_path: Path, content: bytes = b"weights") -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(content)
    return path


def record(class_name: str, name: str) -> VideoRecord:
    return VideoRecord(class_name, name, Path(f"{name}.mp4"), None)


def test_cache_round_trip(tmp_path):
    expected = signature(make_checkpoint(tmp_path))
    path = cache_path("Handgun", "A1", tmp_path / "cache")
    write_cache(path, {**expected, "fps": 25.0, "n_frames": 4, "boxes": BOXES})
    data = read_cache(path, expected)
    assert data["boxes"] == BOXES  # string keys come back as 0-based int frame indices
    assert data["n_frames"] == 4
    assert not list(path.parent.glob("*.tmp"))  # atomic write leaves no temp file


def test_cache_invalid_when_signature_changes(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    path = cache_path("Handgun", "A1", tmp_path)
    write_cache(path, {**signature(checkpoint), "fps": 25.0, "n_frames": 4, "boxes": {}})
    checkpoint.write_bytes(b"retrained weights")  # size and mtime change
    assert read_cache(path, signature(checkpoint)) is None
    assert read_cache(path, signature(checkpoint, imgsz=960)) is None
    assert read_cache(tmp_path / "missing.json") is None


def test_resume_skips_valid_caches(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    expected = signature(checkpoint)
    valid, missing, stale = record("Handgun", "A"), record("Handgun", "B"), record("No_Gun", "C")
    base = {"fps": 25.0, "n_frames": 1, "boxes": {}}
    write_cache(cache_path("Handgun", "A", tmp_path), {**expected, **base})
    write_cache(cache_path("No_Gun", "C", tmp_path), {**expected, **base, "floor": 0.5})
    assert pending([valid, missing, stale], expected, tmp_path) == [missing, stale]


def test_filter_by_confidence():
    assert filter_boxes(BOXES, 0.25) == {0: [[0, 0, 10, 10, 0.9]], 3: [[5, 5, 15, 15, 0.3]]}
    assert filter_boxes(BOXES, 0.5) == {0: [[0, 0, 10, 10, 0.9]]}  # empty frames dropped
    assert filter_boxes(BOXES, 0.95) == {}


def test_iou_and_matching():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    gt = {0: [(0, 0, 10, 10, "Handgun")], 3: [(0, 0, 10, 10, "Handgun")], 7: [(0, 0, 4, 4, "x")]}
    # frame 0 matches exactly; frame 3 overlaps with IoU 25/175 < 0.5; frame 7 has no model box
    assert matched_frames(gt, BOXES) == 1
    facts = threshold_facts(gt, BOXES, n_frames=8)
    assert [row["threshold"] for row in facts] == list(THRESHOLDS)
    assert facts[0]["frames_with_model_box"] == 2 and facts[0]["model_boxes"] == 3
    assert facts[-1]["model_boxes"] == 1 and facts[-1]["gt_frames"] == 3


def test_summary_rows_per_class():
    cache = {"n_frames": 4, "boxes": BOXES}
    gt = {0: [(0, 0, 10, 10, "Handgun")]}
    rows = summary_rows([("Handgun", gt, cache), ("No_Gun", {}, cache)])
    hand = [r for r in rows if r["class"] == "Handgun" and r["threshold"] == 0.05][0]
    none = [r for r in rows if r["class"] == "No_Gun" and r["threshold"] == 0.05][0]
    assert hand["frac_frames_with_model_box"] == 0.5 and hand["frac_gt_frames_matched_iou50"] == 1
    assert none["frac_gt_frames_matched_iou50"] is None


def test_round_robin_with_demo_videos_first():
    records = [record(c, f"{c[0]}{i}") for c in ("Handgun", "Machine_Gun", "No_Gun") for i in "123"]
    first = [("Handgun", "H2"), ("Machine_Gun", "M3"), ("No_Gun", "N1")]
    names = [r.name for r in order_videos(records, first)]
    assert names[:3] == ["H2", "M3", "N1"]
    assert names[3:] == ["H1", "M1", "N2", "H3", "M2", "N3"]  # one class at a time


class FakeDetector:
    """Mocked inference: a box only on odd frames; works per frame and in batches."""

    def __init__(self):
        self.seen = 0

    def _one(self):
        self.seen += 1
        if self.seen % 2 == 0:
            return (Detection(0, "firearm", 0.8, 1, 2, 3, 4),)
        return ()

    def predict(self, frame):
        return self._one()

    def predict_batch(self, frames):
        return [self._one() for _ in frames]


def write_video(path: Path, frames: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 24))
    for i in range(frames):
        writer.write(np.full((24, 32, 3), i * 40, dtype=np.uint8))
    writer.release()


def test_detect_video_same_with_and_without_batching(tmp_path):
    video = tmp_path / "v.mp4"
    write_video(video)
    single = detect_video(FakeDetector(), video, batch=1)
    batched = detect_video(FakeDetector(), video, batch=2)  # 5 frames: batches of 2, 2, 1
    assert single == batched
    assert sorted(single["boxes"]) == [1, 3] and single["n_frames"] == 5


@pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not available")
def test_render_from_cached_boxes_without_model(tmp_path):
    video = tmp_path / "v.mp4"
    write_video(video)
    out = tmp_path / "out.mp4"
    cached = {1: [[1, 1, 9, 9, 0.9]], 2: [[1, 1, 9, 9, 0.1]]}
    sidecar = render_annotated_video(video, {}, None, out, 0.25, cached_boxes=cached)
    assert sidecar["model_on"] and sorted(sidecar["model_boxes"]) == ["1"]
    assert json.loads(out.with_suffix(".json").read_text())["frames"] == 5

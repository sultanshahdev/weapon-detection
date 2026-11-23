"""Loader and renderer tests on a tiny synthetic dataset (no real footage needed)."""

import json

import cv2
import numpy as np
import pytest

from weapon_detection.data.mendeley_videos import list_videos, load_boxes, video_info
from weapon_detection.detection.annotated_video import (
    cache_name,
    check_playable,
    ffmpeg_path,
    render_annotated_video,
)
from weapon_detection.detection.model import Detection

FRAMES = 5


def write_video(path, frames=FRAMES, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i * 40, dtype=np.uint8))
    writer.release()


def write_label(path, annotations):
    data = {
        "categories": [{"id": 1, "name": "Handgun"}],
        "images": [{"id": i, "width": 64, "height": 48} for i in range(1, FRAMES + 1)],
        "annotations": annotations,
    }
    path.write_text(json.dumps(data))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "Gun_Action_Recognition_Dataset"
    for cls, name in [("No_Gun", "N1"), ("Handgun", "B2"), ("Handgun", "A1")]:
        write_video(root / cls / name / "video.mp4")
    write_label(
        root / "Handgun" / "A1" / "label.json",
        [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 8]},
            {"id": 2, "image_id": 3, "category_id": 1, "bbox": [0, 0, 4, 4]},
        ],
    )
    (root / "Handgun" / "B2" / "label.json").write_text("")  # empty file
    return root


def test_list_videos_sorted_and_labels(dataset):
    # Accepts the parent folder too, and orders by class then video name.
    records = list_videos(dataset.parent)
    assert [(r.class_name, r.name) for r in records] == [
        ("Handgun", "A1"),
        ("Handgun", "B2"),
        ("No_Gun", "N1"),
    ]
    assert records[0].label_path.name == "label.json"
    assert records[2].label_path is None
    assert list_videos(dataset) == records  # deterministic


def test_load_boxes_xywh_to_xyxy_and_zero_based(dataset):
    boxes = load_boxes(dataset / "Handgun" / "A1" / "label.json")
    # image_id 1 -> frame 0, image_id 3 -> frame 2; [x, y, w, h] -> [x1, y1, x2, y2]
    assert boxes == {0: [(10.0, 5.0, 30.0, 13.0, "Handgun")], 2: [(0.0, 0.0, 4.0, 4.0, "Handgun")]}


def test_load_boxes_missing_or_empty(dataset):
    assert load_boxes(None) == {}
    assert load_boxes(dataset / "nope.json") == {}
    assert load_boxes(dataset / "Handgun" / "B2" / "label.json") == {}


def test_video_info(dataset):
    info = video_info(dataset / "Handgun" / "A1" / "video.mp4")
    assert (info["width"], info["height"], info["frames"]) == (64, 48, FRAMES)


def test_cache_name_changes_with_settings():
    a = cache_name("Handgun_A1", True, "models/best.pt", 0.25)
    assert a != cache_name("Handgun_A1", True, "models/best.pt", 0.30)
    assert a != cache_name("Handgun_A1", False, "models/best.pt", 0.25)
    assert cache_name("Handgun_A1", True, None, 0.25).endswith("nomodel.mp4")


class FakeDetector:
    """Mocked inference: one box on odd frames, with one low-confidence box always."""

    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        found = [Detection(0, "firearm", 0.10, 1, 1, 5, 5)]
        if self.calls % 2 == 0:
            found.append(Detection(0, "firearm", 0.90, 2, 2, 20, 20))
        return tuple(found)


@pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not available")
def test_render_writes_h264_and_sidecar(dataset, tmp_path):
    video = dataset / "Handgun" / "A1" / "video.mp4"
    out = tmp_path / "out" / "a.mp4"
    boxes = load_boxes(dataset / "Handgun" / "A1" / "label.json")
    sidecar = render_annotated_video(video, boxes, FakeDetector(), out, conf=0.25)
    assert sidecar["frames"] == FRAMES
    # Only the 0.90 boxes pass conf 0.25, on frames 1 and 3 (0-based).
    assert sorted(sidecar["model_boxes"]) == ["1", "3"]
    assert json.loads(out.with_suffix(".json").read_text()) == sidecar
    playable = check_playable(out)
    assert playable["frames"] == FRAMES
    assert playable["codec"] in {"avc1", "h264", "H264"}
    assert not list(out.parent.glob("*.partial.mp4"))

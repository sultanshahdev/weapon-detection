from base64 import b64decode
from contextlib import closing
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from weapon_detection.detection import playlist
from weapon_detection.detection.model import Detection
from weapon_detection.ui import monitor
from weapon_detection.ui.demo import detection_status, encoded_frame, screen

ROOT = Path(__file__).resolve().parents[1]
HIT = Detection(17, "Handgun", 0.87, 5, 5, 45, 35)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return (HIT,)


def write_clip(path: Path) -> None:
    capture = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6, (64, 48))
    assert capture.isOpened()
    try:
        capture.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        capture.release()


def test_discovery_skip_and_loop(tmp_path: Path, caplog) -> None:
    filenames = ("02.avi", "01_bad.avi", "00.avi")
    for name in filenames:
        path = tmp_path / name
        if "bad" in name:
            path.write_bytes(b"bad clip")
        else:
            write_clip(path)
    (tmp_path / "notes.txt").touch()
    (tmp_path / "alias.avi").symlink_to(tmp_path / "00.avi")
    paths = monitor.approved_files(tmp_path, monitor.VIDEO_SUFFIXES)
    assert [p.name for p in paths] == sorted(filenames)
    detector = FakeDetector()
    with closing(
        playlist.playlist_frames(paths, detector, stride=1, active=lambda: True)
    ) as frames:
        output = list(islice(frames, 4))
    assert [path.name for path, _ in output] == ["00.avi", "02.avi", "00.avi", "02.avi"]
    assert all(frame.detections == (HIT,) for _, frame in output)
    assert detector.calls == 4
    assert caplog.text.count("Skipping demo clip 01_bad.avi") == 1


def test_all_unplayable_stops_without_retry(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.touch()
    with pytest.raises(ValueError, match="No playable clips remain"):
        list(playlist.playlist_frames([bad], FakeDetector(), stride=1, active=lambda: True))


def test_empty_app_shows_instructions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WD_DEMO_PLAYLIST", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "demo_app.py")).run()
    assert not app.exception
    assert any("datasets/demo_playlist/" in elm.proto.body for elm in app.get("html"))


def test_closing_playlist_releases_capture(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "actual.avi"
    write_clip(path)
    capture = cv2.VideoCapture(str(path))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    with closing(
        playlist.playlist_frames([path], FakeDetector(), stride=1, active=lambda: True)
    ) as frames:
        next(frames)
        assert capture.isOpened()
    assert not capture.isOpened()


def test_status_annotation_and_channel_order(monkeypatch) -> None:
    overlay_labels = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *_: overlay_labels.append(text))
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    annotated = playlist.annotate_demo(source, (HIT,))
    decoded = cv2.imdecode(np.frombuffer(b64decode(encoded_frame(annotated)), np.uint8), 1)
    assert overlay_labels == ["Handgun 0.87"]
    assert not source.any()
    assert decoded[15, 5, 2] > 160 and decoded[15, 5, 0] < 90
    assert detection_status((HIT,))[0] == "FIREARM DETECTED"
    assert detection_status(())[0] == "NO FIREARM DETECTED IN CURRENT FRAME"
    page = screen(source="<script>.avi", image=encoded_frame(annotated), detections=(HIT,), fps=2.4)
    assert "&lt;script&gt;.avi" in page and "2.4 FPS" in page


def test_model_cache_reuses_detector_for_same_checkpoint(tmp_path: Path, monkeypatch) -> None:
    created = []

    def load(path, **settings):
        created.append(path)
        return FakeDetector()

    path = tmp_path / "checkpoint.pt"
    key = (path, (0, 0, 0), 0.35, 416)
    monkeypatch.setattr(monitor, "YoloDetector", load)
    try:
        assert monitor.cached_detector(*key) is monitor.cached_detector(*key)
        assert created == [path]
    finally:
        monitor.cached_detector.clear(*key)

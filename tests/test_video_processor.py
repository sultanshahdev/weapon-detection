from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import pytest

from weapon_detection.detection.model import Detection
from weapon_detection.detection.video_processor import VideoStats, process_video


class FakeDetector:
    def predict(self, frame):
        return (Detection(4, "Handgun", 0.8, 5, 5, 20, 20),)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(6):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_invalid_video(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    with pytest.raises(ValueError, match="does not exist"):
        list(process_video(path, FakeDetector()))
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Could not open"):
        list(process_video(path, FakeDetector()))


def test_real_codec_stride_limit_and_output(video: Path, tmp_path: Path) -> None:
    output = tmp_path / "result.mp4"
    stats = VideoStats()
    frames = list(
        process_video(
            video,
            FakeDetector(),
            output=output,
            stats=stats,
            frame_stride=2,
            max_frames=2,
            output_size=(32, 24),
        )
    )
    assert [frame.frame_index for frame in frames] == [0, 2]
    assert [frame.timestamp_seconds for frame in frames] == [0, 0.2]
    assert frames[0].annotated_frame.shape == (48, 64, 3)
    assert stats.decoded_frames == 3
    assert stats.processed_frames == stats.detection_count == 2
    assert stats.stopped_at_limit and stats.reported_total_frames == 6
    assert stats.summary()["estimated_video_duration_seconds"] == pytest.approx(0.6)
    capture = cv2.VideoCapture(str(output))
    try:
        success, frame = capture.read()
        assert success and frame.shape == (24, 32, 3)
        assert capture.get(cv2.CAP_PROP_FPS) == 5
    finally:
        capture.release()


def test_writer_failure_and_early_close(
    video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released = []

    class BrokenWriter:
        def isOpened(self):
            return False

        def release(self):
            released.append(True)

    output = tmp_path / "failed.avi"
    with closing(process_video(video, FakeDetector(), output=output)) as frames:
        assert next(frames).frame_index == 0
    assert not output.exists()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: BrokenWriter())
    with pytest.raises(ValueError, match="writer"):
        list(process_video(video, FakeDetector(), output=output))
    assert released == [True] and not output.exists()
    assert list(tmp_path.iterdir()) == [video]

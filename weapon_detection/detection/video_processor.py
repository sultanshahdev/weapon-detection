"""Stream prerecorded frames and optionally write annotated MP4, AVI or WebM."""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from weapon_detection.detection.inference import annotate_frame
from weapon_detection.detection.model import Detection, Detector


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    timestamp_seconds: float
    detections: tuple[Detection, ...]
    annotated_frame: np.ndarray
    inference_latency_ms: float


@dataclass(slots=True)
class VideoStats:
    decoded_frames: int = 0
    processed_frames: int = 0
    detection_count: int = 0
    reported_total_frames: int | None = None
    source_fps: float = 0.0
    elapsed_seconds: float = 0.0
    stopped_at_limit: bool = False
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        duration = (
            self.reported_total_frames / self.source_fps
            if self.reported_total_frames and self.source_fps
            else None
        )
        return {
            "decoded_frames": self.decoded_frames,
            "processed_frames": self.processed_frames,
            "detection_count": self.detection_count,
            "reported_total_frames": self.reported_total_frames,
            "source_fps": self.source_fps,
            "estimated_video_duration_seconds": duration,
            "elapsed_seconds": self.elapsed_seconds,
            "stopped_at_limit": self.stopped_at_limit,
            "processed_frames_per_second": (
                self.processed_frames / self.elapsed_seconds if self.elapsed_seconds > 0 else None
            ),
            "decoded_frames_per_second": (
                self.decoded_frames / self.elapsed_seconds if self.elapsed_seconds > 0 else None
            ),
            "inference_mean_ms": (float(np.mean(self.latencies_ms)) if self.latencies_ms else None),
            "inference_median_ms": (
                float(np.median(self.latencies_ms)) if self.latencies_ms else None
            ),
            "inference_p95_ms": (
                float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else None
            ),
        }


def process_video(
    source: Path,
    detector: Detector,
    *,
    stats: VideoStats | None = None,
    output: Path | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    output_size: tuple[int, int] | None = None,
    annotator: Callable[[np.ndarray, tuple[Detection, ...]], np.ndarray] = annotate_frame,
) -> Iterator[FrameResult]:
    """Yield selected frames. Close this iterator if consumption stops early."""
    if frame_stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("Frame stride and maximum processed frames must be positive.")
    if output_size and (not output or min(output_size) < 2 or any(v % 2 for v in output_size)):
        raise ValueError("Output size requires an output file and positive even dimensions.")
    if not source.is_file():
        raise ValueError(f"Video file does not exist: {source}")
    codec = None
    if output is not None:
        codec = {".mp4": "mp4v", ".avi": "MJPG", ".webm": "VP80"}.get(output.suffix.lower())
        if not codec or output.resolve() == source.resolve() or output.exists():
            raise ValueError("Use a new .mp4, .avi or .webm output path distinct from the input.")

    record_latencies = stats is not None
    stats = stats if stats is not None else VideoStats()
    start = perf_counter()
    capture = cv2.VideoCapture(str(source))
    writer = None
    temp_dir = None
    temporary_video = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source}")
        stats.source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(stats.source_fps) or stats.source_fps <= 0:
            raise ValueError("Video has no usable frame rate.")
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        stats.reported_total_frames = int(total) if math.isfinite(total) and total >= 1 else None
        while True:
            ok, frame = capture.read()
            if not ok:
                if not stats.decoded_frames:
                    raise ValueError(f"Could not decode a frame from {source}")
                if (
                    stats.reported_total_frames
                    and stats.decoded_frames < stats.reported_total_frames
                ):
                    raise ValueError(f"Decoding stopped early at frame {stats.decoded_frames}.")
                break
            index = stats.decoded_frames
            stats.decoded_frames += 1
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
                raise ValueError(f"Invalid decoded frame at index {index}.")
            if index % frame_stride:
                continue
            tick = perf_counter()
            detections = detector.predict(frame)
            latency = (perf_counter() - tick) * 1000
            annotated = annotator(frame, detections)
            if output is not None:
                size = output_size or (annotated.shape[1] & ~1, annotated.shape[0] & ~1)
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temp_dir = TemporaryDirectory(dir=output.parent)
                    temporary_video = Path(temp_dir.name) / output.name
                    writer = cv2.VideoWriter(
                        str(temporary_video),
                        cv2.VideoWriter_fourcc(*codec),
                        stats.source_fps / frame_stride,
                        size,
                    )
                    if not writer.isOpened():
                        raise ValueError(f"Could not open {codec} video writer for {output}")
                frame_to_write = (
                    annotated if annotated.shape[:2] == size[::-1] else cv2.resize(annotated, size)
                )
                writer.write(frame_to_write)
            stats.processed_frames += 1
            stats.detection_count += len(detections)
            if record_latencies:
                stats.latencies_ms.append(latency)
            yield FrameResult(index, index / stats.source_fps, detections, annotated, latency)
            if max_frames is not None and stats.processed_frames >= max_frames:
                stats.stopped_at_limit = True
                break
        if writer is not None:
            writer.release()
            writer = None
            verification = cv2.VideoCapture(str(temporary_video))
            try:
                written = 0
                while verification.grab():
                    written += 1
                if written != stats.processed_frames:
                    raise ValueError(
                        f"Expected {stats.processed_frames} written frames, got {written}."
                    )
            finally:
                verification.release()
            output.hardlink_to(temporary_video)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if temp_dir is not None:
            temp_dir.cleanup()
        stats.elapsed_seconds = perf_counter() - start

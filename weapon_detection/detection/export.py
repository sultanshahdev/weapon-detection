"""Shared frame-level CSV schema for the CLI and dashboard."""

from collections.abc import Iterator
from dataclasses import asdict

from weapon_detection.detection.video_processor import FrameResult
from weapon_detection.utils.csv import spreadsheet_safe

FIELDS = (
    "video_name",
    "frame_index",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "normalized_class",
)


def detection_rows(source: str, frame: FrameResult) -> Iterator[dict]:
    for detection in frame.detections:
        yield {
            "video_name": spreadsheet_safe(source),
            "frame_index": frame.frame_index,
            "timestamp_seconds": frame.timestamp_seconds,
            **asdict(detection),
            "class_name": spreadsheet_safe(detection.class_name),
        }

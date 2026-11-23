"""Connect video processing with temporal alerting and persistent evidence."""

import csv
import logging
import shutil
from collections.abc import Callable
from contextlib import closing, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from weapon_detection.detection.export import FIELDS, detection_rows
from weapon_detection.detection.model import Detector
from weapon_detection.detection.video_processor import FrameResult, VideoStats, process_video
from weapon_detection.incidents.alert_engine import AlertEngine, AlertRules, ConfirmedEvent
from weapon_detection.storage.evidence import EvidenceStore
from weapon_detection.storage.repository import IncidentRepository, utc_now

logger = logging.getLogger(__name__)
Progress = Callable[[FrameResult, VideoStats, int, dict | None, float], None]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    run_id: str
    source_name: str
    video: Path
    detections_csv: Path
    stats: dict
    incident_count: int


def persist_event(
    event: ConfirmedEvent,
    frame: FrameResult,
    repository: IncidentRepository,
    evidence: EvidenceStore,
) -> dict:
    if existing := repository.get_incident(event.id):
        return existing
    if event.frame_index != frame.frame_index or event.timestamp != frame.timestamp_seconds:
        raise ValueError("Evidence frame does not match the confirmed event.")
    try:
        snapshot = evidence.save(event.run_id, event.id, frame.annotated_frame)
    except FileExistsError:
        if existing := repository.get_incident(event.id):
            return existing
        raise

    detection = event.detection
    record = {
        "id": event.id,
        "run_id": event.run_id,
        "source_name": event.source_name,
        "created_at_utc": utc_now(),
        "source_frame_index": event.frame_index,
        "timestamp_seconds": event.timestamp,
        "normalized_class": detection.normalized_class,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox_x1": detection.x1,
        "bbox_y1": detection.y1,
        "bbox_x2": detection.x2,
        "bbox_y2": detection.y2,
        "confirmation_count": event.confirmation_count,
        "detection_count": event.detection_count,
        "snapshot_path": snapshot.path,
        "snapshot_sha256": snapshot.sha256,
        "review_status": "new",
    }
    try:
        if repository.add_incident(record):
            return record
        existing = repository.get_incident(event.id)
        if existing is None:
            raise RuntimeError("Existing evidence record could not be read.")
        if existing["snapshot_path"] != snapshot.path:
            evidence.delete(snapshot.path)
        return existing
    except Exception:
        with suppress(OSError):
            evidence.delete(snapshot.path)
        raise


def analyze_video(
    source: Path,
    detector: Detector,
    repository: IncidentRepository,
    evidence: EvidenceStore,
    output_dir: Path,
    *,
    source_name: str,
    source_type: str,
    model: str,
    image_size: int,
    rules: AlertRules,
    stride: int,
    max_frames: int,
    progress: Progress | None = None,
) -> AnalysisResult:
    if min(stride, max_frames, image_size) < 1:
        raise ValueError("Frame stride, limit and image size must be positive.")
    run_id = uuid4().hex
    directory = output_dir / "runs" / run_id
    video = directory / "annotated.webm"
    csv_path = directory / "detections.csv"
    stats = VideoStats()
    engine = AlertEngine(run_id, source_name, rules)
    incident_count = 0
    started = perf_counter()
    repository.create_run(
        run_id,
        source_name,
        source_type,
        model,
        confidence=rules.confidence,
        image_size=image_size,
        stride=stride,
        max_frames=max_frames,
        rules=asdict(rules),
    )
    try:
        directory.mkdir(parents=True)
        with (
            csv_path.open("x", newline="", encoding="utf-8") as stream,
            closing(
                process_video(
                    source,
                    detector,
                    stats=stats,
                    output=video,
                    frame_stride=stride,
                    max_frames=max_frames,
                )
            ) as frames,
        ):
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for frame in frames:
                writer.writerows(detection_rows(source_name, frame))
                event = engine.observe(frame.frame_index, frame.timestamp_seconds, frame.detections)
                incident = None
                if event is not None:
                    incident = persist_event(event, frame, repository, evidence)
                    incident_count += 1
                if progress is not None:
                    progress(frame, stats, incident_count, incident, perf_counter() - started)
        if not video.is_file() or not video.stat().st_size:
            raise OSError("Annotated video output is missing or empty.")
        summary = stats.summary()
        repository.finish_run(run_id, "completed", summary)
    except BaseException as exc:
        status = "failed" if isinstance(exc, Exception) else "cancelled"
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
        try:
            repository.finish_run(run_id, status, stats.summary(), type(exc).__name__)
        except Exception:
            logger.exception("Could not finalize run %s as %s", run_id, status)
        raise

    return AnalysisResult(run_id, source_name, video, csv_path, summary, incident_count)

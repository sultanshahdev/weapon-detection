"""Persist processing runs, confirmed incidents and review metadata."""

import json
from datetime import UTC, datetime
from pathlib import Path

from weapon_detection.storage.database import connect, initialize

REVIEW_STATUSES = ("new", "reviewed", "dismissed")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class IncidentRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        initialize(path)

    def create_run(
        self,
        run_id: str,
        source: str,
        source_type: str,
        model: str,
        *,
        confidence: float,
        image_size: int,
        stride: int,
        max_frames: int,
        rules: dict,
    ) -> None:
        with connect(self.path) as db:
            db.execute(
                """INSERT INTO processing_runs (
                    id, source_name, source_type, model_identifier, started_at_utc, status,
                    confidence_threshold, image_size, frame_stride, max_frames, alert_rules_json
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    source,
                    source_type,
                    model,
                    utc_now(),
                    confidence,
                    image_size,
                    stride,
                    max_frames,
                    json.dumps(rules),
                ),
            )

    def finish_run(self, run_id: str, status: str, stats: dict, error: str | None = None) -> None:
        if status not in ("completed", "failed", "cancelled"):
            raise ValueError("Invalid final run status.")
        with connect(self.path) as db:
            updated = db.execute(
                """UPDATE processing_runs SET status=?, completed_at_utc=?,
                    total_source_frames=?, decoded_frames=?, processed_frames=?, detection_count=?,
                    processing_duration_seconds=?, mean_inference_latency_ms=?, error_message=?
                    WHERE id=? AND status='running'""",
                (
                    status,
                    utc_now(),
                    stats["reported_total_frames"],
                    stats["decoded_frames"],
                    stats["processed_frames"],
                    stats["detection_count"],
                    stats["elapsed_seconds"],
                    stats["inference_mean_ms"],
                    error,
                    run_id,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("Processing run is missing or already finalized.")

    def add_incident(self, record: dict) -> bool:
        with connect(self.path) as db:
            result = db.execute(
                """INSERT INTO incidents (
                    id, run_id, created_at_utc, source_name, source_frame_index,
                    timestamp_seconds, normalized_class, class_name, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, confirmation_count, detection_count,
                    snapshot_path, snapshot_sha256
                ) VALUES (
                    :id, :run_id, :created_at_utc, :source_name, :source_frame_index,
                    :timestamp_seconds, :normalized_class, :class_name, :confidence,
                    :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2, :confirmation_count, :detection_count,
                    :snapshot_path, :snapshot_sha256
                ) ON CONFLICT(id) DO NOTHING""",
                record,
            )
            return result.rowcount == 1

    def get_incident(self, incident_id: str) -> dict | None:
        with connect(self.path) as db:
            row = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
            return dict(row) if row else None

    def list_incidents(
        self,
        *,
        status: str | None = None,
        source: str | None = None,
        confidence: float = 0,
    ) -> list[dict]:
        with connect(self.path) as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT * FROM incidents
                   WHERE (? IS NULL OR review_status=?) AND (? IS NULL OR source_name=?)
                     AND confidence >= ?
                   ORDER BY created_at_utc DESC, id DESC""",
                    (status, status, source, source, confidence),
                )
            ]

    def review(self, incident_id: str, status: str, notes: str) -> None:
        if status not in REVIEW_STATUSES or len(notes) > 4000:
            raise ValueError("Select a valid review status and keep notes within 4,000 characters.")
        with connect(self.path) as db:
            updated = db.execute(
                "UPDATE incidents SET review_status=?, review_notes=?, reviewed_at_utc=? "
                "WHERE id=?",
                (status, notes, utc_now(), incident_id),
            )
            if updated.rowcount != 1:
                raise ValueError("Incident no longer exists.")

    def list_runs(self) -> list[dict]:
        with connect(self.path) as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT r.*,
                       (SELECT count(*) FROM incidents WHERE run_id=r.id) AS incident_count
                   FROM processing_runs r ORDER BY started_at_utc DESC, r.id DESC"""
                )
            ]

"""Versioned SQLite schema and a per-operation connection lifecycle."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS processing_runs (
    id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('upload', 'sample')),
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
    model_identifier TEXT NOT NULL,
    confidence_threshold REAL NOT NULL CHECK(confidence_threshold BETWEEN 0 AND 1),
    image_size INTEGER NOT NULL CHECK(image_size > 0),
    frame_stride INTEGER NOT NULL CHECK(frame_stride > 0),
    max_frames INTEGER NOT NULL CHECK(max_frames > 0),
    alert_rules_json TEXT NOT NULL,
    total_source_frames INTEGER,
    decoded_frames INTEGER,
    processed_frames INTEGER,
    detection_count INTEGER,
    processing_duration_seconds REAL,
    mean_inference_latency_ms REAL,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES processing_runs(id),
    created_at_utc TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_frame_index INTEGER NOT NULL CHECK(source_frame_index >= 0),
    timestamp_seconds REAL NOT NULL CHECK(timestamp_seconds >= 0),
    normalized_class TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL, bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL,
    confirmation_count INTEGER NOT NULL CHECK(confirmation_count > 0),
    detection_count INTEGER NOT NULL CHECK(detection_count > 0),
    snapshot_path TEXT NOT NULL UNIQUE,
    snapshot_sha256 TEXT NOT NULL CHECK(length(snapshot_sha256) = 64),
    review_status TEXT NOT NULL DEFAULT 'new'
        CHECK(review_status IN ('new','reviewed','dismissed')),
    review_notes TEXT NOT NULL DEFAULT '',
    reviewed_at_utc TEXT,
    UNIQUE(run_id, source_frame_index)
);
CREATE INDEX IF NOT EXISTS incidents_created_idx ON incidents(created_at_utc DESC);
CREATE INDEX IF NOT EXISTS incidents_status_idx ON incidents(review_status, created_at_utc DESC);
"""


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(path, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        with db:
            yield db


def initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as db:
        # Serialize schema startup, including across separate Streamlit sessions.
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version == 1:
            return
        if (
            version != 0
            or db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('incidents', 'processing_runs')"
            ).fetchone()
        ):
            raise ValueError("Incident database schema is unsupported. Use a separate database.")
        for statement in SCHEMA.strip().split(";"):
            if statement.strip():
                db.execute(statement)
        db.execute("PRAGMA user_version = 1")

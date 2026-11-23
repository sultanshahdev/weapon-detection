import csv
import hashlib
import io
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from weapon_detection.detection.model import Detection
from weapon_detection.detection.video_processor import FrameResult
from weapon_detection.incidents.alert_engine import AlertEngine, AlertRules
from weapon_detection.incidents.incident_manager import analyze_video, persist_event
from weapon_detection.storage.database import connect
from weapon_detection.storage.evidence import EvidenceStore
from weapon_detection.storage.reports import incident_csv
from weapon_detection.storage.repository import IncidentRepository

POSITIVE = Detection(19, "Pistol", 0.82, 2, 2, 35, 28)


class FakeDetector:
    def __init__(self, positive=True):
        self.positive = positive

    def predict(self, frame):
        return (POSITIVE,) if self.positive else ()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(8):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.fixture
def storage(tmp_path):
    return IncidentRepository(tmp_path / "db.sqlite3"), EvidenceStore(tmp_path / "evidence")


def run_analysis(source, storage, tmp_path, detector):
    return analyze_video(
        source,
        detector,
        *storage,
        tmp_path,
        source_name="=test.avi",
        source_type="upload",
        model="fixture",
        image_size=416,
        rules=AlertRules(),
        stride=1,
        max_frames=8,
    )


def test_integration_persistence_evidence_and_csv(source, storage, tmp_path):
    result = run_analysis(source, storage, tmp_path, FakeDetector())
    assert result.video.stat().st_size > 0
    with result.detections_csv.open() as report:
        assert len(list(csv.DictReader(report))) == 8
    repository = IncidentRepository(tmp_path / "db.sqlite3")
    [incident] = repository.list_incidents()
    assert incident["source_frame_index"] == 2
    assert incident["run_id"] == result.run_id
    assert result.stats["detection_count"] == 8 and result.incident_count == 1
    assert repository.list_runs()[0]["status"] == "completed"
    evidence = storage[1].read(incident["snapshot_path"])
    assert hashlib.sha256(evidence).hexdigest() == incident["snapshot_sha256"]
    incident_id = incident["id"]
    repository.review(incident_id, "dismissed", "=1+1")
    [updated] = repository.list_incidents(status="dismissed", source="=test.avi", confidence=0.8)
    assert updated["confidence"] == incident["confidence"]
    assert updated["reviewed_at_utc"] is not None
    assert repository.list_incidents(status="new") == []
    assert repository.list_incidents(confidence=0.9) == []
    [row] = list(csv.DictReader(io.StringIO(incident_csv([updated]).decode())))
    assert (row["source_name"], row["review_notes"]) == ("'=test.avi", "'=1+1")


def test_blank_produces_no_incidents(source, storage, tmp_path):
    result = run_analysis(source, storage, tmp_path, FakeDetector(False))
    assert result.incident_count == result.stats["detection_count"] == 0
    assert storage[0].list_incidents() == []


def test_disk_failure_keeps_database_consistent(source, storage, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(storage[1], "save", fail)
    with pytest.raises(OSError, match="disk full"):
        run_analysis(source, storage, tmp_path, FakeDetector())
    assert storage[0].list_incidents() == []
    assert storage[0].list_runs()[0]["status"] == "failed"
    assert not list((tmp_path / "runs").iterdir())


def test_sql_failure_removes_snapshot_and_retry_is_idempotent(storage, monkeypatch):
    repository, store = storage
    rules = AlertRules(confirmation=1)
    frame = FrameResult(0, 0, (POSITIVE,), np.zeros((48, 64, 3), dtype=np.uint8), 1)
    event = AlertEngine("run", "clip", rules).observe(0, 0, frame.detections)
    with pytest.raises(sqlite3.IntegrityError):
        persist_event(event, frame, repository, store)
    assert store.read("run/run-0.jpg") is None
    repository.create_run(
        "run",
        "clip",
        "sample",
        "fixture",
        confidence=0.35,
        image_size=416,
        stride=1,
        max_frames=1,
        rules={},
    )
    persist_event(event, frame, repository, store)
    persist_event(event, frame, repository, store)
    assert len(repository.list_incidents()) == 1
    assert len(list(store.root.rglob("*.jpg"))) == 1


def test_invalid_video_marks_failed(tmp_path, storage):
    bogus = tmp_path / "bad.avi"
    bogus.write_bytes(b"invalid video")
    with pytest.raises(ValueError, match="Could not open"):
        run_analysis(bogus, storage, tmp_path, FakeDetector())
    assert storage[0].list_runs()[0]["status"] == "failed"


def test_database_transaction_rolls_back_and_preserves_existing_schema(tmp_path):
    path = tmp_path / "legacy.db"
    with connect(path) as db:
        db.execute("CREATE TABLE incidents (original INT)")
        db.execute("INSERT INTO incidents VALUES (7)")
    with pytest.raises(ValueError, match="unsupported"):
        IncidentRepository(path)
    with pytest.raises(RuntimeError), connect(path) as db:
        db.execute("UPDATE incidents SET original=10")
        raise RuntimeError("rollback")
    with connect(path) as db:
        assert db.execute("SELECT original FROM incidents").fetchone()[0] == 7


def test_evidence_rejects_bad_paths_images_and_collisions(storage, tmp_path, monkeypatch):
    _, store = storage
    frame = np.zeros((15, 15, 3), dtype=np.uint8)
    for name in ("../bad.jpg", "/tmp/bad.jpg", "nested/run/bad.jpg"):
        with pytest.raises(ValueError):
            store.read(name)
    store.root.mkdir()
    (store.root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        store.save("escape", "bad", frame)
    with pytest.raises(ValueError):
        store.save("safe", "bad", np.empty(0))
    store.save("safe", "ok", frame)
    with pytest.raises(FileExistsError):
        store.save("safe", "ok", frame)
    assert store.read("safe/missing.jpg") is None

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="disk full"):
        store.save("safe", "no-file", frame)
    assert [path.name for path in (store.root / "safe").iterdir()] == ["ok.jpg"]

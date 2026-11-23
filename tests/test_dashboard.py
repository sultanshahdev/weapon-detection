from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from weapon_detection.detection.model import Detection
from weapon_detection.storage.evidence import EvidenceStore
from weapon_detection.storage.repository import IncidentRepository
from weapon_detection.ui import monitor

APP = Path(__file__).resolve().parents[1] / "app.py"


def find(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_dashboard_upload_rerun_review_and_missing_evidence(tmp_path, monkeypatch):
    for setting, value in {
        "WD_OUTPUT_DIR": tmp_path,
        "WD_INPUT_DIR": tmp_path / "uploads",
        "WD_DATABASE_PATH": tmp_path / "db.sqlite3",
        "WD_EVIDENCE_DIR": tmp_path / "evidence",
        "WD_MODEL_PATH": tmp_path / "fixture.pt",
    }.items():
        monkeypatch.setenv(setting, str(value))
    (tmp_path / "fixture.pt").touch()
    monkeypatch.setattr(monitor, "PROJECT_ROOT", tmp_path)
    detector = SimpleNamespace(
        detector=SimpleNamespace(firearm_classes={2: "Pistol"}),
        predict=lambda image: (Detection(2, "Pistol", 0.8, 2, 2, 16, 16),),
    )
    monkeypatch.setattr(monitor, "cached_detector", lambda *args: detector)

    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 6, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(4):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception and not app.error
    assert any("No incidents match" in item.value for item in app.info)
    repository = IncidentRepository(tmp_path / "db.sqlite3")
    assert not repository.list_runs()
    app.file_uploader[0].set_value(("../fixture.avi", video.read_bytes(), "video/x-msvideo")).run()
    find(app.number_input, "Frame stride").set_value(1)
    find(app.button, "Start Analysis").click().run()
    assert not app.exception and not app.error
    [row] = repository.list_incidents()
    assert row["source_name"] == "fixture.avi"
    assert len(app.download_button) == 4
    assert not list((tmp_path / "uploads").iterdir())
    app.run()
    assert len(repository.list_runs()) == 1
    find(app.selectbox, "Review decision").set_value("dismissed")
    find(app.text_area, "Review notes").set_value("Fixture review")
    find(app.button, "Save review").click().run()
    assert not app.exception and not app.error
    [dismissed] = repository.list_incidents(status="dismissed")
    assert dismissed["review_notes"] == "Fixture review"
    EvidenceStore(tmp_path / "evidence").delete(row["snapshot_path"])
    app.run()
    assert not app.exception and not app.error
    assert any("Evidence snapshot is missing" in item.value for item in app.warning)
    find(app.selectbox, "Review status filter").set_value("new").run()
    assert any("No incidents match" in item.value for item in app.info)

from pathlib import Path

import pytest

from weapon_detection.utils.config import ConfigurationError, Settings, load_settings


def test_defaults_without_side_effects(tmp_path: Path) -> None:
    assert load_settings(tmp_path, environ={}) == Settings(
        model_path=None,
        confidence_threshold=0.5,
        input_dir=tmp_path / "inputs",
        output_dir=tmp_path / "outputs",
        database_path=tmp_path / "outputs/incidents.sqlite3",
        evidence_dir=tmp_path / "outputs/evidence",
        alert_confirmation_frames=3,
        alert_cooldown_seconds=5,
        max_upload_mb=200,
    )
    assert list(tmp_path.iterdir()) == []


def test_environment_precedence_and_path_resolution(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "WD_MODEL_PATH=models/local.pt\n"
        "WD_CONFIDENCE_THRESHOLD=0.70\n"
        "WD_OUTPUT_DIR=review\n"
        "WD_DATABASE_PATH=archive.db\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "snapshots"
    settings = load_settings(
        tmp_path, environ={"WD_CONFIDENCE_THRESHOLD": "0.80", "WD_EVIDENCE_DIR": str(evidence)}
    )
    assert settings.model_path == tmp_path / "models/local.pt"
    assert settings.confidence_threshold == 0.8
    assert settings.database_path == tmp_path / "review/archive.db"
    assert settings.evidence_dir == evidence


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("WD_CONFIDENCE_THRESHOLD", "1.01"),
        ("WD_CONFIDENCE_THRESHOLD", "nan"),
        ("WD_ALERT_CONFIRMATION_FRAMES", "0"),
        ("WD_ALERT_CONFIRMATION_FRAMES", "1.5"),
        ("WD_ALERT_COOLDOWN_SECONDS", "-1"),
        ("WD_ALERT_COOLDOWN_SECONDS", "inf"),
        ("WD_MAX_UPLOAD_MB", "201"),
        ("WD_MAX_UPLOAD_MB", "0"),
        ("WD_INPUT_DIR", " "),
        ("WD_MODEL_PATH", "model\x00.pt"),
    ],
)
def test_invalid_value_reports_setting(tmp_path: Path, setting: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=setting):
        load_settings(tmp_path, environ={setting: value})

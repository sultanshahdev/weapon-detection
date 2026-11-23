"""Read and validate local application settings without creating files."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_SERVER_UPLOAD_MB = 200
# Default per-box confidence for drawing boxes and feeding the alert rule in the dashboard
# (override with WD_CONFIDENCE_THRESHOLD). It is a display/alert setting, not tuned on any
# data, and it is NOT the validation-chosen image-level threshold (0.12, used in
# predict_app.py from outputs/ml/chosen_settings.json).
DEFAULT_BOX_CONFIDENCE = 0.50
# Temporal alert defaults (WD_ALERT_*). They were chosen by hand and NOT tuned on any data.
DEFAULT_ALERT_CONFIRMATION_FRAMES = 3
DEFAULT_ALERT_WINDOW = 5
DEFAULT_ALERT_COOLDOWN_SECONDS = 5
DEFAULT_ALERT_MAX_GAP_SECONDS = 3
DEFAULT_ALERT_RESET_MISSES = 3


class ConfigurationError(ValueError):
    """An application setting is missing a valid value."""


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path | None
    confidence_threshold: float
    input_dir: Path
    output_dir: Path
    database_path: Path
    evidence_dir: Path
    alert_confirmation_frames: int
    alert_cooldown_seconds: float
    max_upload_mb: int
    alert_window: int = 5
    alert_max_gap_seconds: float = 3
    alert_reset_misses: int = 3
    max_video_frames: int = 300


def _path(base: Path, value: str) -> Path:
    return (base / Path(value).expanduser()).resolve()


def load_settings(
    root: Path = PROJECT_ROOT,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load defaults, then .env, then the supplied or process environment."""
    root = root.resolve()
    try:
        file_values = dotenv_values(root / ".env", interpolate=False)
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError("Could not read the project .env file.") from exc
    env = os.environ if environ is None else environ

    def read(key: str, default: str) -> str:
        name = f"WD_{key}"
        value = env.get(name, file_values.get(name, default))
        if value is None:
            raise ConfigurationError(f"{name} requires a value.")
        return value.strip()

    def number(key: str, default: str, cast: type[int] | type[float]) -> int | float:
        try:
            value = cast(read(key, default))
        except ValueError as exc:
            raise ConfigurationError(f"WD_{key} must be a valid {cast.__name__}.") from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigurationError(f"WD_{key} must be finite.")
        return value

    def resolve_path(key: str, value: str, base: Path) -> Path:
        if not value:
            raise ConfigurationError(f"WD_{key} requires a nonempty path.")
        try:
            return _path(base, value)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ConfigurationError(f"WD_{key} must be a valid path.") from exc

    confidence = float(number("CONFIDENCE_THRESHOLD", str(DEFAULT_BOX_CONFIDENCE), float))
    frames = int(number("ALERT_CONFIRMATION_FRAMES", str(DEFAULT_ALERT_CONFIRMATION_FRAMES), int))
    cooldown = float(number("ALERT_COOLDOWN_SECONDS", str(DEFAULT_ALERT_COOLDOWN_SECONDS), float))
    upload = int(number("MAX_UPLOAD_MB", str(MAX_SERVER_UPLOAD_MB), int))
    window = int(number("ALERT_WINDOW", str(DEFAULT_ALERT_WINDOW), int))
    gap = float(number("ALERT_MAX_GAP_SECONDS", str(DEFAULT_ALERT_MAX_GAP_SECONDS), float))
    misses = int(number("ALERT_RESET_MISSES", str(DEFAULT_ALERT_RESET_MISSES), int))
    max_video_frames = int(number("MAX_VIDEO_FRAMES", "300", int))
    if window < frames:
        raise ConfigurationError("WD_ALERT_WINDOW must be at least WD_ALERT_CONFIRMATION_FRAMES.")
    if gap <= 0 or misses < 1 or max_video_frames < 1:
        raise ConfigurationError(
            "WD_ALERT_MAX_GAP_SECONDS, WD_ALERT_RESET_MISSES and WD_MAX_VIDEO_FRAMES "
            "must be positive."
        )
    if not 0 <= confidence <= 1:
        raise ConfigurationError("WD_CONFIDENCE_THRESHOLD must be between 0 and 1.")
    if frames < 1:
        raise ConfigurationError("WD_ALERT_CONFIRMATION_FRAMES must be positive.")
    if cooldown < 0:
        raise ConfigurationError("WD_ALERT_COOLDOWN_SECONDS must be nonnegative.")
    if not 1 <= upload <= MAX_SERVER_UPLOAD_MB:
        raise ConfigurationError(f"WD_MAX_UPLOAD_MB must be between 1 and {MAX_SERVER_UPLOAD_MB}.")

    def path(key: str, default: str, base: Path) -> Path:
        return resolve_path(key, read(key, default), base)

    model_value = read("MODEL_PATH", "")
    output_dir = path("OUTPUT_DIR", "outputs", root)
    return Settings(
        model_path=resolve_path("MODEL_PATH", model_value, root) if model_value else None,
        confidence_threshold=confidence,
        input_dir=path("INPUT_DIR", "inputs", root),
        output_dir=output_dir,
        database_path=path("DATABASE_PATH", "incidents.sqlite3", output_dir),
        evidence_dir=path("EVIDENCE_DIR", "evidence", output_dir),
        alert_confirmation_frames=frames,
        alert_cooldown_seconds=cooldown,
        max_upload_mb=upload,
        alert_window=window,
        alert_max_gap_seconds=gap,
        alert_reset_misses=misses,
        max_video_frames=max_video_frames,
    )

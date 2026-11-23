"""Write and read annotated evidence within a configured directory."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: str
    sha256: str


class EvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        if (
            len(path.parts) != 2
            or any(
                re.fullmatch(r"[a-zA-Z0-9_-]+", part) is None for part in (path.parts[0], path.stem)
            )
            or path.suffix != ".jpg"
        ):
            raise ValueError("Invalid evidence path.")
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root) or resolved != self.root / path:
            raise ValueError("Evidence path leaves the configured directory or uses a symlink.")
        return resolved

    def save(self, run_id: str, incident_id: str, frame: np.ndarray) -> Snapshot:
        relative = f"{run_id}/{incident_id}.jpg"
        target = self.resolve(relative)
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size or frame.dtype != np.uint8:
            raise ValueError("Evidence requires a nonempty uint8 BGR image.")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok or not encoded.size:
            raise OSError("Evidence JPEG encoding failed.")
        payload = encoded.tobytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent) as temporary:
            staged = Path(temporary) / "snapshot.jpg"
            if staged.write_bytes(payload) != len(payload):
                raise OSError("Evidence snapshot write was incomplete.")
            target.hardlink_to(staged)
        return Snapshot(relative, hashlib.sha256(payload).hexdigest())

    def read(self, relative: str) -> bytes | None:
        try:
            return self.resolve(relative).read_bytes()
        except FileNotFoundError:
            return None

    def delete(self, relative: str) -> None:
        self.resolve(relative).unlink(missing_ok=True)

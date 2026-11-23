"""Render a single, continuously replaced presentation display."""

import atexit
from base64 import b64encode
from collections.abc import Iterator
from contextlib import contextmanager
from html import escape
from threading import Condition, Event

import cv2
import numpy as np

from weapon_detection.detection.model import Detection

DEMO_CSS = """
<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"], footer { display:none !important; }
[data-testid="stMainBlockContainer"] { padding:0 !important; max-width:none !important; }
[data-testid="stMain"] { overflow:hidden; }
.monitor { height:100vh; height:100dvh; display:grid; box-sizing:border-box;
  grid-template-rows:auto minmax(0,1fr) auto; gap:1px; background:#26303a;
  color:#f1f5f9; font-family:system-ui,sans-serif; }
.monitor-top,.monitor-bottom { display:flex; align-items:center; justify-content:space-between;
  gap:1rem; background:#10161d; padding:.65rem clamp(.65rem,2vw,1.65rem); }
.monitor-top h1 { margin:0; padding:0; font-weight:650; font-size:clamp(.8rem,1.45vw,1.15rem);
  letter-spacing:.08em; color:#f1f5f9; }
.monitor-live,.monitor-bottom { font-family:ui-monospace,monospace;
  font-size:clamp(.65rem,1vw,.85rem);
  letter-spacing:.04em; }
.monitor-live { color:#a8eee4; text-align:right; }
.monitor-viewport { background:#080b10; min-width:0; min-height:0;
  display:grid; place-items:center; }
.monitor-viewport img { display:block; width:100%; height:100%; min-height:0; object-fit:contain; }
.monitor-viewport p { color:#d0dae5; text-align:center; margin:1rem; }
.monitor-bottom { display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); }
.monitor-source { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.monitor-fps { text-align:right; white-space:nowrap; }
.monitor-hit { color:#ff8588; }
.monitor-clear { color:#a8eee4; }
@media (max-width:600px) { .monitor-bottom { grid-template-columns:minmax(0,1fr) auto; }
  .monitor-state { grid-row:2; grid-column:1 / -1; text-align:center; } }
</style>
"""


def detection_status(detections: tuple[Detection, ...]) -> tuple[str, str]:
    if any(item.normalized_class == "firearm" for item in detections):
        return "FIREARM DETECTED", "monitor-hit"
    return "NO FIREARM DETECTED IN CURRENT FRAME", "monitor-clear"


def encoded_frame(frame: np.ndarray) -> str:
    """Encode BGR directly with OpenCV. A browser decodes the resulting RGB JPEG."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 720 / height)
    if scale < 1:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Could not encode the current display frame.")
    return b64encode(jpeg).decode("ascii")


def screen(
    *,
    source: str = "",
    detections: tuple[Detection, ...] = (),
    image: str = "",
    fps: float | None = None,
    message: str = "",
) -> str:
    status, color = detection_status(detections) if image else ("", "")
    picture = (
        f'<img src="data:image/jpeg;base64,{image}" alt="Current analyzed video frame">'
        if image
        else f"<p>{escape(message)}</p>"
    )
    return f"""
      <section class="monitor">
        <div class="monitor-top">
          <h1>WEAPON DETECTION — LIVE DEMONSTRATION</h1>
          <span class="monitor-live">LIVE MODEL INFERENCE</span>
        </div>
        <div class="monitor-viewport">{picture}</div>
        <div class="monitor-bottom">
          <span class="monitor-source">{escape(source)}</span>
          <strong class="monitor-state {color}">{status}</strong>
          <span class="monitor-fps">{f"{fps:.1f} FPS" if fps is not None else ""}</span>
        </div>
      </section>
    """


# Streamlit's runner threads can outlive server shutdown. Wait for OpenCV and
# PyTorch to leave the playback context before the interpreter tears them down.
_shutdown = Event()
_workers = Condition()
_worker_count = 0


def accepting_frames() -> bool:
    return not _shutdown.is_set()


@contextmanager
def playback_worker() -> Iterator[None]:
    global _worker_count
    with _workers:
        _worker_count += 1
    try:
        yield
    finally:
        with _workers:
            _worker_count -= 1
            _workers.notify_all()


@atexit.register
def _drain_playback_workers() -> None:
    _shutdown.set()
    with _workers:
        _workers.wait_for(lambda: _worker_count == 0, timeout=15)

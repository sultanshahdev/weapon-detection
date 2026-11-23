"""Draw ground-truth (green) and model (red) boxes on a video and save a browser-ready MP4.

OpenCV's own "mp4v" output does not play in browsers, so frames are piped to ffmpeg and
encoded as H.264 (yuv420p, +faststart). ffmpeg is taken from PATH, or from the optional
imageio-ffmpeg package if it is installed.
"""

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from weapon_detection.data.mendeley_videos import GtBox
from weapon_detection.detection.model import Detector

# Default per-box confidence for the red boxes. This is a DISPLAY setting only: 0.25 is the
# Ultralytics default and it was NOT tuned or chosen on the Mendeley footage.
DEFAULT_CONFIDENCE = 0.25

GT_COLOR = (0, 200, 0)  # BGR green: dataset annotations
MODEL_COLOR = (0, 0, 255)  # BGR red: model predictions
FONT = cv2.FONT_HERSHEY_SIMPLEX

# One model box: x1, y1, x2, y2, confidence.
ModelBox = tuple[float, float, float, float, float]


def ffmpeg_path() -> str | None:
    """ffmpeg on PATH first, then the binary bundled with imageio-ffmpeg, else None."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def cache_name(video_id: str, show_gt: bool, checkpoint: str | None, conf: float) -> str:
    """File name that changes whenever an overlay setting changes."""
    gt = "gt" if show_gt else "nogt"
    model = f"model-{Path(checkpoint).stem}-conf{conf:.2f}" if checkpoint else "nomodel"
    return f"{video_id}__{gt}__{model}.mp4"


def _label(image: np.ndarray, text: str, x: int, y: int, color) -> None:
    cv2.putText(image, text, (x, max(12, y)), FONT, 0.5, color, 1, cv2.LINE_AA)


def draw_overlays(
    frame: np.ndarray,
    frame_index: int,
    gt_boxes: list[GtBox] | None,
    model_boxes: list[ModelBox] | None,
) -> np.ndarray:
    """Return a copy of frame with boxes, a GT/MODEL legend and the frame number.

    gt_boxes=None or model_boxes=None means that overlay is switched off (the legend
    then only names the sources that are on).
    """
    image = frame.copy()
    for x1, y1, x2, y2, name in gt_boxes or []:
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), GT_COLOR, 2)
        _label(image, f"GT {name}", round(x1), round(y1) - 5, GT_COLOR)
    for x1, y1, x2, y2, conf in model_boxes or []:
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), MODEL_COLOR, 2)
        # Text under the box so it never covers the green GT label above it.
        _label(image, f"firearm {conf:.2f}", round(x1), round(y2) + 14, MODEL_COLOR)
    # Legend (top-left) on a dark strip so it is readable on any background.
    legend = []
    if gt_boxes is not None:
        legend.append(("GT = dataset annotation", GT_COLOR))
    if model_boxes is not None:
        legend.append(("MODEL = my detector", MODEL_COLOR))
    for row, (text, color) in enumerate(legend):
        y = 18 + row * 18
        cv2.rectangle(image, (0, y - 14), (210, y + 4), (0, 0, 0), -1)
        _label(image, text, 4, y, color)
    # Frame number (bottom-right), 0-based like the frame browser.
    height, width = image.shape[:2]
    text = f"frame {frame_index}"
    cv2.rectangle(image, (width - 110, height - 22), (width, height), (0, 0, 0), -1)
    _label(image, text, width - 104, height - 6, (255, 255, 255))
    return image


def model_boxes_for(detector: Detector, frame: np.ndarray, conf: float) -> list[ModelBox]:
    """Run the detector on one frame and keep boxes with confidence >= conf."""
    found = detector.predict(frame)
    return [(d.x1, d.y1, d.x2, d.y2, d.confidence) for d in found if d.confidence >= conf]


def render_annotated_video(
    video_path: Path,
    boxes_gt: dict[int, list[GtBox]] | None,
    predictor: Detector | None,
    out_path: Path,
    conf: float = DEFAULT_CONFIDENCE,
    progress: Callable[[float], None] | None = None,
    cached_boxes: dict[int, list] | None = None,
) -> dict:
    """Write an H.264 MP4 with overlays to out_path and a JSON sidecar next to it.

    boxes_gt=None hides the green overlay; predictor=None hides the red overlay.
    cached_boxes (precomputed {frame: [[x1, y1, x2, y2, conf], ...]}) replaces the
    predictor: boxes are only filtered by conf and the model is not called.
    The sidecar (<out_path>.json) stores the model boxes per frame, so the app can show
    statistics and single frames later without running the model again.
    Returns the sidecar contents.
    """
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found. Install it or run: uv add imageio-ffmpeg")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(out_path.stem + ".partial.mp4")
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
        "-i", "-",
        # H.264 needs even width and height; this is a no-op for 640x480.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temp_path),
    ]  # fmt: skip
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    model_boxes: dict[int, list[ModelBox]] = {}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gt = None if boxes_gt is None else boxes_gt.get(index, [])
            if cached_boxes is not None:
                found = [tuple(b) for b in cached_boxes.get(index, []) if b[4] >= conf]
            elif predictor is not None:
                found = model_boxes_for(predictor, frame, conf)
            else:
                found = None
            if found:
                model_boxes[index] = found
            encoder.stdin.write(draw_overlays(frame, index, gt, found).tobytes())
            index += 1
            if progress is not None:
                progress(min(1.0, index / total))
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
        os.replace(temp_path, out_path)
    finally:
        capture.release()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        temp_path.unlink(missing_ok=True)  # managed temp file: removed on failure too
    sidecar = {
        "frames": index,
        "fps": fps,
        "model_on": predictor is not None or cached_boxes is not None,
        "conf": conf,
        # JSON keys must be strings; frame indices are 0-based.
        "model_boxes": {str(k): [list(b) for b in v] for k, v in model_boxes.items()},
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar))
    return sidecar


def check_playable(path: Path) -> dict:
    """Read the file back with OpenCV: frame count and codec tag (expect avc1/h264)."""
    capture = cv2.VideoCapture(str(path))
    try:
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
        frames = 0
        while capture.grab():
            frames += 1
    finally:
        capture.release()
    return {"frames": frames, "codec": fourcc.decode(errors="replace").strip("\x00")}

"""Automatic CPU inference presentation screen."""

import logging
import os
from collections import deque
from contextlib import closing
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import streamlit as st
from streamlit import config, runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

from weapon_detection.detection.model import default_checkpoint
from weapon_detection.detection.playlist import playlist_frames
from weapon_detection.ui.demo import (
    DEMO_CSS,
    accepting_frames,
    encoded_frame,
    playback_worker,
    screen,
)
from weapon_detection.ui.monitor import VIDEO_SUFFIXES, approved_files, cached_detector
from weapon_detection.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
# Default per-box confidence for this screen (override with WD_DEMO_CONFIDENCE). A display
# setting, not tuned on any data; the status line reflects boxes at this confidence.
DEMO_BOX_CONFIDENCE = 0.35


@st.cache_resource(scope="session", show_spinner=False)
def session_guard() -> Lock:
    return Lock()


def main() -> None:
    st.set_page_config(page_title="Weapon Detection — Live Demonstration", layout="wide")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    # This stand-alone screen replaces a live JPEG at one delta path. Keep those
    # messages out of Streamlit's cross-run cache, which ages only on script completion.
    config.set_option("global.minCachedMessageSize", 10_000_000.0)
    st.html(DEMO_CSS)
    display = st.empty()
    playlist = Path(os.getenv("WD_DEMO_PLAYLIST", str(PROJECT_ROOT / "datasets/demo_playlist")))
    clips = approved_files(playlist, VIDEO_SUFFIXES)
    if not clips:
        display.html(
            screen(message="Place MP4, AVI, MOV, MKV or WebM clips in datasets/demo_playlist/.")
        )
        return
    # Fine-tuned model if present, otherwise the OIV7 fallback (same rule as every app).
    checkpoint = default_checkpoint(PROJECT_ROOT)
    if checkpoint is None:
        display.html(screen(message="Place a checkpoint in models/ (see README)."))
        return
    context = get_script_run_ctx()

    def active() -> bool:
        return (
            accepting_frames()
            and context is not None
            and runtime.exists()
            and runtime.get_instance().is_active_session(context.session_id)
        )

    try:
        image_size = int(os.getenv("WD_DEMO_IMGSZ", "416"))
        stride = int(os.getenv("WD_DEMO_STRIDE", "1"))
        confidence = float(os.getenv("WD_DEMO_CONFIDENCE", str(DEMO_BOX_CONFIDENCE)))
        if image_size < 32 or image_size % 32 or stride < 1 or not 0 <= confidence <= 1:
            raise ValueError("Invalid demo inference settings.")
    except ValueError:
        display.html(
            screen(
                message=(
                    "Check WD_DEMO_IMGSZ, WD_DEMO_STRIDE and WD_DEMO_CONFIDENCE before restarting."
                )
            )
        )
        return

    with session_guard(), playback_worker():
        display.html(screen(message="Loading the local YOLO checkpoint on CPU…"))
        try:
            revision = checkpoint.stat()
            detector = cached_detector(
                checkpoint,
                (revision.st_mtime_ns, revision.st_size, revision.st_ino),
                confidence,
                image_size,
            )
        except (ValueError, OSError, RuntimeError):
            logger.exception("Demo checkpoint could not load")
            display.html(screen(message=f"Check the local checkpoint at {checkpoint.name}."))
            return

        timestamps: deque[float] = deque(maxlen=16)
        try:
            with closing(playlist_frames(clips, detector, stride=stride, active=active)) as frames:
                for path, frame in frames:
                    try:
                        jpeg = encoded_frame(frame.annotated_frame)
                    except (ValueError, cv2.error) as exc:
                        logger.warning(
                            "Skipping demo frame %s in %s: %s", frame.frame_index, path, exc
                        )
                        continue
                    timestamps.append(perf_counter())
                    span = timestamps[-1] - timestamps[0]
                    fps = (len(timestamps) - 1) / span if span > 0 else None
                    display.html(
                        screen(
                            source=path.name,
                            detections=frame.detections,
                            image=jpeg,
                            fps=fps,
                        )
                    )
        except (ValueError, RuntimeError, OSError, cv2.error):
            logger.exception("Demo playlist stopped")
            display.html(
                screen(message="No playable clips remain. Check the playlist files and codecs.")
            )


if __name__ == "__main__":
    main()

"""Stream one analyzed frame at a time from an ordered local playlist."""

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from functools import partial
from pathlib import Path

import cv2

from weapon_detection.detection.inference import annotate_frame
from weapon_detection.detection.model import Detector
from weapon_detection.detection.video_processor import FrameResult, process_video

logger = logging.getLogger(__name__)
annotate_demo = partial(annotate_frame, color=(30, 35, 255), show_class_name=True)


def playlist_frames(
    paths: Sequence[Path],
    detector: Detector,
    *,
    stride: int,
    active: Callable[[], bool],
) -> Iterator[tuple[Path, FrameResult]]:
    # Remove failed clips for this session so corruption cannot trigger a rapid retry loop.
    available = list(paths)
    while available and active():
        for path in tuple(available):
            if not active():
                return
            try:
                with closing(
                    process_video(path, detector, frame_stride=stride, annotator=annotate_demo)
                ) as frames:
                    while active():
                        frame = next(frames, None)
                        if frame is None:
                            break
                        yield path, frame
                    else:
                        return
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                logger.warning("Skipping demo clip %s: %s", path.name, exc)
                available.remove(path)
    if active():
        raise ValueError(
            "No playable clips remain. Check the files and codecs in the demo playlist."
        )

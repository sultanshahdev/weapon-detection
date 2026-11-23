"""Annotate BGR video frames without modifying the original image."""

import cv2
import numpy as np

from weapon_detection.detection.model import Detection


def annotate_frame(
    frame: np.ndarray,
    detections: tuple[Detection, ...],
    *,
    color: tuple[int, int, int] = (0, 200, 255),
    show_class_name: bool = False,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    for detection in detections:
        left, right = (min(width - 1, max(0, round(x))) for x in (detection.x1, detection.x2))
        top, bottom = (min(height - 1, max(0, round(y))) for y in (detection.y1, detection.y2))
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        label = detection.class_name if show_class_name else "Firearm"
        cv2.putText(
            image,
            f"{label} {detection.confidence:.2f}",
            (left, max(12, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return image

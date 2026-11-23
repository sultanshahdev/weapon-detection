import numpy as np

from weapon_detection.detection.inference import annotate_frame
from weapon_detection.detection.model import Detection


def test_annotation_clips_boxes_and_preserves_input() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    detection = Detection(8, "Pistol", 0.87, -10, -30, 80, 70)
    output = annotate_frame(frame, (detection,))
    assert output.shape == frame.shape
    assert output.any()
    assert not frame.any()
    assert np.array_equal(frame, annotate_frame(frame, ()))

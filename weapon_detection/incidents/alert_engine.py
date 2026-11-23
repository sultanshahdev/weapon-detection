"""Source-level temporal confirmation for one prerecorded run."""

from collections import deque
from dataclasses import dataclass
from math import isfinite

from weapon_detection.detection.model import Detection


@dataclass(frozen=True, slots=True)
class AlertRules:
    confidence: float = 0.35
    confirmation: int = 3
    window: int = 5
    max_gap: float = 3
    reset_misses: int = 3
    cooldown: float = 10

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Alert confidence must be between 0 and 1.")
        if not 1 <= self.confirmation <= self.window or self.reset_misses < 1:
            raise ValueError(
                "Require positive counts and a window at least as large as confirmation."
            )
        if not isfinite(self.max_gap) or self.max_gap <= 0:
            raise ValueError("Maximum observation gap must be finite and positive.")
        if not isfinite(self.cooldown) or self.cooldown < 0:
            raise ValueError("Cooldown must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class ConfirmedEvent:
    id: str
    run_id: str
    source_name: str
    frame_index: int
    timestamp: float
    detection: Detection
    confirmation_count: int
    detection_count: int


class AlertEngine:
    def __init__(self, run_id: str, source_name: str, rules: AlertRules) -> None:
        self.run_id = run_id
        self.source_name = source_name
        self.rules = rules
        self._window: deque[bool] = deque(maxlen=rules.window)
        self._previous_frame = -1
        self._previous_time: float | None = None
        self._emitted_time: float | None = None
        self._active = False
        self._misses = 0

    def observe(
        self, frame_index: int, timestamp: float, detections: tuple[Detection, ...]
    ) -> ConfirmedEvent | None:
        if (
            frame_index <= self._previous_frame
            or not isfinite(timestamp)
            or timestamp < 0
            or (self._previous_time is not None and timestamp < self._previous_time)
        ):
            raise ValueError("Observations need increasing frame indices and ordered finite times.")
        if self._previous_time is not None and timestamp - self._previous_time > self.rules.max_gap:
            # An unobserved interval interrupts candidate confirmation. It is not evidence
            # of absence and must not rearm an already active episode.
            self._window.clear()
            self._misses = 0
        self._previous_frame = frame_index
        self._previous_time = timestamp
        qualifying = tuple(
            detection
            for detection in detections
            if detection.normalized_class == "firearm"
            and self.rules.confidence <= detection.confidence <= 1
        )
        self._misses = 0 if qualifying else self._misses + 1
        if self._misses >= self.rules.reset_misses:
            self._active = False
            self._window.clear()
        self._window.append(bool(qualifying))
        count = sum(self._window)
        if self._active or not qualifying or count < self.rules.confirmation:
            return None
        if self._emitted_time is not None and timestamp - self._emitted_time < self.rules.cooldown:
            return None
        self._active = True
        self._emitted_time = timestamp
        return ConfirmedEvent(
            self.event_id(frame_index),
            self.run_id,
            self.source_name,
            frame_index,
            timestamp,
            max(qualifying, key=lambda detection: detection.confidence),
            count,
            len(qualifying),
        )

    def event_id(self, frame_index: int) -> str:
        return f"{self.run_id}-{frame_index}"

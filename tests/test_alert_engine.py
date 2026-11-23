import pytest

from weapon_detection.detection.model import Detection
from weapon_detection.incidents.alert_engine import AlertEngine, AlertRules

POSITIVE = Detection(19, "Pistol", 0.82, 2, 2, 9, 9)
LOW = Detection(19, "Pistol", 0.2, 2, 2, 9, 9)


def test_window_counts_processed_observations_and_multiple_boxes_once():
    engine = AlertEngine("run", "source", AlertRules())
    assert engine.observe(0, 0, (POSITIVE, POSITIVE)) is None
    assert engine.observe(2, 0.5, (POSITIVE,)) is None
    assert engine.observe(4, 1, (LOW,)) is None
    event = engine.observe(6, 1.5, (POSITIVE, POSITIVE))
    assert (event.id, event.frame_index, event.timestamp) == ("run-6", 6, 1.5)
    assert (event.confirmation_count, event.detection_count) == (3, 2)


def test_isolated_hit_expires_and_gap_resets_candidate():
    engine = AlertEngine("run", "source", AlertRules(confirmation=2, window=3))
    for index, detections in enumerate(((POSITIVE,), (), (), (POSITIVE,))):
        assert engine.observe(index, index, detections) is None
    assert engine.observe(4, 10, (POSITIVE,)) is None
    assert engine.observe(5, 11, (POSITIVE,)) is not None


def test_active_episode_requires_observed_absence_and_cooldown():
    rules = AlertRules(confirmation=2, window=3, reset_misses=2, cooldown=10)
    engine = AlertEngine("run", "source", rules)
    sequence = [
        (0, True),
        (1, True),
        (2, False),
        (3, False),
        (4, True),
        (5, True),
        (8, True),
        (11, True),
        (30, True),
        (31, True),
        (32, False),
        (33, False),
        (34, True),
        (35, True),
    ]
    events = [
        event
        for index, (timestamp, positive) in enumerate(sequence)
        if (event := engine.observe(index, timestamp, (POSITIVE,) if positive else ()))
    ]
    assert [event.timestamp for event in events] == [1, 11, 35]
    fresh = AlertEngine("second", "source", rules)
    assert fresh.observe(0, 0, (POSITIVE,)) is None
    assert fresh.observe(1, 1, (POSITIVE,)).run_id == "second"


@pytest.mark.parametrize("index,timestamp", [(0, 2), (1, 0), (1, float("nan")), (1, -1)])
def test_rejects_invalid_or_out_of_order_observations(index, timestamp):
    engine = AlertEngine("run", "source", AlertRules())
    engine.observe(0, 1, ())
    with pytest.raises(ValueError, match="ordered finite"):
        engine.observe(index, timestamp, ())


@pytest.mark.parametrize(
    "settings",
    [
        {"window": 2},
        {"confirmation": 0},
        {"cooldown": -1},
        {"max_gap": 0},
        {"reset_misses": 0},
        {"confidence": 2},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        AlertRules(**settings)

"""Tests for weapon_detection.ml.classical using a small synthetic feature matrix."""

import numpy as np
import pytest

from weapon_detection.ml.classical import (
    best_threshold,
    compute_metrics,
    firearm_scores,
    fit_and_choose,
)


def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """60 train, 20 val, 20 test rows; feature 0 is informative. Test rows are NaN."""
    rng = np.random.default_rng(42)
    y = np.array([1] * 70 + [0] * 30)
    rng.shuffle(y)
    x = rng.normal(size=(100, 3)) + y[:, None] * 2.0
    split = np.array(["train"] * 60 + ["val"] * 20 + ["test"] * 20)
    x[split == "test"] = np.nan  # any model that touched test rows would crash
    return x, y, split


def test_scaler_is_fitted_on_train_only() -> None:
    x, y, split = synthetic_data()
    pipeline, chosen = fit_and_choose("logreg", x, y, split)  # NaN test rows: never read
    scaler = pipeline.named_steps["scaler"]
    assert scaler.mean_ == pytest.approx(x[split == "train"].mean(axis=0))
    assert scaler.n_samples_seen_ == 60
    assert chosen["threshold"] is not None


def test_random_forest_never_reads_test_rows() -> None:
    x, y, split = synthetic_data()
    pipeline, chosen = fit_and_choose("random_forest", x, y, split)
    assert pipeline.named_steps["model"].n_features_in_ == 3
    assert 0 < chosen["threshold"] < 1


def test_baseline_predicts_only_majority_class() -> None:
    x, y, split = synthetic_data()
    pipeline, chosen = fit_and_choose("baseline", x, y, split)
    majority = int(np.bincount(y[split == "train"]).argmax())
    predictions = pipeline.predict(x[split == "val"])
    assert set(predictions) == {majority}
    assert chosen["threshold"] is None
    assert set(firearm_scores(pipeline, x[split == "val"])) == {float(majority)}


def test_threshold_uses_only_given_validation_scores() -> None:
    y_val = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, f1 = best_threshold(y_val, scores)
    assert f1 == pytest.approx(1.0)
    assert 0.3 < threshold <= 0.7
    # The search has no access to anything else: same inputs -> same answer.
    assert best_threshold(y_val.copy(), scores.copy()) == (threshold, f1)


def test_metrics_match_hand_computed_example() -> None:
    # TP=2, FN=1, FP=1, TN=1
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.4, 0.3, 0.6])
    m = compute_metrics(y_true, y_pred, scores)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["neg_precision"] == pytest.approx(1 / 2)
    assert m["neg_recall"] == pytest.approx(1 / 2)
    assert m["neg_f1"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    # ROC-AUC by hand: positive/negative score pairs ranked correctly = 5 of 6.
    assert m["roc_auc"] == pytest.approx(5 / 6)

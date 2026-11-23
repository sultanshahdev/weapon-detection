"""Tests for compare.py helpers (synthetic prediction CSVs) and predict.py (fake models)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from weapon_detection.ml import predict
from weapon_detection.ml.classical import compute_metrics
from weapon_detection.ml.compare import (
    METRIC_COLUMNS,
    check_consistency,
    final_table,
    markdown_table,
    requirement_rows,
)

RUN = {
    "key": "fake_model",
    "model": "fake",
    "feature_set": None,
    "name": "Fake model",
    "family": "linear model",
    "main": True,
}


def write_synthetic(tmp_path):
    """A predictions CSV and the matching metrics table (rounded like the real one)."""
    rng = np.random.default_rng(42)
    tables, rows = [], []
    for split in ["val", "test"]:
        y_true = rng.integers(0, 2, size=40)
        score = rng.random(40)
        y_pred = (score >= 0.5).astype(int)
        tables.append(
            pd.DataFrame(
                {
                    "image": [f"{split}{i}.jpg" for i in range(40)],
                    "split": split,
                    "y_true": y_true,
                    "score": score,
                    "y_pred": y_pred,
                }
            )
        )
        metrics = compute_metrics(y_true, y_pred, score)
        rows.append(
            {
                "model": "fake",
                "feature_set": "x",
                "split": split,
                "n_images": 40,
                "threshold": 0.5,
                "predict_ms_per_image": 1.0,
                **metrics,
            }
        )
    pd.concat(tables).to_csv(tmp_path / "fake_model_predictions.csv", index=False)
    return pd.DataFrame(rows).round(4)


def test_consistency_check_passes_and_catches_changes(tmp_path) -> None:
    metrics = write_synthetic(tmp_path)
    assert check_consistency(metrics, [RUN], results_dir=tmp_path) == []
    metrics.loc[1, "macro_f1"] += 0.01  # pretend the table was edited by hand
    problems = check_consistency(metrics, [RUN], results_dir=tmp_path)
    assert len(problems) == 1 and "macro_f1" in problems[0]


def test_final_table_and_markdown(tmp_path) -> None:
    metrics = write_synthetic(tmp_path)
    table = final_table(metrics, [RUN], "test")
    assert list(table["model"]) == ["Fake model"]
    assert table.loc[0, "main_model"] == "yes"
    assert set(METRIC_COLUMNS) <= set(table.columns)
    text = markdown_table(table)
    assert text.startswith("| model |") and "Fake model" in text


def test_requirement_rows_are_strict() -> None:
    facts = {
        "n_images": 9425,
        "n_handcrafted": 10,
        "n_hog": 1764,
        "has_baseline": True,
        "trained_model_families": {"LR": "linear model", "RF": "tree ensemble"},
        "has_confusion": True,
        "has_leakage_audit": True,
        "has_eda": True,
        "has_error_tables": True,
        "error_explanations_written": False,
        "gui_smoke_ok": False,
        "notebooks_executed": {"a": True, "b": False},
    }
    status = {r["requirement"]: r["status"] for r in requirement_rows(facts)}
    assert status[">= 500 images"] == "met"
    assert status[">= 3 models from >= 2 families (>= 1 classical)"] == "partly"
    assert status["error analysis (5-10 wrong predictions)"] == "partly"
    assert status["GUI that takes audience input and predicts live"] == "not yet"
    assert status["documented notebooks"] == "partly"
    facts["trained_model_families"]["YOLO"] = "CNN object detector"
    status = {r["requirement"]: r["status"] for r in requirement_rows(facts)}
    assert status[">= 3 models from >= 2 families (>= 1 classical)"] == "met"


class FakePipeline:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        assert x.shape == (1, 10 + 1764)  # handcrafted + HOG, like training
        return np.array([[0.3, 0.7]])

    def predict(self, x):
        return np.array([1])


def test_classical_predict_uses_threshold() -> None:
    image = np.random.default_rng(0).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    result = predict.classical_predict(FakePipeline(), image, "handcrafted_hog", 0.8)
    assert result["score"] == 0.7 and result["is_firearm"] is False  # 0.7 < 0.8
    result = predict.classical_predict(FakePipeline(), image, "handcrafted_hog", 0.5)
    assert result["is_firearm"] is True
    assert predict.label_text(True) == "firearm likely"
    assert predict.label_text(False) == "no firearm detected"


class FakeYolo:
    names = {0: "firearm"}

    def predict(self, image, **kwargs):
        boxes = SimpleNamespace(
            xyxy=np.array([[5, 5, 30, 30], [10, 10, 40, 40]]),
            cls=np.array([0, 0]),
            conf=np.array([0.6, 0.9]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_yolo_predict_with_fake_model() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    result = predict.yolo_predict(FakeYolo(), image, threshold=0.95, imgsz=640)
    assert result["score"] == 0.9 and result["is_firearm"] is False
    assert result["image_rgb"].shape == (50, 50, 3)
    assert result["image_rgb"].sum() > 0  # boxes were drawn


def test_decode_image_rejects_garbage() -> None:
    assert predict.decode_image(b"not an image") is None
    assert predict.decode_image(b"") is None


def test_model_menu_hides_missing_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(predict, "CLASSICAL_DIR", tmp_path)
    monkeypatch.setattr(predict, "YOLO_MODELS", {"Fine-tuned": ("ft", tmp_path / "none.pt")})
    (tmp_path / "logreg_handcrafted_hog.joblib").write_text("x")
    names, notes = predict.model_menu({"logreg_handcrafted_hog": {"threshold": 0.5}})
    assert names == ["Logistic Regression (HOG)"]
    assert any("none.pt" in note for note in notes)

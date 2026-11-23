"""Prediction helpers for the audience GUI (predict_app.py). No Streamlit code here.

The classical models use features_from_array(), the SAME function used for training.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from weapon_detection.ml.common import REPO_ROOT
from weapon_detection.ml.features import features_from_array
from weapon_detection.ml.yolo_eval import MIN_CONFIDENCE, firearm_class_ids, max_firearm_score

CLASSICAL_DIR = REPO_ROOT / "models" / "classical"

# GUI name -> (settings/model key, feature set or checkpoint path)
CLASSICAL_MODELS = {
    "Majority baseline": ("baseline_handcrafted", "handcrafted"),
    "Logistic Regression (HOG)": ("logreg_handcrafted_hog", "handcrafted_hog"),
    "Random Forest (HOG)": ("random_forest_handcrafted_hog", "handcrafted_hog"),
}
YOLO_MODELS = {
    "YOLOv8n pretrained (Open Images)": (
        "yolov8n_oiv7_pretrained",
        REPO_ROOT / "models" / "yolov8n-oiv7.pt",
    ),
    "YOLOv8n fine-tuned": ("yolov8n_finetuned", REPO_ROOT / "models" / "firearm_yolov8n_best.pt"),
}


def label_text(is_firearm: bool) -> str:
    return "firearm likely" if is_firearm else "no firearm detected"


def decode_image(data: bytes) -> np.ndarray | None:
    """Bytes of a jpg/png -> BGR image, or None if it cannot be decoded."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def model_menu(settings: dict) -> tuple[list[str], list[str]]:
    """Models that can be offered, and one-line notes for those that cannot."""
    names, notes = [], []
    for name, (key, _) in CLASSICAL_MODELS.items():
        if (CLASSICAL_DIR / f"{key}.joblib").exists() and key in settings:
            names.append(name)
        else:
            notes.append(f"{name}: model file or settings missing (run classical.py).")
    for name, (key, path) in YOLO_MODELS.items():
        if not Path(path).exists():
            notes.append(f"{name}: checkpoint {Path(path).name} not found.")
        elif key not in settings:
            notes.append(f"{name}: not evaluated yet (run yolo_eval.py to set its threshold).")
        else:
            names.append(name)
    return names, notes


def feature_vector(image_bgr: np.ndarray, feature_set: str) -> np.ndarray:
    """One-row feature matrix, built exactly like the training features."""
    handcrafted, hog = features_from_array(image_bgr)
    if feature_set == "handcrafted":
        return np.array([handcrafted], dtype=np.float32)
    return np.hstack([np.array(handcrafted, dtype=np.float32), hog.astype(np.float32)])[None, :]


def classical_predict(pipeline, image_bgr: np.ndarray, feature_set: str, threshold) -> dict:
    """Score = probability of 'firearm'. The baseline (threshold None) uses predict()."""
    start = time.perf_counter()
    x = feature_vector(image_bgr, feature_set)
    column = list(pipeline.classes_).index(1)
    score = float(pipeline.predict_proba(x)[0, column])
    is_firearm = bool(pipeline.predict(x)[0]) if threshold is None else score >= threshold
    ms = (time.perf_counter() - start) * 1000
    return {"score": score, "is_firearm": is_firearm, "ms": ms, "image_rgb": None}


def draw_firearm_boxes(image_bgr: np.ndarray, result, firearm_ids: list[int]) -> np.ndarray:
    """Copy of the image with only the firearm boxes drawn; returned as RGB."""
    canvas = image_bgr.copy()
    boxes = zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    for (x1, y1, x2, y2), class_id, conf in boxes:
        if int(class_id) in firearm_ids:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(
                canvas,
                f"firearm {conf:.2f}",
                (int(x1), max(15, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def yolo_predict(model, image_bgr: np.ndarray, threshold: float, imgsz: int) -> dict:
    """Same scoring as yolo_eval.py: highest firearm-box confidence (conf >= 0.05)."""
    firearm_ids = firearm_class_ids(model.names)
    start = time.perf_counter()
    result = model.predict(
        image_bgr, conf=MIN_CONFIDENCE, imgsz=imgsz, device="cpu", verbose=False
    )[0]
    score = max_firearm_score(result, firearm_ids)
    ms = (time.perf_counter() - start) * 1000
    return {
        "score": score,
        "is_firearm": score >= threshold,
        "ms": ms,
        "image_rgb": draw_firearm_boxes(image_bgr, result, firearm_ids),
    }

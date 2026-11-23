"""Evaluate ANY YOLO checkpoint as an image classifier on the SAME val/test images.

Run with, for example:
    uv run python -m weapon_detection.ml.yolo_eval --checkpoint models/yolov8n-oiv7.pt \
        --name yolov8n_oiv7_pretrained --label "YOLOv8n pretrained (Open Images, not fine-tuned)"

Protocol (same as the classical models):
  * image score = highest confidence among FIREARM boxes (0 if there is none), conf >= 0.05;
  * threshold chosen on VAL by macro-F1 and saved to chosen_settings.json BEFORE test;
  * test evaluated once; rows added to results/classical_metrics.csv (image-level only).
For a fine-tuned checkpoint (class 0 = "firearm") Ultralytics detection metrics on the test
split (mAP50, mAP50-95, precision, recall) go to a SEPARATE file: <name>_detection_metrics.json.
"""

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weapon_detection.ml.classical import (
    RESULTS_DIR,
    best_threshold,
    compute_metrics,
    error_analysis,
    save_confusion_matrix,
    save_metric_rows,
    save_settings,
)
from weapon_detection.ml.common import FIGURE_DIR, MANIFEST_PATH, REPO_ROOT
from weapon_detection.ml.features import load_features

PRETRAINED_FIREARM_NAMES = {"handgun", "rifle", "shotgun"}  # Open Images V7 class names
MIN_CONFIDENCE = 0.05
YOLO_DATA_YAML = REPO_ROOT / "outputs" / "yolo_firearm" / "data.yaml"


def firearm_class_ids(names: dict[int, str]) -> list[int]:
    """Fine-tuned model: the 'firearm' class. Open Images model: Handgun, Rifle, Shotgun."""
    fine_tuned = [i for i, n in names.items() if n.lower() == "firearm"]
    if fine_tuned:
        return fine_tuned
    pretrained = [i for i, n in names.items() if n.lower() in PRETRAINED_FIREARM_NAMES]
    if not pretrained:
        raise ValueError("Checkpoint has no firearm classes")
    return pretrained


def max_firearm_score(result, firearm_ids: list[int]) -> float:
    """Highest confidence among boxes whose class is a firearm; 0.0 if there are none."""
    classes = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    scores = [c for k, c in zip(classes, confidences) if int(k) in firearm_ids]
    return max(scores, default=0.0)


def freeze_threshold(key: str, y_val: np.ndarray, val_scores: np.ndarray, extra: dict) -> float:
    """Pick the threshold on VALIDATION data only and save it before any test prediction."""
    threshold, val_f1 = best_threshold(y_val, val_scores)
    save_settings({key: {"threshold": threshold, "val_macro_f1_at_threshold": val_f1, **extra}})
    return threshold


def score_images(model, paths: list[str], imgsz: int, firearm_ids: list[int]):
    """Return (scores, milliseconds per image) for a list of image paths, one image at a time."""
    scores = []
    start = time.perf_counter()
    for path in paths:
        result = model.predict(
            str(REPO_ROOT / path), conf=MIN_CONFIDENCE, imgsz=imgsz, device="cpu", verbose=False
        )[0]
        scores.append(max_firearm_score(result, firearm_ids))
    ms_per_image = (time.perf_counter() - start) * 1000 / len(paths)
    return np.array(scores), ms_per_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-level evaluation of a YOLO checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="short key used in file names")
    parser.add_argument("--label", default=None, help="model name shown in the metrics table")
    parser.add_argument("--imgsz", type=int, default=None, help="default: checkpoint's imgsz")
    parser.add_argument("--skip-detection-val", action="store_true")
    args = parser.parse_args()
    label = args.label or args.name
    plt.switch_backend("Agg")

    from ultralytics import YOLO  # imported here so the module loads fast in tests

    model = YOLO(args.checkpoint)
    firearm_ids = firearm_class_ids(model.names)
    imgsz = args.imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))
    print(f"Firearm classes: {[(i, model.names[i]) for i in firearm_ids]}, imgsz {imgsz}")

    manifest = pd.read_csv(MANIFEST_PATH)
    y = manifest["firearm_present"].to_numpy()
    split = manifest["split"].to_numpy()
    paths = manifest["image_path"].to_numpy()

    model.predict(str(REPO_ROOT / paths[0]), imgsz=imgsz, device="cpu", verbose=False)  # warm-up

    # ---- VALIDATION: choose and save the threshold -------------------------------------
    val = split == "val"
    val_scores, val_ms = score_images(model, list(paths[val]), imgsz, firearm_ids)
    settings = {"checkpoint": str(args.checkpoint), "imgsz": imgsz, "min_confidence": 0.05}
    threshold = freeze_threshold(args.name, y[val], val_scores, settings)
    print(f"Threshold {threshold} saved to chosen_settings.json - now evaluating test once.")

    # ---- TEST: evaluated once with the frozen threshold --------------------------------
    test = split == "test"
    test_scores, test_ms = score_images(model, list(paths[test]), imgsz, firearm_ids)

    rows, tables = [], []
    for part, mask, scores, ms in [
        ("val", val, val_scores, val_ms),
        ("test", test, test_scores, test_ms),
    ]:
        y_pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "model": label,
                "feature_set": f"end-to-end YOLO (imgsz {imgsz})",
                "main_model": False,
                "split": part,
                "n_images": int(mask.sum()),
                "threshold": threshold,
                **compute_metrics(y[mask], y_pred, scores),
                "train_seconds": float("nan"),
                "predict_ms_per_image": ms,
            }
        )
        tables.append(
            pd.DataFrame(
                {
                    "image": paths[mask],
                    "split": part,
                    "y_true": y[mask],
                    "score": scores,
                    "y_pred": y_pred,
                }
            )
        )
        if part == "test":
            save_confusion_matrix(
                y[mask],
                y_pred,
                f"confusion_{args.name}_test.png",
                f"{label} - test (threshold {threshold})",
            )
    pd.concat(tables).to_csv(RESULTS_DIR / f"{args.name}_predictions.csv", index=False)
    save_metric_rows(rows)
    print(pd.DataFrame(rows).drop(columns=["model", "feature_set"]).round(4).to_string(index=False))

    handcrafted, _, names = load_features(manifest)
    brightness = handcrafted[:, names.index("mean_brightness")]
    error_analysis(manifest, brightness, args.name, out_name=f"error_analysis_{args.name}")

    # ---- Detection metrics (fine-tuned checkpoint only), kept in a SEPARATE file ----------
    is_fine_tuned = [model.names[i].lower() for i in firearm_ids] == ["firearm"]
    if is_fine_tuned and not args.skip_detection_val:
        metrics = model.val(
            data=str(YOLO_DATA_YAML),
            split="test",
            imgsz=imgsz,
            device="cpu",
            plots=False,
            project=str(REPO_ROOT / "outputs" / "yolo_runs"),
            name=f"{args.name}_test_val",
        )
        detection = {
            "checkpoint": str(args.checkpoint),
            "split": "test",
            "imgsz": imgsz,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        out = RESULTS_DIR / f"{args.name}_detection_metrics.json"
        out.write_text(json.dumps(detection, indent=2))
        print(f"Detection metrics (test): {detection}")
    print(f"Wrote predictions, metrics, figures to {RESULTS_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()

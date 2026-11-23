"""Fine-tune a YOLO detector on outputs/yolo_firearm (standard transfer learning).

Run with (GPU, e.g. Colab):
    uv run python -m weapon_detection.ml.yolo_train
CPU fallback preset (smaller images, fewer epochs):
    uv run python -m weapon_detection.ml.yolo_train --cpu-preset
"""

import argparse

import torch

from weapon_detection.ml.common import REPO_ROOT, SEED

DEFAULT_DATA = REPO_ROOT / "outputs" / "yolo_firearm" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "outputs" / "yolo_runs"


def build_train_args(args: argparse.Namespace) -> dict:
    """Turn command-line options into the keyword arguments for model.train()."""
    epochs, imgsz, batch = args.epochs, args.imgsz, args.batch
    if args.cpu_preset:
        epochs, imgsz, batch = 8, 416, 8  # small enough to finish on a laptop CPU
    device = 0 if torch.cuda.is_available() else "cpu"  # 0 = first GPU
    return {
        "data": str(args.data),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": args.patience,
        "seed": SEED,
        "deterministic": True,
        "workers": args.workers,
        "plots": True,
        "device": device,
        "project": str(args.project),
        "name": args.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the firearm dataset")
    parser.add_argument("--model", default="yolov8n.pt", help="starting checkpoint (COCO)")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="firearm_yolov8n")
    parser.add_argument("--cpu-preset", action="store_true", help="imgsz=416, epochs=8, batch=8")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help is fast

    train_args = build_train_args(args)
    print(f"Starting checkpoint: {args.model}")
    print("Exact training arguments:")
    for key, value in train_args.items():
        print(f"  {key} = {value}")
    YOLO(args.model).train(**train_args)


if __name__ == "__main__":
    main()

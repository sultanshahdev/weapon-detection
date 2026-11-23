"""Fine-tune a trusted local YOLO checkpoint using a prepared dataset."""

import argparse
from pathlib import Path

import yaml

from weapon_detection.data.annotations import read_names
from weapon_detection.data.dataset import SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu locally or a GPU index in the cloud")
    parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        model = args.model.resolve()
        if not model.is_file() or model.suffix != ".pt":
            raise ValueError("Supply a trusted local .pt starting checkpoint.")
        if min(args.epochs, args.imgsz, args.batch) < 1:
            raise ValueError("Epochs, size and batch must be positive.")
        if args.device != "cpu" and not args.device.isdecimal():
            raise ValueError("Use cpu or a numeric cloud GPU index.")
        data = args.data.resolve()
        if read_names(data) != {0: "firearm"}:
            raise ValueError("Expected the prepared single firearm class.")
        document = yaml.safe_load(data.read_text(encoding="utf-8"))
        root = Path(document["path"])
        if root != data.parent or any(
            document.get(split) != f"images/{split}"
            or not (root / "images" / split).is_dir()
            or not (root / "labels" / split).is_dir()
            for split in SPLITS
        ):
            raise ValueError(
                "Expected prepared local splits. Update path after moving the dataset."
            )
        if document.get("download"):
            raise ValueError("Remove download directives from the local dataset configuration.")
        from ultralytics import YOLO

        YOLO(str(model), task="detect").train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.output.resolve()),
            name="firearm",
            seed=args.seed,
            deterministic=True,
            workers=0,
            plots=False,
            amp=False,
        )
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

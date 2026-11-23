"""Run CPU firearm detection on a local video and optionally export annotated video and CSV."""

import argparse
import csv
import json
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from weapon_detection.detection.export import FIELDS, detection_rows
from weapon_detection.detection.model import FIREARM_ALIASES, YoloDetector
from weapon_detection.detection.video_processor import VideoStats, process_video
from weapon_detection.utils.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--classes-only", action="store_true", help="Load and show firearm mappings"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--alias", action="append", help="Override default exact firearm aliases")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    if args.input is None and not args.classes_only:
        parser.error("--input is required for video processing.")
    try:
        settings = load_settings()
        model = args.model or settings.model_path
        if model is None:
            raise ValueError("Set --model or WD_MODEL_PATH to a trusted local checkpoint.")
        detector = YoloDetector(
            model,
            device=args.device,
            confidence=settings.confidence_threshold if args.conf is None else args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            aliases=args.alias or FIREARM_ALIASES,
        )
        print(json.dumps({"firearm_classes": detector.firearm_classes, "device": args.device}))
        if not args.classes_only:
            run(args, detector)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.error(str(exc))


def run(args: argparse.Namespace, detector: YoloDetector) -> None:
    stats = VideoStats()
    with ExitStack() as stack:
        exporter = None
        if args.csv:
            forbidden = {p.resolve() for p in (args.input, args.output, detector.checkpoint) if p}
            if args.csv.exists() or args.csv.resolve() in forbidden:
                raise ValueError("CSV must be a new path distinct from inputs and video output.")
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(stack.enter_context(TemporaryDirectory(dir=args.csv.parent))) / "run.csv"
            stream = stack.enter_context(stage.open("w", newline="", encoding="utf-8"))
            exporter = csv.DictWriter(stream, fieldnames=FIELDS)
            exporter.writeheader()
        frames = stack.enter_context(
            closing(
                process_video(
                    args.input,
                    detector,
                    stats=stats,
                    output=args.output,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                    output_size=tuple(args.output_size) if args.output_size else None,
                )
            )
        )
        for frame in frames:
            if exporter is not None:
                exporter.writerows(detection_rows(args.input.name, frame))
        if exporter is not None:
            stream.close()
            args.csv.hardlink_to(stage)
    print(json.dumps(stats.summary(), indent=2))


if __name__ == "__main__":
    main()

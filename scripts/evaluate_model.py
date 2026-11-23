"""Evaluate held-out images or benchmark a local video with a trusted CPU checkpoint."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import cv2
import yaml

from weapon_detection.detection.evaluation import evaluate_dataset, validate_native_map_config
from weapon_detection.detection.model import FIREARM_ALIASES, YoloDetector
from weapon_detection.detection.video_processor import VideoStats, process_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path, help="Prepared dataset root with a test split")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45, help="Prediction NMS IoU")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--alias", action="append", help="Override default exact firearm aliases")
    parser.add_argument("--ultralytics-map", action="store_true", help="Run native validation")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--overlap-note", default="Unknown relative to checkpoint training sources."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_report.json"))
    args = parser.parse_args()

    report = {"status": "unavailable"}
    try:
        if args.model is None or not args.model.is_file():
            raise ValueError("No local checkpoint supplied or checkpoint file is missing.")
        if args.dataset is None and args.video is None:
            raise ValueError("Supply a prepared test dataset, a local video, or both.")
        detector = YoloDetector(
            args.model,
            confidence=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            aliases=args.alias or FIREARM_ALIASES,
        )
        with detector.checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update(
            checkpoint=str(detector.checkpoint),
            checkpoint_sha256=checksum,
            device="cpu",
            classes=detector.firearm_classes,
            confidence=args.conf,
            nms_iou=args.iou,
            imgsz=args.imgsz,
            overlap_note=args.overlap_note,
            python=platform.python_version(),
            torch=version("torch"),
            ultralytics=version("ultralytics"),
            opencv=cv2.__version__,
        )
        if args.dataset is not None:
            report["dataset"] = {
                "path": str(args.dataset.resolve()),
                **evaluate_dataset(args.dataset, detector, threshold=args.match_iou),
            }
            if args.ultralytics_map:
                if detector.class_names.keys() != {0} or detector.firearm_classes.keys() != {0}:
                    report["dataset"]["map_status"] = (
                        "Unavailable. Native validation requires checkpoint class 0 to be the "
                        "sole firearm class. The operating-point results use alias mapping."
                    )
                else:
                    data = validate_native_map_config(args.dataset)
                    report["dataset"].update(detector.native_map(data, args.output.parent))
                    report["dataset"]["map_status"] = "Ultralytics native validation on test split."
                    report["dataset"]["map_confidence_floor"] = 0.001
        if args.video is not None:
            stats = VideoStats()
            for _ in process_video(
                args.video,
                detector,
                stats=stats,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            ):
                pass
            report["video"] = {
                "name": args.video.name,
                "frame_stride": args.frame_stride,
                "max_frames": args.max_frames,
                "timing": "Adapter wall time includes preprocessing and first-call warmup "
                "if this is the first input.",
                **stats.summary(),
            }
        report["status"] = "completed"
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        report["status"] = "partial" if "dataset" in report else "unavailable"
        report["reason"] = str(exc)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    except OSError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

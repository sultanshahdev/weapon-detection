"""Inspect a local VOC or YOLO dataset."""

import argparse
import json
from pathlib import Path

import yaml

from weapon_detection.data.dataset import inspect_dataset
from weapon_detection.detection.model import FIREARM_ALIASES, normalize_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--names", type=Path, help="YOLO data YAML (defaults to dataset/data.yaml)")
    parser.add_argument("--firearm-label", action="append", help="Override default exact aliases")
    parser.add_argument("--voc-one-based", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON report location")
    args = parser.parse_args()
    try:
        _, report = inspect_dataset(
            args.dataset,
            {normalize_label(n) for n in args.firearm_label or FIREARM_ALIASES},
            names_file=args.names,
            voc_one_based=args.voc_one_based,
        )
        rendered = json.dumps(report, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        print(rendered)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if not report["valid_pairs"] or report["image_errors"] or report["orphan_annotations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

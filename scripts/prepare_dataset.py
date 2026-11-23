"""Export verified VOC or YOLO records as a single-class Ultralytics dataset."""

import argparse
import json
from pathlib import Path

import yaml

from weapon_detection.data.dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True, help="Local preparation JSON")
    parser.add_argument("--names", type=Path, help="Source YOLO class names YAML")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voc-one-based", action="store_true")
    parser.add_argument("--preserve-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be an object.")
        source = manifest.get("source")
        mapping = manifest.get("mapping")
        groups = manifest.get("groups", {})
        negatives = manifest.get("verified_negatives", [])
        if (
            not isinstance(source, dict)
            or not isinstance(mapping, dict)
            or not isinstance(groups, dict)
            or not isinstance(negatives, list)
            or not all(isinstance(item, str) for item in negatives)
            or not all(isinstance(item, str) and item.strip() for item in groups.values())
            or not all(isinstance(item, str) and item.strip() for item in mapping.values())
        ):
            raise ValueError("Expected source, mapping, groups and verified_negatives fields.")
        report = prepare_dataset(
            args.dataset,
            args.output,
            mapping=mapping,
            negatives=set(negatives),
            source=source,
            groups=groups,
            ratios=tuple(args.ratios),
            seed=args.seed,
            names_file=args.names,
            voc_one_based=args.voc_one_based,
            preserve_test=args.preserve_test,
            overwrite=args.overwrite,
        )
        print(json.dumps(report["split_counts"]))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

"""Export a one-class YOLO dataset ("firearm") that uses the SAME split as the classical models.

Run with:  uv run python -m weapon_detection.ml.yolo_data [--train-cap 3200]

Layout:  outputs/yolo_firearm/images/{train,val,test}  and  labels/{train,val,test}
Labels are rewritten from the ORIGINAL extracted label files: Gun and Pistol -> class 0,
every other class (knife, card, notes, phone) is dropped. Images left with no firearm box
get an EMPTY label file, which YOLO treats as a background image.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from weapon_detection.ml.common import (
    FIREARM_CLASS_NAMES,
    MANIFEST_PATH,
    REPO_ROOT,
    SEED,
    find_data_yaml,
    load_class_names,
    parse_label_line,
)

EXPORT_DIR = REPO_ROOT / "outputs" / "yolo_firearm"
ZIP_PATH = REPO_ROOT / "outputs" / "yolo_colab" / "yolo_firearm.zip"
DEFAULT_TRAIN_CAP = 3200
SPLITS = ["train", "val", "test"]


def label_path_for(image_path: Path) -> Path:
    """Roboflow layout: <split>/images/x.jpg  ->  <split>/labels/x.txt"""
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def rewrite_label_text(text: str, class_names: list[str]) -> str:
    """Keep only firearm boxes, as class 0 in 'cx cy w h' form (polygons become boxes)."""
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        class_id, (x_min, y_min, x_max, y_max) = parse_label_line(line)
        if class_names[class_id].lower() not in FIREARM_CLASS_NAMES:
            continue  # knife, card, currency notes, mobile -> dropped
        # Clip to the image, then convert corners to centre / width / height.
        x_min, y_min = max(0.0, x_min), max(0.0, y_min)
        x_max, y_max = min(1.0, x_max), min(1.0, y_max)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def check_label_text(text: str) -> None:
    """Every line must be '0 cx cy w h' with normalised, positive-size boxes."""
    for line in text.splitlines():
        values = line.split()
        assert len(values) == 5 and values[0] == "0", f"Bad line: {line}"
        cx, cy, w, h = (float(v) for v in values[1:])
        assert 0 < w <= 1 and 0 < h <= 1, f"Bad size: {line}"
        assert 0 <= cx - w / 2 + 1e-6 and cx + w / 2 <= 1 + 1e-6, f"Out of image: {line}"
        assert 0 <= cy - h / 2 + 1e-6 and cy + h / 2 <= 1 + 1e-6, f"Out of image: {line}"


def choose_images(manifest: pd.DataFrame, train_cap: int) -> pd.DataFrame:
    """Train: all negatives + a seeded sample of positives (at most train_cap images).

    Val and test: the full splits, so YOLO is scored on exactly the same images.
    """
    train = manifest[manifest["split"] == "train"]
    negatives = train[train["firearm_present"] == 0]
    positives = train[train["firearm_present"] == 1]
    n_positives = max(0, min(len(positives), train_cap - len(negatives)))
    sampled = positives.sample(n=n_positives, random_state=SEED)
    others = manifest[manifest["split"].isin(["val", "test"])]
    return pd.concat([negatives, sampled, others]).sort_values(["split", "image_path"])


def set_data_yaml_path(yaml_path: Path, dataset_folder: str) -> None:
    """Point data.yaml at the folder where the dataset now lives (e.g. /content/yolo_firearm)."""
    data = yaml.safe_load(Path(yaml_path).read_text())
    data["path"] = str(dataset_folder)
    Path(yaml_path).write_text(yaml.safe_dump(data, sort_keys=False))


def export(train_cap: int = DEFAULT_TRAIN_CAP) -> dict:
    manifest = pd.read_csv(MANIFEST_PATH)
    chosen = choose_images(manifest, train_cap)
    names = chosen["image_path"].map(lambda p: Path(p).name)
    assert names.is_unique, "Two images share a file name"
    assert chosen["image_path"].is_unique, "An image appears in two splits"

    dataset_root = REPO_ROOT / Path(manifest["image_path"].iloc[0]).parents[2]
    class_names = load_class_names(find_data_yaml(dataset_root))

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)  # generated folder: rebuild from scratch every time
    for split in SPLITS:
        (EXPORT_DIR / "images" / split).mkdir(parents=True)
        (EXPORT_DIR / "labels" / split).mkdir(parents=True)

    summary = {
        s: {"images": 0, "with_firearm": 0, "background": 0, "firearm_boxes": 0} for s in SPLITS
    }
    for _, row in chosen.iterrows():
        source = REPO_ROOT / row["image_path"]
        split = row["split"]
        label_file = label_path_for(source)
        original = label_file.read_text() if label_file.exists() else ""
        new_text = rewrite_label_text(original, class_names)
        check_label_text(new_text)
        n_boxes = len(new_text.splitlines())
        # The YOLO label must agree with the manifest's image-level label.
        assert (n_boxes > 0) == bool(row["firearm_present"]), f"Label mismatch: {source}"

        shutil.copy2(source, EXPORT_DIR / "images" / split / source.name)
        (EXPORT_DIR / "labels" / split / f"{source.stem}.txt").write_text(new_text)
        summary[split]["images"] += 1
        summary[split]["with_firearm"] += int(n_boxes > 0)
        summary[split]["background"] += int(n_boxes == 0)
        summary[split]["firearm_boxes"] += n_boxes

    data = {
        "path": str(EXPORT_DIR),  # local absolute path; use set_data_yaml_path() elsewhere
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "firearm"},
    }
    (EXPORT_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    summary["train_cap"] = train_cap
    summary["seed"] = SEED
    summary["split_column"] = "split (v2)"
    (EXPORT_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2))

    # Zip with the folder name at the top, so unzipping gives yolo_firearm/...
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(EXPORT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, Path("yolo_firearm") / path.relative_to(EXPORT_DIR))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-cap", type=int, default=DEFAULT_TRAIN_CAP)
    args = parser.parse_args()
    summary = export(args.train_cap)
    for split in SPLITS:
        print(split, summary[split])
    size_mb = ZIP_PATH.stat().st_size / 1e6
    print(f"Wrote {EXPORT_DIR} and {ZIP_PATH} ({size_mb:.1f} MB)")
    # Double-check: no file name appears in two splits.
    seen = [p.name for s in SPLITS for p in (EXPORT_DIR / "images" / s).iterdir()]
    assert len(seen) == len(set(seen))


if __name__ == "__main__":
    main()

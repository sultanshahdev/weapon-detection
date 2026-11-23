"""Shared paths, constants and small helper functions for the ML experiment."""

import zipfile
from pathlib import Path

import yaml

# One fixed seed used everywhere so every run gives the same result.
SEED = 42

# Folder layout. Everything generated goes under outputs/ml/ (git-ignored).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = REPO_ROOT / "datasets" / "raw" / "Weapon Detection.zip"
EXTRACT_DIR = REPO_ROOT / "datasets" / "raw" / "weapon_detection_extracted"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ml"
FIGURE_DIR = OUTPUT_DIR / "figures"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
FEATURES_PATH = OUTPUT_DIR / "features.npz"

# Split folder names used by Roboflow exports.
ORIGINAL_SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Which class names count as a firearm. Chosen after printing data.yaml, whose names are
# ['Card', 'Currency Notes', 'Gun', 'Knife', 'Mobile', 'Pistol']. Edit this set to change
# the task. Names are compared in lower case.
FIREARM_CLASS_NAMES = {"gun", "pistol"}


def safe_extract(zip_path: Path, target_dir: Path) -> None:
    """Extract a ZIP, refusing any member whose path would escape target_dir."""
    target_dir = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            destination = (target_dir / member).resolve()
            # "../evil.txt" or "/etc/passwd" would resolve outside the target folder.
            if not destination.is_relative_to(target_dir):
                raise ValueError(f"Unsafe path in ZIP: {member}")
        archive.extractall(target_dir)


def find_data_yaml(dataset_root: Path) -> Path:
    """Return the first data.yaml found under dataset_root."""
    matches = sorted(dataset_root.rglob("data.yaml"))
    if not matches:
        raise FileNotFoundError(f"No data.yaml under {dataset_root}")
    return matches[0]


def load_class_names(yaml_path: Path) -> list[str]:
    """Read the class names from data.yaml (list form or {id: name} form)."""
    data = yaml.safe_load(yaml_path.read_text())
    names = data["names"]
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    return [str(name) for name in names]


def parse_label_line(line: str) -> tuple[int, list[float]]:
    """Parse one YOLO label line into (class_id, [x_min, y_min, x_max, y_max]).

    Box line:     class cx cy w h            (5 values)
    Polygon line: class x1 y1 x2 y2 x3 y3 ... (odd count, at least 7 values)
    All coordinates are normalised to 0..1.
    """
    values = line.split()
    class_id = int(values[0])
    numbers = [float(value) for value in values[1:]]
    if len(numbers) == 4:
        cx, cy, w, h = numbers
        return class_id, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    if len(numbers) >= 6 and len(numbers) % 2 == 0:
        xs = numbers[0::2]
        ys = numbers[1::2]
        return class_id, [min(xs), min(ys), max(xs), max(ys)]
    raise ValueError(f"Malformed label line: {line!r}")


def box_area(box: list[float]) -> float:
    """Area of a normalised box = fraction of the image it covers."""
    x_min, y_min, x_max, y_max = box
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def source_name(file_name: str) -> str:
    """Roboflow names copies 'photo_jpg.rf.<hash>.jpg'; the part before '.rf.' is the source."""
    if ".rf." in file_name:
        return file_name.split(".rf.")[0]
    return Path(file_name).stem


def relative_path(path: Path) -> str:
    """Path relative to the repo when possible (keeps the manifest portable)."""
    path = path.resolve()
    if path.is_relative_to(REPO_ROOT):
        return path.relative_to(REPO_ROOT).as_posix()
    return path.as_posix()

"""Read VOC XML and YOLO detection annotations in normalized coordinates."""

import math
from pathlib import Path
from xml.etree import ElementTree

import yaml

from weapon_detection.detection.model import normalize_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
Box = tuple[float, float, float, float]
Labels = list[tuple[str, Box]]


def read_names(path: Path) -> dict[int, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = document.get("names") if isinstance(document, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (
        not isinstance(names, dict)
        or not names
        or any(
            type(key) is not int or key < 0 or not isinstance(value, str) or not value.strip()
            for key, value in names.items()
        )
    ):
        raise ValueError("YOLO data YAML must provide source class IDs and names.")
    return {key: normalize_label(value) for key, value in names.items()}


def xyxy_to_yolo(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box has zero area, invalid coordinates, or lies outside the image.")
    return (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def yolo_to_xyxy(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    return ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)


def read_labels(
    path: Path, width: int, height: int, names: dict[int, str], *, voc_one_based: bool = False
) -> Labels:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if path.suffix.lower() == ".xml":
        annotation = ElementTree.parse(path).getroot()
        size = (
            int(annotation.findtext("size/width", "")),
            int(annotation.findtext("size/height", "")),
        )
        if annotation.tag != "annotation" or size != (width, height):
            raise ValueError("VOC image dimensions disagree with decoded dimensions.")
        labels = []
        for item in annotation.findall("object"):
            name = normalize_label(item.findtext("name", ""))
            if not name:
                raise ValueError("VOC object is missing its class name.")
            x1, y1, x2, y2 = (
                float(item.findtext(f"bndbox/{key}", ""))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
            offset = int(voc_one_based)
            labels.append((name, xyxy_to_yolo((x1 - offset, y1 - offset, x2, y2), width, height)))
        return labels

    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            key, x, y, w, h = line.split()
            box = (float(x), float(y), float(w), float(h))
            left, top, right, bottom = yolo_to_xyxy(box, 1, 1)
            if not (
                all(math.isfinite(value) and 0 <= value <= 1 for value in box)
                and right > left
                and bottom > top
                and min(left, top) >= -1e-9
                and max(right, bottom) <= 1 + 1e-9
            ):
                raise ValueError("Nonfinite, zero-area or out-of-bounds box.")
            labels.append((names[int(key)], box))
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid YOLO annotation on line {line_number}: {exc}") from exc
    return labels

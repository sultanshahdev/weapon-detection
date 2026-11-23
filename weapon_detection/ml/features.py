"""Step 2: compute pixel-only features for every image in the manifest.

Run with:  uv run python -m weapon_detection.ml.features   (add --force to recompute)
Inputs are pixels only - never boxes, class names, file names or the split.
No scaler is fitted here; scaling is fitted on the training split inside the model later.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.feature import hog

from weapon_detection.ml.common import FEATURES_PATH, MANIFEST_PATH, REPO_ROOT

IMAGE_SIZE = 128  # every image is resized to 128x128 before computing features
DARK_THRESHOLD = 50  # grey level below which a pixel counts as "dark"

HANDCRAFTED_NAMES = [
    "mean_brightness",
    "contrast_std",
    "edge_density",
    "mean_blue",
    "mean_green",
    "mean_red",
    "mean_saturation",
    "laplacian_sharpness",
    "dark_fraction",
    "histogram_entropy",
]


def handcrafted_features(image_bgr: np.ndarray) -> list[float]:
    """Ten interpretable numbers describing a 128x128 BGR image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 100, 200)  # 255 where an edge is found, 0 elsewhere
    histogram = np.bincount(gray.flatten(), minlength=256) / gray.size
    nonzero = histogram[histogram > 0]

    return [
        float(gray.mean()),  # mean_brightness
        float(gray.std()),  # contrast_std
        float((edges > 0).mean()),  # edge_density
        float(image_bgr[:, :, 0].mean()),  # mean_blue
        float(image_bgr[:, :, 1].mean()),  # mean_green
        float(image_bgr[:, :, 2].mean()),  # mean_red
        float(hsv[:, :, 1].mean()),  # mean_saturation
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # laplacian_sharpness
        float((gray < DARK_THRESHOLD).mean()),  # dark_fraction
        float(-(nonzero * np.log2(nonzero)).sum()),  # histogram_entropy (bits)
    ]


def hog_features(image_bgr: np.ndarray) -> np.ndarray:
    """Histogram of Oriented Gradients on the grey 128x128 image (1764 numbers)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


def features_from_array(image_bgr: np.ndarray) -> tuple[list[float], np.ndarray]:
    """Resize a BGR image to 128x128 and return (handcrafted, hog).

    Used for training AND by the prediction GUI, so both see identical features.
    """
    image = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return handcrafted_features(image), hog_features(image)


def features_for_image(path: Path) -> tuple[list[float], np.ndarray]:
    """Load one image from disk and return (handcrafted, hog)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image {path}")
    return features_from_array(image)


def compute_features(paths: list[Path], n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Compute features for many images in parallel. Returns (handcrafted, hog) arrays."""
    results = Parallel(n_jobs=n_jobs)(delayed(features_for_image)(p) for p in paths)
    handcrafted = np.array([r[0] for r in results], dtype=np.float32)
    hog_matrix = np.array([r[1] for r in results], dtype=np.float32)
    return handcrafted, hog_matrix


def load_features(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (handcrafted, hog, handcrafted_names) in the SAME row order as the manifest.

    Rows are matched by image path, never by position, so re-splitting the manifest
    cannot silently attach features to the wrong image.
    """
    cache = np.load(FEATURES_PATH, allow_pickle=False)
    cached_paths = cache["image_path"].tolist()
    if set(cached_paths) != set(manifest["image_path"]):
        raise ValueError("features.npz and manifest.csv list different images; rerun --force")
    row_of_path = {path: row for row, path in enumerate(cached_paths)}
    rows = [row_of_path[path] for path in manifest["image_path"]]
    handcrafted = cache["handcrafted"][rows]
    hog_matrix = cache["hog"][rows]
    # Final check: the reordered paths must equal the manifest paths one by one.
    assert [cached_paths[r] for r in rows] == manifest["image_path"].tolist()
    return handcrafted, hog_matrix, cache["handcrafted_names"].tolist()


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    image_paths = manifest["image_path"].tolist()

    # Reuse the cache if it was built for exactly the same images (order does not matter,
    # because load_features() joins by image path).
    if FEATURES_PATH.exists() and "--force" not in sys.argv:
        cached = np.load(FEATURES_PATH, allow_pickle=False)
        if set(cached["image_path"].tolist()) == set(image_paths):
            print(f"Cache is up to date: {FEATURES_PATH} (use --force to recompute)")
            return

    print(f"Computing features for {len(image_paths)} images ...")
    handcrafted, hog_matrix = compute_features([REPO_ROOT / p for p in image_paths])
    np.savez_compressed(
        FEATURES_PATH,
        image_path=np.array(image_paths),
        handcrafted=handcrafted,
        handcrafted_names=np.array(HANDCRAFTED_NAMES),
        hog=hog_matrix,
    )
    print(f"handcrafted: {handcrafted.shape}  hog: {hog_matrix.shape}")
    print(f"Wrote {FEATURES_PATH}")


if __name__ == "__main__":
    main()

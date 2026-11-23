"""Leakage-audit helpers: file hashes, perceptual hashes and union-find grouping."""

import hashlib
from pathlib import Path

import cv2
import numpy as np


def image_info(path: Path) -> dict:
    """Read one image and return its SHA-256, 64-bit dHash, width and height.

    'ok' is False when OpenCV cannot decode the file (a corrupt image).
    """
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"sha256": sha256, "dhash": 0, "width": 0, "height": 0, "ok": False}
    height, width = image.shape
    return {"sha256": sha256, "dhash": dhash(image), "width": width, "height": height, "ok": True}


def dhash(gray: np.ndarray) -> int:
    """Difference hash: shrink to 9x8, compare each pixel with its right neighbour."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]  # 8 rows x 8 comparisons = 64 bits
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def near_duplicate_pairs(hashes: list[int], max_distance: int) -> list[tuple[int, int]]:
    """All index pairs (i, j), i < j, whose dHashes differ in at most max_distance bits."""
    array = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(array) - 1):
        # XOR marks the differing bits; bitwise_count counts them (Hamming distance).
        distances = np.bitwise_count(array[i + 1 :] ^ array[i])
        for offset in np.nonzero(distances <= max_distance)[0]:
            pairs.append((i, i + 1 + int(offset)))
    return pairs


def pairs_within_groups(keys: list[str]) -> list[tuple[int, int]]:
    """All index pairs that share the same key (e.g. same SHA-256 or same source name)."""
    members: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        members.setdefault(key, []).append(index)
    pairs = []
    for indices in members.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append((indices[a], indices[b]))
    return pairs


def find(parent: list[int], i: int) -> int:
    """Union-find: follow parents up to the root of i's group."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # path halving keeps the trees short
        i = parent[i]
    return i


def union(parent: list[int], a: int, b: int) -> None:
    """Union-find: put a and b in the same group."""
    root_a = find(parent, a)
    root_b = find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def group_ids(n_items: int, pairs: list[tuple[int, int]]) -> list[str]:
    """Merge all linked pairs transitively and return one group id per item."""
    parent = list(range(n_items))
    for a, b in pairs:
        union(parent, a, b)
    return [f"g{find(parent, i):05d}" for i in range(n_items)]

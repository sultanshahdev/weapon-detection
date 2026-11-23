"""Run the detector ONCE per Mendeley demo video and cache the raw boxes (demo only).

Run with:  uv run python scripts/precompute_demo.py            (the 10 selected videos)
           uv run python scripts/precompute_demo.py --all      (every extracted video)
           uv run python scripts/precompute_demo.py --benchmark  (per-frame vs batched)

- Boxes are stored at DETECTION_FLOOR (0.05), imgsz 640, CPU, every frame, in
  outputs/demo_cache/<class>/<video>.json. Writes are atomic (temp file then rename).
- Resumable: a cache whose checkpoint name/size/mtime, imgsz and floor match is skipped.
- One failing video is logged and the run continues. Ctrl-C stops cleanly.
- After each video, the two app MP4s (GT only; GT + model at 0.25) are rendered from the
  cache into outputs/demo_videos/ without calling the model.
- Only one copy can run at a time (a lock file in outputs/demo_cache/).
"""

import argparse
import fcntl
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weapon_detection.data.mendeley_videos import (  # noqa: E402
    DATASET_ROOT,
    list_videos,
    load_boxes,
    video_info,
)
from weapon_detection.detection.annotated_video import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    cache_name,
    render_annotated_video,
)
from weapon_detection.detection.demo_cache import (  # noqa: E402
    CACHE_DIR,
    DETECTION_FLOOR,
    IMGSZ,
    cache_path,
    detect_video,
    order_videos,
    pending,
    read_cache,
    save_summary,
    signature,
    summary_rows,
    write_cache,
)
from weapon_detection.detection.model import YoloDetector, default_checkpoint  # noqa: E402

SEED = 42
SELECTION = CACHE_DIR / "demo_selection.json"
VIDEO_DIR = ROOT / "outputs" / "demo_videos"
# Frames per model call, set from the --benchmark result (2026-11-23, this CPU,
# Handgun/PCH1_C1_P4_V1_HB_2): per-frame 3.54 frames/s, batch 16 6.45 frames/s, same boxes
# within 1e-3. Batch 16 is used only because it was faster AND gave the same boxes.
BATCH_SIZE = 16

log = logging.getLogger("precompute")


def same_boxes(a: dict, b: dict, tolerance: float = 1e-3) -> bool:
    """True if both results have the same frames, box counts and values within tolerance."""
    if a.keys() != b.keys():
        return False
    for frame in a:
        if len(a[frame]) != len(b[frame]):
            return False
        if not np.allclose(np.array(a[frame]), np.array(b[frame]), atol=tolerance):
            return False
    return True


def render_from_cache(record, gt: dict, cache: dict, checkpoint: Path) -> None:
    """GT-only and GT + model MP4s at the default confidence (no model call)."""
    video_id = f"{record.class_name}_{record.name}"
    for boxes in (None, cache["boxes"]):
        name = cache_name(
            video_id, True, checkpoint.name if boxes is not None else None, DEFAULT_CONFIDENCE
        )
        out = VIDEO_DIR / name
        if out.exists() and out.with_suffix(".json").exists():
            continue
        start = time.perf_counter()
        render_annotated_video(
            record.video_path, gt, None, out, DEFAULT_CONFIDENCE, cached_boxes=boxes
        )
        log.info("  rendered %s in %.1f s", name, time.perf_counter() - start)


def benchmark(detector: YoloDetector, records: list) -> None:
    """B4: per-frame vs batch 16 on one video, then the estimated total time."""
    record = records[0]
    timings = {}
    results = {}
    for batch in (1, 16):
        start = time.perf_counter()
        results[batch] = detect_video(detector, record.video_path, batch)
        timings[batch] = results[batch]["n_frames"] / (time.perf_counter() - start)
    same = same_boxes(results[1]["boxes"], results[16]["boxes"])
    print(f"video {record.class_name}/{record.name}, {results[1]['n_frames']} frames")
    print(f"per-frame: {timings[1]:.2f} frames/s   batch 16: {timings[16]:.2f} frames/s")
    print(f"same boxes within 1e-3: {same}")
    keep = 16 if same and timings[16] > timings[1] else 1
    total = sum(video_info(r.video_path)["frames"] for r in records)
    print(f"use batch {keep}; {len(records)} videos, {total} frames, "
          f"estimated {total / timings[keep] / 60:.1f} min")


def setup_logging() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(CACHE_DIR / "precompute.log"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def chosen_records(use_all: bool) -> tuple[list, list[tuple[str, str]]]:
    """Videos to process and the (class, name) pairs that go first."""
    records = list_videos(DATASET_ROOT)
    if use_all or not SELECTION.exists():
        return records, []
    selection = json.loads(SELECTION.read_text())["videos"]
    keys = [(v["class_name"], v["name"]) for v in selection]
    first = []  # top-ranked video of each class, in selection order
    for class_name, name in keys:
        if class_name not in {c for c, _ in first}:
            first.append((class_name, name))
    return [r for r in records if (r.class_name, r.name) in set(keys)], first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="every extracted video")
    parser.add_argument("--benchmark", action="store_true", help="per-frame vs batch 16")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_OFFLINE", "true")
    random.seed(SEED)
    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)

    checkpoint = default_checkpoint(ROOT)  # fine-tuned model, then the OIV7 fallback
    if checkpoint is None:
        raise SystemExit("No checkpoint in models/.")
    records, first = chosen_records(args.all)
    records = order_videos(records, first)
    if not records:
        raise SystemExit(f"No videos found under {DATASET_ROOT}")
    detector = YoloDetector(checkpoint, confidence=DETECTION_FLOOR, imgsz=IMGSZ)
    if args.benchmark:
        benchmark(detector, records)
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(CACHE_DIR / "precompute.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another precompute run is already active.") from None
    setup_logging()
    expected = signature(checkpoint)
    frames_left = sum(video_info(r.video_path)["frames"] for r in records)
    done_frames, busy_seconds, failures = 0, 0.0, []
    log.info("start: %d videos (%d without a valid cache), %d frames, checkpoint %s, batch %d",
             len(records), len(pending(records, expected)), frames_left, checkpoint.name,
             BATCH_SIZE)
    try:
        for number, record in enumerate(records, 1):
            path = cache_path(record.class_name, record.name)
            gt = load_boxes(record.label_path)
            n = video_info(record.video_path)["frames"]
            frames_left -= n
            try:
                cache = read_cache(path, expected)
                if cache is None:
                    start = time.perf_counter()
                    cache = {**expected, **detect_video(detector, record.video_path, BATCH_SIZE)}
                    write_cache(path, cache)
                    cache = read_cache(path, expected)
                    busy_seconds += time.perf_counter() - start
                    done_frames += n
                    rate = done_frames / busy_seconds
                    log.info("[%d/%d] %s/%s: %d frames at %.2f frames/s, ETA %.1f min",
                             number, len(records), record.class_name, record.name, n, rate,
                             frames_left / rate / 60)
                else:
                    log.info("[%d/%d] %s/%s: valid cache, skipped",
                             number, len(records), record.class_name, record.name)
                render_from_cache(record, gt, cache, checkpoint)
            except Exception:
                failures.append(record.name)
                log.exception("[%d/%d] %s/%s FAILED, continuing",
                              number, len(records), record.class_name, record.name)
    except KeyboardInterrupt:
        log.info("interrupted; finished caches are kept, rerun to resume")
        raise SystemExit(130) from None
    entries = []
    for record in records:
        cache = read_cache(cache_path(record.class_name, record.name), expected)
        if cache is not None:
            entries.append((record.class_name, load_boxes(record.label_path), cache))
    save_summary(summary_rows(entries))
    log.info("done: %d cached, %d failed %s", len(entries), len(failures), failures)


if __name__ == "__main__":
    main()

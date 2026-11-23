"""Headless smoke test of the Mendeley video demo (video_demo_app.py).

Run with:  uv run python scripts/smoke_video_demo.py [video folder name]
1. List the extracted videos and render ONE Handgun/Machine_Gun video with the GT overlay
   only, then with GT + model overlay if a checkpoint exists. Print paths, frame counts,
   codec and timings. Existing cached renders are deleted first so the timing is real.
2. Start `streamlit run video_demo_app.py --server.headless true` and check /_stcore/health.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weapon_detection.data.mendeley_videos import DATASET_ROOT, list_videos, load_boxes  # noqa: E402
from weapon_detection.detection.annotated_video import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    cache_name,
    check_playable,
    render_annotated_video,
)
from weapon_detection.detection.model import YoloDetector  # noqa: E402

OUT_DIR = ROOT / "outputs" / "demo_videos"
CHECKPOINTS = [ROOT / "models/firearm_yolov8n_best.pt", ROOT / "models/yolov8n-oiv7.pt"]
PORT = 8598


def render(record, detector, checkpoint) -> None:
    name = cache_name(
        f"{record.class_name}_{record.name}",
        True,
        checkpoint.name if detector else None,
        DEFAULT_CONFIDENCE,
    )
    out = OUT_DIR / name
    out.unlink(missing_ok=True)
    start = time.perf_counter()
    sidecar = render_annotated_video(
        record.video_path, load_boxes(record.label_path), detector, out, DEFAULT_CONFIDENCE
    )
    seconds = time.perf_counter() - start
    playable = check_playable(out)
    boxes = sum(len(v) for v in sidecar["model_boxes"].values())
    print(f"  output: {out.relative_to(ROOT)}")
    print(f"  frames rendered {sidecar['frames']}, read back {playable['frames']}, "
          f"codec {playable['codec']}, {out.stat().st_size / 1e6:.2f} MB")
    print(f"  time {seconds:.1f} s ({sidecar['frames'] / seconds:.1f} frames/s)")
    if detector:
        print(f"  model boxes {boxes} on {len(sidecar['model_boxes'])} frames")


def health_check() -> int:
    """Start the app headless, wait for the health endpoint, return its HTTP status."""
    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "video_demo_app.py"),
        "--server.headless", "true", "--server.port", str(PORT),
    ]  # fmt: skip
    env = {**os.environ, "YOLO_OFFLINE": "true"}
    server = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health") as reply:
                    return reply.status
            except OSError:
                time.sleep(1)
        return 0
    finally:
        server.terminate()
        server.wait(timeout=30)


def main() -> None:
    records = list_videos(DATASET_ROOT)
    print(f"dataset root: {DATASET_ROOT}  videos found: {len(records)}")
    labelled = [r for r in records if r.label_path and r.class_name != "No_Gun"]
    if not labelled:
        raise SystemExit("No labelled Handgun/Machine_Gun video found. Extract the dataset first.")
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    record = next((r for r in labelled if r.name == wanted), labelled[0])
    print(f"video: {record.class_name}/{record.name}")

    print("GT overlay only:")
    render(record, None, None)
    checkpoint = next((p for p in CHECKPOINTS if p.is_file()), None)
    if checkpoint is None:
        print("No checkpoint found; model overlay skipped.")
    else:
        os.environ.setdefault("YOLO_OFFLINE", "true")
        print(f"GT + model overlay ({checkpoint.name}, conf {DEFAULT_CONFIDENCE}):")
        render(record, YoloDetector(checkpoint, confidence=DEFAULT_CONFIDENCE), checkpoint)
    print(f"streamlit health status: {health_check()}")


if __name__ == "__main__":
    main()

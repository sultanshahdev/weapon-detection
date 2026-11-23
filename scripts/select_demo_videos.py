"""Choose the 10 Mendeley demo videos from the dataset's OWN annotations (no model involved).

Run with:  uv run python scripts/select_demo_videos.py
Writes outputs/demo_cache/demo_selection.json and outputs/demo_cache/demo_selection.md.

Rule (fixed before looking at any model output):
- Handgun and Machine_Gun: score = (fraction of frames that have a GT box)
  x (median GT box area as % of the frame). A high score means the firearm is visible in
  many frames and is not tiny, which is what makes a demo easy to follow. Top 4 per class.
- No_Gun has no boxes, so 2 videos are drawn at random with seed 42 from all 118 names.
Model predictions are never used here, so the choice cannot favour the model.
"""

import json
import random
import statistics
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weapon_detection.data.mendeley_videos import DATASET_ROOT  # noqa: E402

SEED = 42
PER_LABELLED_CLASS = 4
NO_GUN_COUNT = 2
OUTER_ZIP = ROOT / "datasets" / "raw" / "dataset.zip"
OUT = ROOT / "outputs" / "demo_cache" / "demo_selection.json"


def label_stats(label_path: Path) -> dict:
    """Frames, fraction of frames with a GT box and median box area (% of the frame)."""
    data = json.loads(label_path.read_text())
    frames = len(data["images"])
    size = {im["id"]: im["width"] * im["height"] for im in data["images"]}
    areas = [100 * a["bbox"][2] * a["bbox"][3] / size[a["image_id"]] for a in data["annotations"]]
    with_box = len({a["image_id"] for a in data["annotations"]})
    fraction = with_box / frames if frames else 0.0
    median = statistics.median(areas) if areas else 0.0
    return {
        "frames": frames,
        "gt_frame_fraction": round(fraction, 3),
        "median_gt_area_pct": round(median, 3),
        "score": round(fraction * median, 4),
    }


def no_gun_names() -> list[str]:
    """All No_Gun folder names, read from the inner ZIP's index (no extraction)."""
    with zipfile.ZipFile(OUTER_ZIP) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith(".zip"))
        with outer.open(inner_name) as stream, zipfile.ZipFile(stream) as inner:
            names = {n.split("/")[2] for n in inner.namelist() if n.endswith("video.mp4")
                     and n.split("/")[1] == "No_Gun"}  # fmt: skip
    return sorted(names)


def main() -> None:
    chosen = []
    for class_name in ("Handgun", "Machine_Gun"):
        rows = [
            {"class_name": class_name, "name": p.parent.name, **label_stats(p)}
            for p in sorted((DATASET_ROOT / class_name).glob("*/label.json"))
        ]
        rows.sort(key=lambda r: (-r["score"], r["name"]))  # ties broken by name
        chosen += rows[:PER_LABELLED_CLASS]
    names = no_gun_names()
    for name in sorted(random.Random(SEED).sample(names, NO_GUN_COUNT)):
        chosen.append({"class_name": "No_Gun", "name": name, "score": None})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rule = __doc__.split("Rule")[1].strip()
    OUT.write_text(json.dumps({"rule": rule, "videos": chosen}, indent=2))
    lines = [
        "| class | video | frames | fraction of frames with GT box | median GT area % | score |",
        "|---|---|---|---|---|---|",
    ]
    for r in chosen:
        lines.append(
            f"| {r['class_name']} | {r['name']} | {r.get('frames', '')} | "
            f"{r.get('gt_frame_fraction', '')} | {r.get('median_gt_area_pct', '')} | "
            f"{r['score'] if r['score'] is not None else 'seed 42 draw'} |"
        )
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"No_Gun candidates: {len(names)}. Written {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

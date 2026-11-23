"""Extract the Mendeley firearm-action video dataset (demo source only) once.

Run with:  uv run python scripts/extract_mendeley_videos.py [--max-per-class N]

The Mendeley download (datasets/raw/dataset.zip) is a ZIP that contains ONE inner ZIP,
Gun_Action_Recognition_Dataset.zip. We stream the inner ZIP straight out of the outer one,
so the 2.2 GB inner file is never written to disk. Every member name is checked against
path traversal before anything is written. Files that already exist are skipped.

--max-per-class N extracts all label.json files but only the first N videos per class
(sorted by folder name), for machines with little free disk space. The default (0) means all.
This dataset is used for the video demo only, never for training or reported metrics.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTER_ZIP = ROOT / "datasets" / "raw" / "dataset.zip"
DEST = ROOT / "datasets" / "interim" / "mendeley_firearm_actions"
TOP = "Gun_Action_Recognition_Dataset/"


def safe_target(dest: Path, name: str) -> Path:
    """Return dest/name, refusing absolute paths and '..' tricks."""
    target = (dest / name).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise ValueError(f"Unsafe member name in ZIP: {name}")
    return target


def chosen_members(inner: zipfile.ZipFile, max_per_class: int) -> list[zipfile.ZipInfo]:
    """All label files, plus every video (or the first N per class by folder name)."""
    members = [i for i in inner.infolist() if not i.is_dir()]
    if not any(i.filename.startswith(TOP) for i in members):
        tops = sorted({i.filename.split("/")[0] for i in inner.infolist()})
        raise SystemExit(f"No {TOP} folder in the inner ZIP. Top-level entries: {tops}")
    keep_videos: set[str] = set()
    by_class: dict[str, list[str]] = {}
    for info in members:
        parts = info.filename.split("/")
        if info.filename.endswith(".mp4") and len(parts) == 4:
            by_class.setdefault(parts[1], []).append(info.filename)
    for names in by_class.values():
        names.sort()
        keep_videos.update(names[:max_per_class] if max_per_class > 0 else names)
    chosen = [i for i in members if i.filename.endswith(".json") or i.filename in keep_videos]
    # Sorting by position in the archive means we only ever read the stream forwards,
    # which matters because the outer ZIP is deflated and cannot seek backwards cheaply.
    return sorted(chosen, key=lambda i: i.header_offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-class", type=int, default=0)
    args = parser.parse_args()
    if not OUTER_ZIP.is_file():
        raise SystemExit(f"Dataset ZIP not found: {OUTER_ZIP}")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTER_ZIP) as outer:
        inner_names = [n for n in outer.namelist() if n.endswith(".zip")]
        if len(inner_names) != 1:
            raise SystemExit(f"Expected one inner ZIP, found: {outer.namelist()}")
        with outer.open(inner_names[0]) as stream, zipfile.ZipFile(stream) as inner:
            written = skipped = 0
            for info in chosen_members(inner, args.max_per_class):
                target = safe_target(DEST, info.filename)
                if target.exists() and target.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with inner.open(info) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                written += 1
    print(f"written {written}, already present {skipped}, into {DEST}")


if __name__ == "__main__":
    main()

# Datasets

## Dataset used for all reported results: Roboflow "Weapon Detection-1" v4

The ML experiment and the fine-tuned detector use one dataset. It is a Roboflow export in
YOLOv8 format, saved as `datasets/raw/Weapon Detection.zip` (git-ignored) and extracted by
`weapon_detection/ml/prepare_data.py` into `datasets/raw/weapon_detection_extracted/`.
Attribution, copied exactly from `README.dataset.txt` in the export:

```text
# Weapon Detection-1 > OAWD_6_Classes-sohas-
https://universe.roboflow.com/weapon-detection-oss9n/weapon-detection-1-f9p7t

Provided by a Roboflow user
License: CC BY 4.0
```

- Version: "Weapon Detection-1 - v4 OAWD_6_Classes-sohas-" (`README.roboflow.txt`);
  download page
  <https://universe.roboflow.com/weapon-detection-oss9n/weapon-detection-1-f9p7t/dataset/4>.
- Licence as stated in the dataset's own files (`README.dataset.txt`, `data.yaml`): CC BY 4.0.
- TODO: verify the project title on the Roboflow page (it may be shown as "Pose Integrated
  Firearm Detection"; the files above do not use that name).
- Classes in `data.yaml`: Card, Currency Notes, Gun, Knife, Mobile, Pistol. `Gun` and
  `Pistol` define `firearm_present = 1`; images with only other classes, or no boxes, are
  negatives (`FIREARM_CLASS_NAMES` in `weapon_detection/ml/common.py`).
- Image counts, per-split counts, object counts and the leakage audit are computed by code
  and saved in `outputs/ml/audit.json` and `outputs/ml/report_numbers.json` (copies in
  `report_assets/evidence/`).
- The original Roboflow train/valid/test folders are not used as the split. Images are
  grouped (source names, SHA-256 duplicates, dHash near duplicates, filename families) and
  re-split with **split v2** (`weapon_detection/ml/split_v2.py`). Split v2 was made after
  the preliminary split v1 test results were seen; split v1 is kept in
  `outputs/ml/split_v1/` and disclosed in `outputs/ml/results/split_v1_vs_v2_test.csv`.
- The one-class YOLO export for fine-tuning (`weapon_detection/ml/yolo_data.py`) uses the
  same split; its counts are in `outputs/yolo_firearm/export_summary.json`.
- Known limitations of this data are listed, computed by code, in
  `outputs/report/limitations_facts.md` (also in the README).

## Demo footage

The Mendeley firearm-action videos (last section) are used only by `video_demo_app.py`.
They are not part of any split or reported result.

## Earlier candidate review (not used)

Before the Roboflow dataset was chosen, these sources were reviewed from their published
pages only; none was downloaded or used. `datasets/sources.json` records that metadata.

| Candidate | Published evidence and acquisition decision |
| --- | --- |
| Dangerous Items v2 | The publisher lists a 357.8 MB Pascal VOC archive and a 75/25 test split. Confirm media rights before downloading. Preserve the official test partition if the export can be grouped without overlap. [Zenodo](https://zenodo.org/records/13786228) · [Record metadata](https://zenodo.org/api/records/13786228) |
| Roboflow Pistols | Version 1 lists download access. Public sample boxes are visible and the description includes cartoons and staged imagery. Verify export class names and original media terms before use. [Project](https://universe.roboflow.com/joseph-nelson/pistols) · [Version 1](https://universe.roboflow.com/joseph-nelson/pistols/dataset/1) |
| CCTV Gun Detector | Version 1 lists YOLO and VOC formats. Verify the actual class definitions, media rights and original recording identifiers before use. [Project](https://universe.roboflow.com/mohammadreza-anvari-h6ase/cctv-gun-detector) · [Version 1](https://universe.roboflow.com/mohammadreza-anvari-h6ase/cctv-gun-detector/dataset/1) |
| OD-WeaponDetection | Start here for a bounded manual audit. The README declares CC BY-SA 4.0 and the license file states CC BY 4.0. Resolve the discrepancy and confirm underlying media rights. One VOC XML contains two `pistol` boxes. Its paired image and archive-wide annotation quality remain unverified. [README license](https://github.com/ari-dasci/OD-WeaponDetection#license) · [License file](https://github.com/ari-dasci/OD-WeaponDetection/blob/master/License.md) · [Data tree](https://github.com/ari-dasci/OD-WeaponDetection/tree/master/Pistol%20detection) |

## Generic preparation tools (Phase 2)

The tools below (`scripts/inspect_dataset.py`, `scripts/prepare_dataset.py`) are a general VOC/YOLO preparation path from the dashboard phase. The ML experiment does not use them; it uses `weapon_detection/ml/prepare_data.py`.

### Local workflow

Obtain a reviewed subset directly from the upstream source and preserve its original
source and split information under `datasets/raw/`. Record version, file hashes,
download date, license and attribution. If a download page requires an account, use its
manual export and keep access tokens outside the repo. Use a consistent folder layout
such as `images/<split>/` with `labels/<split>/` or colocated filenames in each split.
For other layouts, organize a working copy into matching relative stems.
Tests cover VOC XML and YOLO bounding-box TXT. COCO JSON is outside the parser scope.

```bash
uv run python scripts/inspect_dataset.py datasets/raw/reviewed \
  --report outputs/dataset_inspection.json --firearm-label gun --firearm-label rifle
uv run python scripts/prepare_dataset.py datasets/raw/reviewed \
  datasets/processed/firearm_detection --manifest datasets/preparation.json
```

The manifest below is a schema example. Supply actual source metadata and paths after
review. `rights_reviewed` is a local reviewer attestation and does not confer rights.

```json
{
  "source": {
    "name": "<dataset>",
    "url": "<official URL>",
    "version": "<version or commit>",
    "license": "<verified terms>",
    "rights_note": "<rights decision and attribution>",
    "rights_reviewed": true
  },
  "mapping": {"gun": "firearm", "rifle": "firearm", "knife": "exclude"},
  "groups": {"images/train/frame1.jpg": "recording-01"},
  "verified_negatives": ["images/train/reviewed_empty.jpg"]
}
```

YOLO imports use the source `data.yaml` or one supplied by `--names`. VOC defaults to
zero-based pixel edges. Use `--voc-one-based` on both commands for legacy 1-based inclusive
pixel boxes. Confirm the coordinate convention by overlaying a representative sample.
Invalid, zero-area, nonfinite and out-of-bounds boxes are reported and block export.
Empty labels count as negatives only when listed by the reviewer. Excluded labels
and skipped unverified negatives are counted. Review excluded classes and inspect each
negative for unlabeled firearms before listing it.

SHA-256 detects identical files. Identical files and source groups merge transitively
before assignment, even through an excluded image. Keep related crops, transcodes, and
augmentations together via group metadata. Visually review near duplicates. Source
separation is unverified for images lacking group metadata. The default seeded
70/15/15 targets are approximate because groups stay intact. With `--preserve-test`,
a directory component named `test` defines the official test partition and cross-split
duplicate or source conflicts cause an error. Use the preparation report to verify the
class balance and actual split counts. Every export contains `data.yaml` and
`preparation_report.json`. Use `--overwrite` only for an existing generated export.

## Video demo source: Mendeley firearm-related actions (demo only)

- Title: "Action recognition and object detection dataset for firearm-related actions"
- Authors: Jesus Ruiz-Santaquiteria, Juan D. Munoz, Francisco J. Maigler, Oscar Deniz,
  Gloria Bueno
- Mendeley Data, version 2, DOI [10.17632/bbzpxhd22j.2](https://doi.org/10.17632/bbzpxhd22j.2),
  <https://data.mendeley.com/datasets/bbzpxhd22j/2>
- Related paper: "Firearm-related action recognition and object detection dataset for
  video surveillance systems", Data in Brief 52 (2024) 110030,
  DOI [10.1016/j.dib.2024.110030](https://doi.org/10.1016/j.dib.2024.110030)
- Licence on the dataset page (Licence field): "CC BY NC 3.0". The page description
  says: "The dataset is licensed under CC BY NC 3.0. However, we do allow commercial use,
  but only if prior permission is obtained." (copied from the dataset page on 2026-11-23).
  Every `label.json` in the download also lists `CC BY NC 3.0`
  (<https://creativecommons.org/licenses/by-nc/3.0/deed.es>). The journal article itself
  is published under CC BY 4.0.
- Use here: non-commercial coursework demonstration only (`video_demo_app.py`). These
  videos are never used for training, validation, threshold selection or any reported
  metric, and they are outside every split. No video from this dataset is committed.

Layout found in the download (`datasets/raw/dataset.zip` contains one inner ZIP,
`Gun_Action_Recognition_Dataset.zip`): `Gun_Action_Recognition_Dataset/{Handgun,
Machine_Gun,No_Gun}/<video folder>/video.mp4`, 141 + 139 + 118 = 398 videos, 2.28 GB.
Handgun and Machine_Gun folders have a COCO-style `label.json` (one category per file,
named after the class). No_Gun folders have no label file. Folder `Handgun/PCH5_C1_P1_V1_HB_1`
has only `label_antiguo.json`, so the loader treats it as unlabelled. Boxes are
`[x, y, w, h]` in pixels with a top-left origin. `images[].id` runs 1..N, where N is the
decoded frame count, so frame ids are 1-based; the loader converts them to 0-based.
Camera C1 videos checked are 640x480 at 25 fps; camera C2 videos checked are 1920x1080 at 30 fps.

Extract with `uv run python scripts/extract_mendeley_videos.py` into
`datasets/interim/mendeley_firearm_actions/` (git-ignored). The script streams the
inner ZIP without writing it to disk and checks every member path. On a disk with
little free space, `--max-per-class N` extracts all labels but only the first N videos
per class.

# Dataset preparation

[Source manifest](../datasets/sources.json) records published metadata, with nulls
for unverified fields. Source pages and example annotations were reviewed. No
candidate dataset archive was downloaded. Local images and annotations have been
counted only for generated test fixtures.

| Candidate | Published evidence and acquisition decision |
| --- | --- |
| Dangerous Items v2 | The publisher lists a 357.8 MB Pascal VOC archive and a 75/25 test split. Confirm media rights before downloading. Preserve the official test partition if the export can be grouped without overlap. [Zenodo](https://zenodo.org/records/13786228) · [Record metadata](https://zenodo.org/api/records/13786228) |
| Roboflow Pistols | Version 1 lists download access. Public sample boxes are visible and the description includes cartoons and staged imagery. Verify export class names and original media terms before use. [Project](https://universe.roboflow.com/joseph-nelson/pistols) · [Version 1](https://universe.roboflow.com/joseph-nelson/pistols/dataset/1) |
| CCTV Gun Detector | Version 1 lists YOLO and VOC formats. Verify the actual class definitions, media rights and original recording identifiers before use. [Project](https://universe.roboflow.com/mohammadreza-anvari-h6ase/cctv-gun-detector) · [Version 1](https://universe.roboflow.com/mohammadreza-anvari-h6ase/cctv-gun-detector/dataset/1) |
| OD-WeaponDetection | Start here for a bounded manual audit. The README declares CC BY-SA 4.0 and the license file states CC BY 4.0. Resolve the discrepancy and confirm underlying media rights. One VOC XML contains two `pistol` boxes. Its paired image and archive-wide annotation quality remain unverified. [README license](https://github.com/ari-dasci/OD-WeaponDetection#license) · [License file](https://github.com/ari-dasci/OD-WeaponDetection/blob/master/License.md) · [Data tree](https://github.com/ari-dasci/OD-WeaponDetection/tree/master/Pistol%20detection) |

## Local workflow

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

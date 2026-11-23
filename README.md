# Real-Time Weapon Detection and AI-Powered Threat Intelligence System

A local academic computer vision prototype for detecting visible firearms in video
and helping people review candidate incidents. The 48-hour MVP targets a
prerecorded-video demonstration with annotated detections, temporal confirmation,
evidence snapshots, SQLite history and CSV export.

## Status

The Streamlit dashboard runs CPU video inference, displays annotated frames,
creates reviewable incidents by temporal confirmation, and provides SQLite history,
analytics, and CSV downloads. The Phase 2 CLI remains available. See [project context](CONTEXT.md), [roadmap](PLANS.md) and [task checklist](TASKS.md).

## Setup and startup

Install uv, then run these commands from the repository root.

```sh
uv sync --locked
cp -n .env.example .env
YOLO_OFFLINE=true uv run streamlit run app.py
```

Open `http://127.0.0.1:8501`. Blank `WD_MODEL_PATH` is valid for startup. The
example settings document the `WD_*` overrides and the 200 MiB Streamlit upload ceiling.
The dashboard enforces the configured application limit within that ceiling.
The lockfile uses CPU PyTorch builds. Initial dependency setup can require network
access. Opening history and analytics requires only the local database. Inference requires a
trusted local checkpoint.

## Development checks

```sh
uv run pytest
uv run ruff check .
uv run python -c "import weapon_detection"
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The stack consists of Python 3.11, Streamlit, Ultralytics YOLO, PyTorch, OpenCV,
standard-library SQLite, NumPy, Pandas, Pillow, PyYAML, python-dotenv and Plotly.
uv manages dependencies, pytest runs tests, and Ruff checks source.

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point |
| `weapon_detection/detection/` | Model and video boundary |
| `weapon_detection/incidents/` | Alert and incident boundary |
| `weapon_detection/storage/` | SQLite and evidence boundary |
| `weapon_detection/ui/` | Dashboard boundary |
| `weapon_detection/utils/` | Settings loader and utilities |
| `tests/` | Configuration, imports and startup tests |
| `docs/` | Architecture, dataset preparation, evaluation and demo |

## Model and data

Use a trusted local checkpoint with verified firearm labels and documented provenance.
Inspect checkpoint class names before inference. Keep datasets, weights, private
footage and generated evidence out of Git and verify source licenses before using
them. The [dataset plan](docs/DATASET.md) and [evaluation protocol](docs/EVALUATION.md)
describe the planned validation process.

## Graphify

When Graphify is available, run `graphify update .` after source changes. Use
`graphify query "<question>"` for codebase questions and `graphify path "<A>" "<B>"` or
`graphify explain "<concept>"` for focused navigation. The generated `graphify-out/`
directory is ignored. See [agent instructions](AGENTS.md) for the complete workflow.

## Limitations and roadmap

Human reviewers interpret incidents. False positives and missed detections are expected
risks, and performance depends on the checkpoint, hardware, input codec and scene.
Phase 2 provides dataset preparation and local YOLO video processing. Phase 3
implements alerts, evidence, history and CSV export. See the end-to-end
[demonstration](docs/DEMO.md). Webcam support is optional.

## Phase 2 CPU inference

Use a trusted checkpoint. Ultralytics publishes the Open Images V7 nano model used
in the local smoke test. Review its [license terms](https://www.ultralytics.com/license)
for your intended use before downloading. The following download writes about 7.2 MB
to the Git-ignored model directory. Runtime loads the existing local file.

```bash
mkdir -p models
curl --fail --location --output models/yolov8n-oiv7.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-oiv7.pt
YOLO_OFFLINE=true uv run python scripts/run_video.py --model models/yolov8n-oiv7.pt --classes-only
YOLO_OFFLINE=true uv run python scripts/run_video.py --input datasets/samples/sample.mp4 \
  --model models/yolov8n-oiv7.pt --output outputs/annotated.mp4 --csv outputs/detections.csv \
  --device cpu --conf 0.35 --imgsz 640 --frame-stride 2
```

The verified checkpoint maps Handgun (238), Rifle (423), and Shotgun (457) to
`firearm`. `--model` and `--conf` override the existing `WD_MODEL_PATH` and
`WD_CONFIDENCE_THRESHOLD` settings. Other controls are CLI arguments. Exact
`--alias` values override the built-in label list. Use `YOLO_OFFLINE=true` to disable
Ultralytics online checks. CSV records are frame-level predictions with coordinates
in source pixels.

See [dataset preparation](docs/DATASET.md) for inspection and normalization,
[evaluation and optional cloud training](docs/EVALUATION.md) for commands and
metrics, and the [demo](docs/DEMO.md) for the verified smoke test and its limits.

## Use the dashboard

1. In **Video Monitoring**, upload a permitted prerecorded video or choose an approved
   local sample from `datasets/samples/`. Select an installed checkpoint and click
   **Start Analysis**.
2. Review preview frames and progress. Use **Open annotated video player** or download
   the WebM and frame-level CSV.
3. In **Incident History**, filter by review status, video source, or confidence.
   Inspect the saved frame, update the review decision and notes, and download all
   or filtered records. **Analytics** shows persisted run and incident totals.

The default rule requires 3 positive processed observations in a 5-observation
window, with confidence 0.50. A gap over 3 seconds clears candidate history.
An active episode rearms after 3 observed misses. A new incident also needs a
5-second cooldown measured from the previous emission in source time. Settings are
adjustable in the dashboard and their defaults are in `.env.example`.

Storage defaults to `outputs/incidents.sqlite3`, `outputs/evidence/<run-id>/`, and
`outputs/runs/<run-id>/`. Uploads are staged under `inputs/` and cleaned up after
analysis. See the [demo guide](docs/DEMO.md) for a disposable database, exact smoke
settings, and cleanup. Use the processed-frame limit to bound CPU work. The dashboard
processes synchronously and offers frame limits in place of an interactive Stop control.

## Automatic full-screen demonstration

```bash
YOLO_OFFLINE=true uv run streamlit run demo_app.py
```

Open the local URL printed by Streamlit and use **F11** for browser fullscreen.
The separate player automatically loops filename-sorted clips in
`datasets/demo_playlist/` using `models/yolov8n-oiv7.pt`. Add permitted MP4,
AVI, MOV, MKV or WebM clips with codecs OpenCV can read. The demo displays
fresh CPU predictions at the rate they finish, without generating video files
or incident records. See the [playlist setup and measured results](docs/DEMO.md).

## Video demo (Mendeley firearm-action footage)

`video_demo_app.py` plays videos from the public Mendeley dataset "Action recognition and
object detection dataset for firearm-related actions" (DOI 10.17632/bbzpxhd22j.2, CC BY NC
3.0, demo use only; see [Dataset](docs/DATASET.md)). Green boxes are the dataset's
annotations, and red boxes are model predictions. The footage is outside every split, and
no threshold or setting was chosen on it.

1. In a browser, open <https://data.mendeley.com/datasets/bbzpxhd22j/2>, choose
   "Download All" and save the ZIP as `datasets/raw/dataset.zip`.
2. Choose the 10 demo videos from the dataset's own annotations (4 Handgun and 4
   Machine_Gun with the most visible, largest GT boxes; 2 No_Gun drawn with seed 42; no
   model involved). Then extract them. Leave out `--selection` to extract all 398 videos.
3. Precompute the model boxes once (confidence floor 0.05, imgsz 640, CPU, batch 16).
   This is resumable, and it also pre-renders the two MP4s per video.

```bash
uv run python scripts/select_demo_videos.py
uv run python scripts/extract_mendeley_videos.py --selection outputs/demo_cache/demo_selection.json
uv run python scripts/precompute_demo.py --benchmark   # per-frame vs batch 16, time estimate
nohup uv run python scripts/precompute_demo.py > outputs/demo_cache/precompute.nohup.out 2>&1 &
tail -f outputs/demo_cache/precompute.log              # progress and ETA (Ctrl-C stops tail only)
pkill -INT -f scripts/precompute_demo.py               # stop; rerun the nohup line to resume
uv run streamlit run video_demo_app.py
uv run python scripts/demo_diagnostics.py              # facts -> outputs/demo_cache/diagnostics.md
uv run python scripts/smoke_video_demo.py              # headless check
```

Selecting a video shows it at once. The confidence slider filters the cached boxes and
re-renders without calling the model (measured: about 1.5 s for 640x480 videos, about
11 s for the 1920x1080 camera-C2 videos). A video without a cache offers
"Run model on this video now". Caches are in `outputs/demo_cache/` and renders in
`outputs/demo_videos/`; both are git-ignored. Rendering needs ffmpeg on PATH or
`uv add imageio-ffmpeg`.

## ML experiment (classical image classification)

A separate course experiment in `weapon_detection/ml/` asks whether an image contains a
firearm. It uses the Roboflow "Weapon Detection-1" v4 YOLOv8 export (CC BY 4.0 per its
`data.yaml`). Place the ZIP at `datasets/raw/Weapon Detection.zip`. All generated files go to
the git-ignored `outputs/ml/` folder.

```sh
uv run python -m weapon_detection.ml.prepare_data  # extract, label, leakage audit, grouped split
uv run python -m weapon_detection.ml.features      # pixel features + HOG -> outputs/ml/features.npz
uv run python -m weapon_detection.ml.eda           # figures + eda_notes.md (train split only)
uv run python -m weapon_detection.ml.classical     # baseline, LR, RF: tune on val, test once
uv run python -m weapon_detection.ml.yolo_data     # one-class YOLO export + Colab zip
uv run python -m weapon_detection.ml.yolo_eval --checkpoint models/yolov8n-oiv7.pt \
  --name yolov8n_oiv7_pretrained --label "YOLOv8n pretrained (Open Images, not fine-tuned)"
uv run pytest -q tests/test_ml_*.py
```

Fine-tuning runs on a GPU. Upload `outputs/yolo_colab/yolo_firearm.zip` to Google Drive
and run `notebooks/03_yolo_colab.ipynb` in Colab. To train locally instead, run
`uv run python -m weapon_detection.ml.yolo_train`, adding `--cpu-preset` on a CPU.
Put the downloaded `best.pt` at `models/firearm_yolov8n_best.pt` and evaluate it on the
same images:

```sh
uv run python -m weapon_detection.ml.yolo_eval --checkpoint models/firearm_yolov8n_best.pt \
  --name yolov8n_finetuned --label "YOLOv8n fine-tuned (firearm)"
```

Image-level rows go to `outputs/ml/results/classical_metrics.csv`. Detection metrics
(mAP50, mAP50-95, precision and recall) go to a separate
`<name>_detection_metrics.json` file and are never mixed with the image-level metrics.

`firearm_present` is 1 when an image has a `Gun` or `Pistol` box. The firearm set is
`FIREARM_CLASS_NAMES` in `weapon_detection/ml/common.py`. The split uses
split v2 rule in `weapon_detection/ml/split_v2.py`. That rule puts the 3 largest
all-negative groups in train. It then draws val and test negatives at random (seed 42) from
at least 8 groups each, with at least 90 images per split and no group above 25%. It
assigns positive groups at random, so val and test each get about 10% of the positive
images. Split v2 is a revision made after the preliminary split v1 test results. Split v1
used `StratifiedGroupKFold`; its artefacts are kept in `outputs/ml/split_v1/` and in the
manifest column `split_v1`. Groups merge Roboflow source names, SHA-256 duplicates and dHash
near duplicates (Hamming distance of 3 or less). They also merge filename families, which
are video-frame sequences such as `KravMagaTraining###`. A family needs a prefix of at
least 4 characters shared by at least 5 images.

`classical.py` fits each scikit-learn pipeline on train only. It picks hyperparameters
and the decision threshold by validation macro-F1 and writes them to
`outputs/ml/chosen_settings.json` before it evaluates on test. Results go to
`outputs/ml/results/`, figures to `outputs/ml/figures/`, `leakage_audit.md` and
`error_analysis.csv` to `outputs/ml/`, and fitted pipelines to the git-ignored
`models/classical/` folder.

### Comparison, notebooks, report evidence and the audience GUI

```sh
uv run python -m weapon_detection.ml.compare   # final tables, fig_01..., report_numbers.json
uv run python scripts/make_notebooks.py        # (re)write notebooks 01, 02, 04
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_classical_models.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_comparison.ipynb
uv run streamlit run predict_app.py            # audience-input prediction page (CPU)
uv run python scripts/smoke_predict_app.py     # headless check -> outputs/ml/gui_smoke_test.json
```

`compare.py` does not retrain anything. It recomputes every metric from the saved
prediction CSVs and fails if any value differs from `classical_metrics.csv`. It writes
these files:

- `outputs/ml/results/final_{test,val}_table.{csv,md}`
- `detection_metrics.md`, kept separate from the image-level tables
- numbered figures `outputs/ml/figures/fig_NN_*.png`
- `figure_index.md`, `requirements_check.md` and `report_numbers.json`
- the evidence pack in `outputs/report/`, which has outlines, facts and placeholders
  but no conclusions

`predict_app.py` accepts an uploaded image or a test-split image and lists only the
models whose files exist. It uses the same feature function as training, and it adds the
fine-tuned YOLO option automatically once `models/firearm_yolov8n_best.pt` exists and
`yolo_eval.py` has been run on it.

After downloading the fine-tuned `best.pt` from Colab, run these commands:

```sh
cp ~/Downloads/best.pt models/firearm_yolov8n_best.pt
uv run python -m weapon_detection.ml.yolo_eval --checkpoint models/firearm_yolov8n_best.pt \
  --name yolov8n_finetuned --label "YOLOv8n fine-tuned (firearm)"
uv run python -m weapon_detection.ml.compare
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_comparison.ipynb
uv run python scripts/smoke_predict_app.py
uv run python -m weapon_detection.ml.compare   # refresh the checklist after the smoke test
```

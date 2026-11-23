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

# Real-Time Weapon Detection and AI-Powered Threat Intelligence System

A local academic computer vision prototype for detecting visible firearms in images and
video and helping people review candidate incidents. It has two parts:

- **ML course experiment** (`weapon_detection/ml/`): does an image contain a firearm?
  Classical models (majority baseline, logistic regression, random forest on handcrafted
  and HOG features) are compared with YOLOv8n, both pretrained on Open Images V7 and
  fine-tuned on the firearm dataset. Every number is computed and saved by code.
- **Applications**: a Streamlit dashboard for prerecorded video with temporal alerts,
  SQLite incident history and CSV export (`app.py`); a full-screen playlist player
  (`demo_app.py`); an audience image-prediction page (`predict_app.py`); and a video demo
  on public footage (`video_demo_app.py`).

This is a research prototype, not a safety system. A prediction alone cannot establish
that a firearm is present. Human review is required.

See [project context](CONTEXT.md), [roadmap](PLANS.md), [task checklist](TASKS.md),
[dataset](docs/DATASET.md), [evaluation](docs/EVALUATION.md),
[architecture](docs/ARCHITECTURE.md) and [demo](docs/DEMO.md).

## Results

<!-- BEGIN GENERATED: results -->
Source: `outputs/ml/results/final_test_table.md` (copied by `scripts/update_readme_results.py`). These are IMAGE-LEVEL classification metrics (firearm present yes/no) on the TEST split of split v2.

### Image-level results on the TEST split

Firearm = positive class. neg_* = no-firearm class. Threshold chosen on validation. ms/image: classical = model only on precomputed features; YOLO = read + inference.

| model | family | main_model | precision | recall | f1 | accuracy | balanced_accuracy | macro_f1 | neg_precision | neg_recall | neg_f1 | roc_auc | pr_auc | threshold | predict_ms_per_image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | trivial (DummyClassifier) |  | 0.9023 | 1.0000 | 0.9486 | 0.9023 | 0.5000 | 0.4743 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.9023 |  | 0.0006 |
| Logistic Regression (handcrafted) | linear model |  | 0.9216 | 0.9194 | 0.9205 | 0.8567 | 0.5986 | 0.5976 | 0.2717 | 0.2778 | 0.2747 | 0.6371 | 0.9382 | 0.3200 | 0.0013 |
| Random Forest (handcrafted) | tree ensemble |  | 0.9290 | 0.9290 | 0.9290 | 0.8719 | 0.6367 | 0.6367 | 0.3444 | 0.3444 | 0.3444 | 0.7257 | 0.9494 | 0.4900 | 0.2478 |
| Logistic Regression (handcrafted + HOG) | linear model | yes | 0.9335 | 0.8953 | 0.9140 | 0.8480 | 0.6532 | 0.6299 | 0.2984 | 0.4111 | 0.3458 | 0.7449 | 0.9583 | 0.4700 | 0.0105 |
| Random Forest (handcrafted + HOG) | tree ensemble | yes | 0.9344 | 0.9434 | 0.9389 | 0.8893 | 0.6662 | 0.6729 | 0.4268 | 0.3889 | 0.4070 | 0.7893 | 0.9677 | 0.7600 | 0.2611 |
| YOLOv8n pretrained (Open Images, not fine-tuned) | CNN object detector |  | 0.9781 | 0.4308 | 0.5982 | 0.4777 | 0.6710 | 0.4262 | 0.1477 | 0.9111 | 0.2543 | 0.6661 | 0.9343 | 0.0100 | 107.7535 |
| YOLOv8n fine-tuned (firearm) | CNN object detector | yes | 0.9647 | 0.9543 | 0.9595 | 0.9273 | 0.8160 | 0.8025 | 0.6162 | 0.6778 | 0.6455 | 0.9211 | 0.9887 | 0.1200 | 146.6271 |

Source: `outputs/ml/results/detection_metrics.md`. These are DETECTION metrics (boxes, mAP) and are not comparable with the image-level table.

### Detection metrics (separate from image-level metrics)

| checkpoint_key | checkpoint | split | imgsz | mAP50 | mAP50_95 | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8n_finetuned | models/firearm_yolov8n_best.pt | test | 640 | 0.7421 | 0.4298 | 0.7836 | 0.6602 |
<!-- END GENERATED: results -->

Columns: `precision`, `recall` and `f1` are for the firearm class. `neg_*` are the same
metrics for the no-firearm class. `macro_f1` is the mean of both F1 scores. `threshold`
was chosen on the validation split (`outputs/ml/chosen_settings.json`). For YOLO, the
image score is the highest box confidence in the image. Detection metrics come from
Ultralytics validation on the same TEST images. Figures are in `report_assets/figures/`
(copies of `outputs/ml/figures/fig_*.png`). The validation table is in
`report_assets/tables/final_val_table.md`.

## Limitations

<!-- BEGIN GENERATED: limitations -->
Source: `outputs/report/limitations_facts.md`.

Computed by `weapon_detection/ml/report_pack.py`.

- Positive share per split (v2): train 87.6%, val 90.2%, test 90.2%.
- Negative images: val 90, test 90.
- 1121 negative images in total; 1094 contain other annotated objects and 27 have no boxes.
- Objects inside negative images: Knife 860, Currency Notes 192, Mobile 88, Card 20.
- 776 negative images belong to filename families (video-frame sequences or photo series).
- All images come from one Roboflow project (https://universe.roboflow.com/weapon-detection-oss9n/weapon-detection-1-f9p7t/dataset/4).
- 31.0% of 8605 train firearm boxes cover less than 5% of the image.
- Classical models resize every image to 128x128 regardless of aspect ratio.
- The split was revised (v1 -> v2) after preliminary v1 test results were seen; v1 results are kept in outputs/ml/split_v1/.
- Hyperparameters and thresholds were chosen on the validation split only.
- YOLOv8n pretrained: validation-chosen threshold 0.01 lies below the 0.05 confidence floor used for prediction.
- Classical feature extraction takes 45.67 ms per image on this CPU; YOLO latencies are in final_test_table.csv.
- No evaluation on real CCTV footage or video streams was performed.
- Fine-tuned YOLO: evaluated.
<!-- END GENERATED: limitations -->

Further limitations of the applications:

- The temporal alert defaults (3 positive observations in a 5-observation window,
  3-second maximum gap, 3 misses to rearm, 5-second cooldown) were set by hand. They were
  **not tuned on any data**.
- The per-box confidence defaults (dashboard 0.50, full-screen player 0.35, video demo
  0.25) are display settings. They were **not tuned on any data**. They are separate from
  the validation-chosen image-level threshold, which `predict_app.py` uses.
- The applications have been checked with smoke clips and the Mendeley demo footage only.
  No incident-level accuracy was measured.

## Reproducing

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required. The lockfile pins CPU
PyTorch builds. The first `uv sync` needs network access.

```sh
uv sync --locked
```

1. Download the Roboflow export "Weapon Detection-1" version 4 in YOLOv8 format from
   <https://universe.roboflow.com/weapon-detection-oss9n/weapon-detection-1-f9p7t/dataset/4>
   and save the ZIP as `datasets/raw/Weapon Detection.zip`. Attribution and licence are in
   [NOTICE.md](NOTICE.md) and [docs/DATASET.md](docs/DATASET.md).
2. Run the pipeline in this order. All outputs go to the git-ignored `outputs/` folder.

```sh
uv run python -m weapon_detection.ml.prepare_data  # extract, label, leakage audit, split v2
uv run python -m weapon_detection.ml.features      # handcrafted + HOG -> outputs/ml/features.npz
uv run python -m weapon_detection.ml.eda           # figures + eda_notes.md (train split only)
uv run python -m weapon_detection.ml.classical     # baseline, LR, RF: tune on val, test once
uv run python -m weapon_detection.ml.yolo_data     # one-class YOLO export + Colab zip
# Colab (GPU): run notebooks/03_yolo_colab.ipynb -> models/firearm_yolov8n_best.pt
uv run python -m weapon_detection.ml.yolo_eval --checkpoint models/yolov8n-oiv7.pt \
  --name yolov8n_oiv7_pretrained --label "YOLOv8n pretrained (Open Images, not fine-tuned)"
uv run python -m weapon_detection.ml.yolo_eval --checkpoint models/firearm_yolov8n_best.pt \
  --name yolov8n_finetuned --label "YOLOv8n fine-tuned (firearm)"
uv run python -m weapon_detection.ml.compare       # final tables, figures, report_numbers.json
uv run python scripts/build_report_assets.py       # copy final evidence into report_assets/
uv run python scripts/update_readme_results.py     # refresh the generated README blocks
```

`models/yolov8n-oiv7.pt` is the official Ultralytics Open Images V7 nano checkpoint.
Review its [licence terms](https://www.ultralytics.com/license), then download it:

```sh
mkdir -p models
curl --fail --location --output models/yolov8n-oiv7.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-oiv7.pt
```

Measured runtimes:

<!-- BEGIN GENERATED: runtimes -->
Measured on this laptop's CPU by `scripts/reproduce_in_copy.sh` (a fresh copy of the tracked files; source `outputs/reproduce/pipeline_runtimes.tsv`). Other jobs shared the CPU during this run, so the times are upper bounds. Colab training is not included.

| step | seconds | exit code |
|---|---|---|
| prepare_data | 122.1 | 0 |
| features | 121.9 | 0 |
| eda | 12.2 | 0 |
| classical | 405.4 | 0 |
| yolo_data | 19.8 | 0 |
| yolo_eval_pretrained | 590.5 | 0 |
| yolo_eval_finetuned | 627.8 | 0 |
| compare | 21.7 | 0 |
<!-- END GENERATED: runtimes -->

`bash scripts/reproduce_in_copy.sh /tmp/wd_repro` reruns every step above except Colab
training in a fresh copy of the tracked files. It times each step and compares the copy's
final TEST table with the repository's.

How the experiment is set up:

- `firearm_present` is 1 when an image has a `Gun` or `Pistol` box (`FIREARM_CLASS_NAMES`
  in `weapon_detection/ml/common.py`). Knife, card, currency-note and phone images are
  negatives.
- Groups merge Roboflow source names, SHA-256 duplicates, dHash near duplicates (Hamming
  distance of 3 or less) and filename families (video-frame sequences such as
  `KravMagaTraining###`, a shared prefix of at least 4 characters and at least 5 images).
  No group spans two splits.
- **Split v2** (`weapon_detection/ml/split_v2.py`) is a revision made **after the
  preliminary split v1 test results were seen**. Split v1 used `StratifiedGroupKFold`.
  Its results are kept and disclosed in `outputs/ml/split_v1/`, in the manifest column
  `split_v1`, and in `outputs/ml/results/split_v1_vs_v2_test.csv`.
  - Split v2 puts the 3 largest all-negative groups in train.
  - It then draws val and test negatives at random (seed 42) from at least 8 groups
    each, with at least 90 images per split and no group above 25%.
  - It assigns positive groups at random, so val and test each get about 10% of the
    positive images.
- Hyperparameters and decision thresholds are chosen by validation macro-F1 only. Each
  model is evaluated on the TEST split once (`outputs/ml/chosen_settings.json`).
- YOLO fine-tuning uses the same split. The train export has all train negatives plus a
  seeded sample of positives, capped at `DEFAULT_TRAIN_CAP` images in
  `weapon_detection/ml/yolo_data.py`.
- Training ran in Colab from COCO-pretrained `yolov8n.pt` (30 epochs, imgsz 640,
  batch 16, patience 10, seed 42). `uv run python -m weapon_detection.ml.yolo_train`
  is a local alternative; add `--cpu-preset` on a CPU.
- `compare.py` retrains nothing. It recomputes every metric from the saved prediction
  CSVs and fails if a value differs from `classical_metrics.csv`. It writes the final
  tables, the numbered figures, `figure_index.md`, `requirements_check.md`,
  `report_numbers.json` and the facts-only evidence pack in `outputs/report/`.

Notebooks:

- 01 data and EDA, 02 classical models, and 04 comparison have stored outputs. Regenerate
  them with `uv run python scripts/make_notebooks.py`, then run
  `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.
- Notebook 04 links its figures from `report_assets/figures/` instead of embedding them.
- `notebooks/03_yolo_colab.ipynb` is a Google Colab notebook. It needs a GPU runtime and
  the uploaded `outputs/yolo_colab/yolo_firearm.zip`.

## Getting the fine-tuned weights

Model weights are not committed (`*.pt` and `models/` are git-ignored).
`models/firearm_yolov8n_best.pt` is the `best.pt` produced by
`notebooks/03_yolo_colab.ipynb`. Download it from the Colab run's Google Drive folder
(`MyDrive/yolo_results/best.pt`) and place it at `models/firearm_yolov8n_best.pt`.

TODO: add the public download link for `firearm_yolov8n_best.pt` here.

## Applications

All apps run on CPU and pick the checkpoint with the same rule (`default_checkpoint` in
`weapon_detection/detection/model.py`): `models/firearm_yolov8n_best.pt` if it exists,
otherwise `models/yolov8n-oiv7.pt`. The adapter reads the checkpoint's class names and
maps only firearm labels (`firearm` in the fine-tuned model; `Handgun`, `Rifle` and
`Shotgun` in Open Images V7). It rejects a checkpoint that has no firearm label.

```sh
cp -n .env.example .env
YOLO_OFFLINE=true uv run streamlit run app.py             # dashboard
YOLO_OFFLINE=true uv run streamlit run demo_app.py        # full-screen playlist player
uv run streamlit run predict_app.py                       # audience image prediction
uv run streamlit run video_demo_app.py                    # Mendeley video demo
```

- **Dashboard (`app.py`).**
  - In **Video Monitoring**, upload a prerecorded video or choose a sample from
    `datasets/samples/`. The checkpoint selector defaults to the rule above;
    `WD_MODEL_PATH` overrides it. Then click **Start Analysis**.
  - **Incident History** filters, reviews and exports incidents. **Analytics** shows
    totals.
  - The per-box confidence default is `DEFAULT_BOX_CONFIDENCE` in
    `weapon_detection/utils/config.py` (overridable with `WD_CONFIDENCE_THRESHOLD`). The
    alert defaults are the `DEFAULT_ALERT_*` constants there; see `.env.example`.
  - Storage defaults to `outputs/incidents.sqlite3`, `outputs/evidence/<run-id>/` and
    `outputs/runs/<run-id>/`.
- **Full-screen player (`demo_app.py`).** Loops the clips in `datasets/demo_playlist/`
  and draws boxes at `DEMO_BOX_CONFIDENCE` (overridable with `WD_DEMO_CONFIDENCE`). Its
  status line reports boxes at that confidence, not an image-level decision. It writes no
  files.
- **Audience page (`predict_app.py`).** Shows an image-level decision for an uploaded or
  test-split image, using each model's validation-chosen threshold from
  `outputs/ml/chosen_settings.json`. For the fine-tuned YOLO model that threshold is 0.12.
- **CLI.**
  `YOLO_OFFLINE=true uv run python scripts/run_video.py --input <clip> --output outputs/annotated.mp4 --csv outputs/detections.csv`
  writes an annotated video and a frame-level CSV (`--model` overrides the checkpoint).

## Video demo

`video_demo_app.py` is a **demo-only** tool. It plays videos from the public Mendeley
dataset "Action recognition and object detection dataset for firearm-related actions"
(DOI 10.17632/bbzpxhd22j.2, CC BY NC 3.0; see [NOTICE.md](NOTICE.md)).

- Green boxes are the dataset's annotations. Red boxes are the model's predictions.
- This footage is outside every split and is **not part of any reported result**. **No
  threshold or setting was chosen on it.**
- The demo uses the 10 videos listed in `outputs/demo_cache/demo_selection.json`. They
  were chosen from the dataset's own annotations with no model involved: 4 Handgun and 4
  Machine_Gun with the most visible, largest boxes, and 2 No_Gun drawn with seed 42.
- The precomputed detections in `outputs/demo_cache/` and the rendered videos in
  `outputs/demo_videos/` are git-ignored. `scripts/select_demo_videos.py` and
  `scripts/precompute_demo.py` regenerate them.

1. In a browser, open <https://data.mendeley.com/datasets/bbzpxhd22j/2>, choose
   "Download All" and save the ZIP as `datasets/raw/dataset.zip`.
2. Select, extract and precompute (confidence floor 0.05, imgsz 640, CPU, batch 16;
   resumable), then start the app:

```sh
uv run python scripts/select_demo_videos.py
uv run python scripts/extract_mendeley_videos.py --selection outputs/demo_cache/demo_selection.json
uv run python scripts/precompute_demo.py --benchmark   # per-frame vs batch 16, time estimate
nohup uv run python scripts/precompute_demo.py > outputs/demo_cache/precompute.nohup.out 2>&1 &
tail -f outputs/demo_cache/precompute.log              # progress and ETA
pkill -INT -f scripts/precompute_demo.py               # stop; rerun the nohup line to resume
uv run streamlit run video_demo_app.py
uv run python scripts/demo_diagnostics.py              # facts -> outputs/demo_cache/diagnostics.md
uv run python scripts/smoke_video_demo.py              # headless check
```

The confidence slider filters the cached boxes and re-renders without calling the model.
Measured re-render times are about 1.5 s for 640x480 videos and about 11 s for the
1920x1080 camera-C2 videos. A video without a cache offers "Run model on this video now".
Rendering needs ffmpeg on PATH or `uv add imageio-ffmpeg`.

## Development checks

```sh
uv run pytest -q
uv run ruff check .
uv run python scripts/smoke_predict_app.py
uv run python scripts/smoke_video_demo.py
```

| Path | Purpose |
| --- | --- |
| `app.py`, `demo_app.py`, `predict_app.py`, `video_demo_app.py` | Streamlit entry points |
| `weapon_detection/ml/` | ML experiment: data, features, EDA, models, comparison |
| `weapon_detection/detection/` | YOLO adapter, video processing, demo cache and rendering |
| `weapon_detection/incidents/`, `storage/` | Alert engine, incidents, SQLite and evidence |
| `weapon_detection/ui/`, `utils/` | Dashboard views and validated settings |
| `notebooks/` | Notebooks 01, 02 and 04 (stored outputs); 03 runs in Colab |
| `report_assets/` | Tracked copies of final tables, figures and evidence files |
| `scripts/` | CLIs, smoke tests, video-demo tools and report helpers |
| `tests/` | Unit tests (CPU, no footage, no checkpoints needed) |

When Graphify is available, run `graphify update .` after source changes (see
[AGENTS.md](AGENTS.md)).

## Licence and attribution

Dataset attribution and licences are in [NOTICE.md](NOTICE.md). The code licence has not
been chosen yet (TODO in NOTICE.md).

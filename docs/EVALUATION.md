# Evaluation

## Measured results (ML experiment)

All reported numbers come from code and are saved under `outputs/ml/` (copies in
`report_assets/`). The final TEST table and the detection metrics are reproduced in the
README "Results" section by `scripts/update_readme_results.py`, which also asserts that
they equal `outputs/ml/report_numbers.json`.

- Task: image-level classification, firearm present yes/no, on the Roboflow "Weapon
  Detection-1" v4 data (see [Dataset](DATASET.md)).
- Split: split v2 (grouped, seed 42). It was revised after the preliminary split v1 test
  results were seen; v1 results are kept in `outputs/ml/split_v1/` and
  `outputs/ml/results/split_v1_vs_v2_test.csv`.
- Protocol: models are fitted on train; hyperparameters and the decision threshold are
  chosen by validation macro-F1 (`outputs/ml/chosen_settings.json`); each model is then
  evaluated once on TEST.
- Models: majority baseline; logistic regression and random forest on handcrafted and on
  handcrafted + HOG features; YOLOv8n pretrained on Open Images V7 (not fine-tuned); YOLOv8n
  fine-tuned on the one-class export in Colab (`notebooks/03_yolo_colab.ipynb`). For YOLO
  the image score is the highest box confidence.
- Image-level metrics (precision, recall, F1, macro-F1, balanced accuracy, ROC-AUC, PR-AUC,
  no-firearm precision/recall/F1, latency) are in `outputs/ml/results/final_test_table.md`
  and `final_val_table.md`. Detection metrics (mAP50, mAP50-95, box precision and recall)
  for the fine-tuned model are in `detection_metrics.md` and are kept separate.
- `weapon_detection/ml/compare.py` recomputes every metric from the saved prediction CSVs
  and fails on any mismatch. It re-measures the feature-extraction time on each run, so that
  one timing value changes between runs.
- Not measured: accuracy on real CCTV footage or video streams; incident-level (temporal
  alert) accuracy.

## Settings that were not tuned

- Temporal alert defaults (`DEFAULT_ALERT_*` in `weapon_detection/utils/config.py`: 3
  positive observations in a 5-observation window, 3 s maximum gap, 3 misses to rearm, 5 s
  cooldown) were set by hand and **not tuned on any data**.
- Per-box confidence defaults (`DEFAULT_BOX_CONFIDENCE` 0.50 in the dashboard,
  `DEMO_BOX_CONFIDENCE` 0.35 in `demo_app.py`, `DEFAULT_CONFIDENCE` 0.25 in the video demo)
  are display settings, **not tuned on any data**. The only validation-chosen threshold used
  by an application is the image-level threshold in `predict_app.py` (0.12 for the
  fine-tuned YOLO model, read from `chosen_settings.json`).

## Video demo (demo only, not an evaluation)

`video_demo_app.py` plays 10 Mendeley videos (`outputs/demo_cache/demo_selection.json`)
with the dataset's boxes and the model's boxes. That footage is outside every split, is not
part of any reported result, and was not used to choose any threshold or setting. The
statistics it shows (`outputs/demo_cache/threshold_summary.md`, `diagnostics.md`) are
labelled "Demo footage statistics, not an evaluation".

## Protocol notes (Phase 2 tools)

Evaluate a vetted checkpoint on a held-out test split grouped by source. Choose
thresholds on validation data before running the test. Record the checkpoint checksum,
class map, split version, image size, confidence, IoU and NMS settings, device, and
software versions. Keep threshold and image-size settings consistent across comparable
runs. Use one-to-one matching between predicted and reference boxes for counts at the
reported operating point.

| Metric | Method |
| --- | --- |
| Precision | TP / (TP + FP) at the stated operating point |
| Recall | TP / (TP + FN) at the stated operating point |
| F1-score | Harmonic mean of precision and recall |
| mAP@50 | Mean average precision at IoU 0.50 |
| mAP@50–95 | Mean AP over IoU thresholds from 0.50 to 0.95 in steps of 0.05 |
| Inference latency | Warm up the model, then report median and p95 in ms per frame |
| End-to-end throughput | Decoded and processed frames per second on the demo device |
| False positives | Count unmatched detections and review representative examples |
| False negatives | Count missed reference boxes and review representative examples |
| Independent video | Report correct episodes, false incidents, misses, and time to confirmation |

Use independent positive and negative clips with reviewer-annotated event intervals.
Record decoding failures, incident timing, snapshots, repeat-alert suppression, and
differences between the clips and training domain. Review both false alerts and misses
for patterns such as low light, occlusion, or lookalike objects. Document sample size
and undefined metrics where denominators are zero. Held-out image-level and detection
results for the ML experiment are listed above; independent-video results are not measured.

## Local evaluation commands

```bash
YOLO_OFFLINE=true uv run python scripts/evaluate_model.py --model models/yolov8n-oiv7.pt \
  --dataset datasets/processed/firearm_detection --match-iou 0.50 --conf 0.35 \
  --overlap-note "Record how training-source overlap was reviewed" \
  --output outputs/held_out_report.json

YOLO_OFFLINE=true uv run python scripts/evaluate_model.py --model models/yolov8n-oiv7.pt \
  --video datasets/samples/sample.mp4 --conf 0.35 --output outputs/video_benchmark.json
```

The evaluator reads normalized YOLO test annotations, verifies matching label files,
and checks exact duplicates across splits. It assigns predictions in descending
confidence to the highest-IoU unmatched ground truth. Every other prediction is a
false positive, and unmatched truth is a false negative. Record the underlying source
groups and checkpoint overlap separately. Undefined ratios and unavailable mAP are
JSON null. The Open Images checkpoint IDs differ from the normalized dataset class,
so its native Ultralytics validation metrics would require remapping. This tool
reports operating-point metrics for that checkpoint and leaves mAP unavailable.
For a compatible checkpoint with a single firearm class at ID 0, add
`--ultralytics-map` to run native validation on the generated test split with a
0.001 confidence floor. This optional path was tested with a mocked validator.

Video benchmarks report adapter wall time, including first-call setup when the video
is the first input. End-to-end elapsed time includes decode and annotation. The video
CLI also includes output writing and verification if enabled. Compare decoded and
processed FPS alongside the stride and frame limit. Time is estimated from the
nominal source frame rate. Accuracy on independent surveillance footage remains
unmeasured.

## Optional cloud training

On an approved Colab or Kaggle GPU runtime, use the provider's GPU-enabled PyTorch
installation and install the compatible Ultralytics version in that separate runtime.
Copy the approved prepared dataset and a trusted local nano starting checkpoint to
the runtime and update the absolute `path` in `data.yaml` to its new location. Keep
the train, validation, and test lists and source groups unchanged.

```bash
python scripts/train_model.py --data /path/to/export/data.yaml --model /path/to/yolo11n.pt \
  --epochs 20 --imgsz 640 --batch 8 --device 0 --output outputs/training
```

Record provider, package versions, seed, model checksum, and split manifest. Download
`weights/best.pt` from the run and put it under `models/` locally. Verify classes and
evaluate the held-out set on CPU before the demo. `scripts/train_model.py` was not used
for the reported model: the fine-tuned checkpoint `models/firearm_yolov8n_best.pt` was
trained with `notebooks/03_yolo_colab.ipynb` in Colab and evaluated with
`weapon_detection/ml/yolo_eval.py`.

The local fallback checkpoint used for smoke testing has SHA-256
`3851dfbf39ed2a076b1f39215cc22dba64eb5646f282bf964703785b0bed6a41`.
The gray-clip benchmark recorded six processed frames, zero detections, and 5.78 s
elapsed at image size 640, confidence 0.25 and stride 1. The benchmark report is
local under `outputs/video_benchmark.json`. These measurements describe this run
and its startup overhead.

## Phase 3 integration observations

These observations predate the fine-tuned model. The packaged Open Images checkpoint and public-domain repeated-image fixture verified
the temporal rule, persistence, evidence hashing, exports, and dashboard display. At
confidence 0.25, 640-pixel inference, stride 1, and a 3-of-5 rule, six actual Handgun
predictions produced one incident at source frame 2. A gray clip produced zero
predictions or incidents. Neither clip provides a representative sample of false
alarms, misses, or accuracy. A persistent false positive can still meet the temporal
rule. Report incident-level results separately when permitted independent clips with
episode annotations become available.

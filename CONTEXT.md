# Project context

## Purpose and MVP

Real-Time Weapon Detection and AI-Powered Threat Intelligence System is an academic
computer vision and incident review prototype. The intended users are student
demonstrators, evaluators, and reviewers of detection evidence. The system marks visible
firearms in video and queues candidate incidents for human judgment. A bounding box
represents a model prediction about an object. Incident records indicate that an alert
rule fired and require review.

A successful demonstration runs locally with a vetted checkpoint and a permitted
prerecorded clip. The user selects the upload source and model, starts processing,
sees bounding boxes, confidence scores, timestamps and alert status, and can review a
confirmed incident with a saved snapshot. The incident survives an application restart,
appears in history, and exports to CSV. Show clear errors for invalid input and present
measured evaluation results with limitations. Webcam selection is optional if local
hardware and time permit.

## User journeys and requirements

- **Upload and detect.** Select a video, choose an available local model, and start a run.
  Validate upload size and codec support. Extract frames sequentially with OpenCV, run
  YOLO, and display filtered firearm detections on annotated frames.
- **Review alerts.** Show confidence and confirmation state. Confirm an incident when
  enough processed observations qualify within the configured sliding window.
  Show the timestamp and annotated evidence so the reviewer can inspect the alert.
- **Browse and export history.** Read persisted incidents from SQLite, open their
  snapshots, and export incident records as CSV.
- **Configure and recover.** Validate startup settings and surface missing checkpoints,
  unreadable video, and persistence errors. Close capture handles and clean temporary
  uploads on success and failure. Verify key behavior with automated tests and an
  independent video exercise.

Keep memory bounded by streaming frames and record CPU throughput and latency. Aim
for a responsive demonstration within the 48-hour implementation window. Keep setup
reproducible via uv.lock, keep model and data provenance documented, and keep processing
and evidence local. Use mock inference for unit testing and describe known model limits
in the UI and demo. Real-time performance is a measurement objective.

## Stack

Python 3.11 and uv provide a repeatable Python-only environment. Streamlit renders the
monitor and history. Ultralytics runs YOLO with PyTorch. OpenCV decodes and annotates
video. NumPy represents frames and Pillow supports image handling where needed.
Python's SQLite library persists incidents. Pandas and Plotly support tabular and
chart views when useful. python-dotenv supplies local environment settings and PyYAML
supports dataset manifests. pytest and Ruff provide fast development checks.

## Detection and data

Start with one normalized class, `firearm`. Separate handgun, rifle, and shotgun
classes are a later option if suitable data supports them. Prefer a small YOLO detector
fine-tuned on verified firearm annotations if time and compute permit. Prepare an
existing Open Images V7-pretrained firearm-capable checkpoint as a fallback, subject to
license, provenance, and local validation. Inspect actual checkpoint class names before
using it, map approved labels explicitly, and reject incompatible checkpoints. Generic
COCO weights must never be assumed to support firearm detection.

Research candidates include Dangerous Items Dataset v2, Roboflow Pistols, CCTV Gun
Detector and OD-WeaponDetection. Source-level licensing observations are in
`datasets/sources.json`. No candidate archive has been acquired. Follow the source grouping and validation process in
[Dataset](docs/DATASET.md).

## Alerts and storage

Filter by configured confidence and count positive processed observations within a
sliding window. Confirm at most one incident until enough consecutive processed
misses reset the episode. Maximum gap and cooldown use prerecorded source time. Consolidate a sustained
qualifying episode into one reviewable incident. Store source reference, timestamps,
confidence, and evidence path in SQLite. Save annotated snapshots under the configured
evidence directory. Keep SQLite and evidence paths configurable and preserve their
consistency on failure.

## Evaluation and risks

Measure precision, recall, F1-score, mAP@50, mAP@50–95, inference latency, and false
positives and negatives. Use independent positive and negative clips for the end-to-end
evaluation. Report the checkpoint, data split, device, and settings with each result.
Phase 2 measured CPU smoke-test timing. Held-out accuracy and independent video
evaluation are pending.

Risks include domain mismatch, poor annotations, limited GPU access, false alarms,
missed detections, slow processing of large videos, codec compatibility, and the
48-hour schedule. Prioritize a short validated clip and CPU-compatible fallback.
Facial recognition, person identification, criminal-intent prediction, automatic
law-enforcement dispatch, physical access control, autonomous defensive responses,
cloud surveillance, and distributed multi-camera orchestration are out of scope.

## Current status

Phase 1 provides the package layout, validated settings, Streamlit status page, locked
dependencies, baseline tests, and development documentation. Phase 2 implements
CPU YOLO video inference, video and CSV output, dataset inspection and preparation,
and fixed-threshold metrics. The official Open Images V7 nano checkpoint was verified
on CPU with Handgun (238), Rifle (423), and Shotgun (457). Phase 3 integrates alerts,
JPEG evidence, SQLite, Streamlit history, analytics, and CSV exports. Webcam support depends on hardware and codec testing.

The Phase 1 environment was verified on Linux with Python 3.11.16, PyTorch
2.15.0+cpu, OpenCV 5.1.0, Ultralytics 8.4.216, Streamlit 1.67.0, and SQLite 3.53.1.
PyTorch reported CUDA unavailable. The pre-existing local environment was missing
OpenCV package files. `uv sync --reinstall-package opencv-python` restored the locked
version, and library imports passed afterward. All 16 tests and Ruff passed.
The headless Streamlit health endpoint returned HTTP 200. Graphify 0.9.59 generated
and queried a local AST graph. The existing Codex hook points to a machine-specific
executable and may need adjustment on another workstation.

Use the Phase 2 CLI commands in README and the fixture details in docs/DEMO.md.
The checkpoint and outputs are local and Git-ignored. The smoke test covers a repeated
public-domain photograph and a gray clip. Dataset rights and accuracy still require
validation on the intended surveillance domain.

Phase 3 verified actual upload, WebM playback and evidence display in headless Chrome.
At stride 2, the repeated-photo fixture yielded three Handgun predictions and one
incident. The gray fixture yielded zero. Review and CSV persistence were verified
with the application and repository. These are integration checks. The dashboard
runs synchronously and retains completed run files locally.

## Automatic presentation screen

`demo_app.py` is a separate Streamlit entry point for looping trusted local clips
under `datasets/demo_playlist/`. It reuses the cached CPU detector, capture iterator,
and frame annotation. It shows inference frames as they finish and writes no videos
or incidents. Chrome verified the existing positive and gray smoke fixtures. Longer
approved presentation footage remains to be added.

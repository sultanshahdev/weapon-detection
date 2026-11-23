# Tasks

## Current priority

Use the verified OIV7 checkpoint for the prerecorded demo. Acquire a rights-reviewed
held-out dataset and independent clips to measure accuracy. Update this file as tasks
change status.

## Completed

- [x] Inspect existing files, dependency configuration, and Graphify hook.
- [x] Document project requirements, architecture, evaluation and demo plans.
- [x] Validate settings, precedence, paths, and upload limits with tests.
- [x] Test import safety and Streamlit startup with missing and invalid settings.
- [x] Sync the locked environment and verify PyTorch, OpenCV and Ultralytics imports.
- [x] Verify pytest, Ruff, and the local Streamlit health endpoint.
- [x] Generate and query the local Graphify graph.

- [x] Verify CPU checkpoint classes and inference on positive and blank smoke clips.
- [x] Stream video and save decoded, annotated output and prediction CSV.
- [x] Validate dataset normalization, label checks, grouping, and YAML export with fixtures.
- [x] Validate one-to-one metrics with fixtures and produce a local CPU video benchmark.

- [x] Test temporal confirmation, observed-absence reset, cooldown and episode consolidation.
- [x] Persist incidents and evidence and test database and snapshot failure cleanup.
- [x] Integrate Streamlit upload, local sample selection, inference, preview and downloads.
- [x] Implement incident review, history filters, analytics and spreadsheet-safe exports.
- [x] Rehearse actual CPU positive and gray-clip smoke runs through Streamlit.

## Partially completed

- [ ] Candidate source-page metadata recorded. Verify final media rights and annotation quality.
- [ ] Metric engine verified with fixtures. Measure held-out accuracy on approved data.

## Blocked

- Held-out accuracy requires an approved labeled split with reviewed source overlap.
- Independent surveillance evaluation requires permitted representative clips.

## Not started

- [ ] Fine-tune only if verified data, compute and schedule permit.
- [ ] Validate webcam operation if the prerecorded workflow is stable.

## Full-screen presentation

- [x] Add automatic filename-sorted CPU inference playlist in a separate Streamlit entry point.
- [x] Verify positive and gray frames, looping, browser refresh, and one display in Chrome.
- [ ] Add approved 10–30-second presentation clips. The existing clips are smoke fixtures.

## ML course experiment

- [x] Extract the Roboflow v4 ZIP safely, label `firearm_present`, run the leakage audit and
  make the grouped StratifiedGroupKFold split (`outputs/ml/manifest.csv`, `audit.json`).
- [x] Cache handcrafted and HOG features and EDA figures under `outputs/ml/`.
- [x] Group video-frame filename families and re-split. No family spans splits now.
- [x] Train the baseline, LR and RF on two feature sets, tune on val and evaluate on test once.
- [ ] Decide on the validation and test negative composition. Each split's negatives
  come almost entirely from one source group. See `outputs/ml/audit.json`.
- [ ] Write the error-analysis explanations (student's own work).
- [ ] Fine-tune YOLOv8n and compare it on the same test images.

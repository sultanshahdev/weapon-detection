# Implementation roadmap

Use the 48-hour window to get the prerecorded-video path working first. Prepare a
vetted local fallback checkpoint before spending time on optional fine-tuning. Gate
the demo on a full rehearsal and measured results.

## Phase 1 — Repository foundation

| Milestone | Deliverables | Dependencies | Acceptance and verification | Status |
| --- | --- | --- | --- | --- |
| Project foundation | Package boundaries, uv lock, AGENTS, CONTEXT, roadmap, tasks, architecture, dataset, evaluation and demo docs | Existing repository, uv and Python 3.11 | Required modules import, locked sync succeeds, project documents describe current versus planned behavior | Complete |
| Startup and configuration | Root Streamlit entry point, settings validation, environment example and upload ceiling | Project foundation | Settings and AppTest cover defaults, overrides, invalid input and missing model. Headless server comes up without checkpoint | Complete |
| Development checks | pytest and Ruff configuration, CPU dependency smoke checks, Graphify graph | Package and startup | Tests and lint pass. PyTorch and Ultralytics import. Graphify writes a graph that can be queried | Complete |

## Phase 2 — Detection engine

| Milestone | Deliverables | Dependencies | Acceptance and verification | Status |
| --- | --- | --- | --- | --- |
| Inspect candidate data | Provenance, license, data selection and conversion manifest | Phase 1 | Inspect rights and labels. Validate coordinates, negatives, duplicates and source grouping in a small reviewed sample | Partial. Publisher metadata reviewed. Dataset rights and archives pending |
| Select model and fallback | Vetted local small YOLO checkpoint and class map, including Open Images V7-pretrained option | Phase 1, checkpoint provenance | Runtime inspection approves firearm labels. Unit-test unsupported and missing model cases. Verify permitted positive and negative samples | Complete. Official OIV7 checkpoint verified with smoke clips |
| Run inference on video | YOLO adapter, frame iterator, confidence filter and annotations | Model selection | Mock inference and test filtering and capture cleanup. Decode and process a short smoke clip on CPU | Complete. MJPG input, MP4 output, and CSV verified |
| Prepare annotations and evaluate | Versioned, source-grouped export, metric report and optional fine-tuning | Candidate data, inference and available compute | Validate split leakage and reproduce results. Measure latency and score the approved model using the documented protocol | Partial. Fixture checks and CPU benchmark complete. Held-out data pending |

## Phase 3 — Application completion

| Milestone | Deliverables | Dependencies | Acceptance and verification | Status |
| --- | --- | --- | --- | --- |
| Create and store incidents | Temporal alert engine, incident manager, SQLite records, PNG or JPEG evidence | Frame and detection contracts | Test confirmation, reset, consolidation, cooldown, write rollback and evidence cleanup | Complete |
| Monitor, review and export | Streamlit upload, annotated frames, alerts, history and CSV | Video processor, incidents and storage | App tests cover status and errors. Reopen stored incidents and match CSV fields to visible records | Complete |
| Verify and demonstrate | End-to-end release checks, expanded documentation, measured results and demo | Working dashboard and local smoke fixtures | Rehearse positive and gray clips, reopen storage, review evidence and export. Run tests and lint | Complete for smoke test. Independent accuracy evaluation remains pending |
| Validate optional webcam | Local camera source | Stable video workflow and usable hardware | Verify permissions, timestamps and safe capture shutdown on demo hardware | Not started |

## Presentation player

| Milestone | Deliverables | Dependencies | Acceptance and verification | Status |
| --- | --- | --- | --- | --- |
| Automatic visual demonstration | Separate full-window Streamlit player, live CPU annotation, filename-ordered playlist | Verified CPU checkpoint and local clips | Verify Chrome startup, positive and blank status, switching, looping, and refresh | Complete with existing smoke fixtures |

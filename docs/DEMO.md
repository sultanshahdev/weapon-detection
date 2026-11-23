# Demonstration

Note: the dashboard and the full-screen player now default to the fine-tuned checkpoint
`models/firearm_yolov8n_best.pt` (fallback `models/yolov8n-oiv7.pt`). The observed results
recorded below were made earlier with `yolov8n-oiv7.pt` and are kept as recorded.

Streamlit supports the upload, analysis, and review walkthrough below. The CLI
variant is also available.

## Dashboard walkthrough

1. Start Streamlit using the isolated demo command below.
2. Select the upload source and upload a permitted prerecorded clip.
3. Select a vetted local detection model and start inference.
4. Show annotated frames with bounding boxes, confidence, timestamp, and alert state.
5. Let qualifying frames generate a reviewable incident.
6. Open the saved annotated evidence and confirm its timestamp.
7. Display the incident in history and verify that it persists after restart.
8. Export CSV and verify the row against the incident and evidence.
9. Present the measured integration results and explain which accuracy questions
   still require independent surveillance evaluation.

## Preparation checklist

- [ ] Verify the checkpoint license, provenance, checksum and firearm class mapping.
- [ ] Obtain permission to use and display the positive and negative clips.
- [ ] Rehearse decoding and storage with the demo hardware and local environment.
- [ ] Record settings, throughput, evaluation results, and representative errors.
- [ ] Rehearse confirmation, evidence review, restart, history, and CSV export offline.
- [ ] Keep a short compatible backup clip and a verified local fallback checkpoint.
- [ ] Verify webcam access only if it will be included in the demonstration.

## Verified CLI smoke test

A one-second 6 FPS MJPG clip was rendered from a public-domain U.S. Army
photograph. It repeats one image and serves as an integration smoke test. Its
training-source overlap is unknown. A separately generated gray clip exercised
zero-result output. [Photograph provenance](https://commons.wikimedia.org/wiki/File:M1911a1.jpg)

```bash
YOLO_OFFLINE=true uv run python scripts/run_video.py --input datasets/samples/m1911a1.avi \
  --model models/yolov8n-oiv7.pt --output outputs/annotated_smoke.mp4 --csv outputs/smoke.csv \
  --device cpu --conf 0.25 --imgsz 640
```

The CPU run decoded and processed six frames, returning one Handgun prediction per
frame. It wrote an MP4 and a six-row CSV in 4.99 seconds, with a median adapter latency of
201 ms and overall throughput of 1.20 processed frames per second. The gray clip
returned zero detections across six frames. These runs verify the inference path. A 6-frame smoke test includes
first-call overhead and cannot establish model accuracy or sustained throughput.
Use fresh output paths for repeat runs.

The soundless export encodes only processed frames using `mp4v` for MP4 or `MJPG` for
AVI. At stride 2 it encodes at half the nominal input FPS and omits intervening frames.
`--max-frames` limits processed frames. `--output-size WIDTH HEIGHT` takes even pixel
dimensions and resizes only the output. CSV pixel coordinates remain in source space.

To rebuild the smoke clip, download the original image from the linked source as
`datasets/samples/m1911a1.jpg` and run from the repository root.

```bash
uv run python - <<'PY'
import cv2
import numpy as np

image = cv2.imread("datasets/samples/m1911a1.jpg")
if image is None:
    raise SystemExit("Save the original photograph before running this command.")
image = cv2.resize(image, (586, 400))
for name, frame in (("m1911a1", image), ("blank", np.full_like(image, 127))):
    writer = cv2.VideoWriter(
        f"datasets/samples/{name}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 6, (586, 400)
    )
    try:
        if not writer.isOpened():
            raise SystemExit("MJPG video writer unavailable.")
        for _ in range(6):
            writer.write(frame)
    finally:
        writer.release()
PY
```

## Run an isolated dashboard demonstration

From the repository root, with the verified model and generated sample clips present:

```bash
uv sync --locked
mkdir -p outputs
demo_dir="$(mktemp -d outputs/dashboard-demo-XXXXXX)"
WD_OUTPUT_DIR="$demo_dir" WD_INPUT_DIR="$demo_dir/uploads" \
WD_DATABASE_PATH=incidents.sqlite3 WD_EVIDENCE_DIR=evidence \
YOLO_OFFLINE=true uv run streamlit run app.py --server.headless=true --server.port=8501
```

Open `http://127.0.0.1:8501`. Upload `datasets/samples/m1911a1.avi` or select it
as a local sample. Select `yolov8n-oiv7.pt`, confidence 0.25, image size 640,
stride 1, confirmation 3, and window 5. Start analysis. Inspect the resulting
evidence and save a review decision and note. Download the incidents CSV. Analyze
`blank.avi` and check that the incident total stays at one. Restart with the same
`demo_dir` to show persistence.

The default rule uses a maximum observation gap of 3 source seconds, 3 misses to
reset, and a 5-second cooldown. The detector streams frames on CPU. Generated
WebM contains only analyzed frames and no audio. Increasing stride omits source
frames and reduces export playback FPS to preserve elapsed source time approximately.
Processing time includes Streamlit progress and output writes. Use the maximum-frame
control to limit work. An abrupt process exit can leave a run with unknown final status.

To reset disposable demo data, start again with a fresh `mktemp` directory. If the
old run is no longer needed, stop Streamlit and remove only that specific generated
directory after verifying its name starts with `outputs/dashboard-demo-`. Production
or research history can use a separate configured directory.

### Observed results

The real stride-1 Streamlit upload test used six source frames, confidence 0.25 and
image size 640. The positive clip recorded six Handgun boxes and one incident at
frame 2 and 0.333 s. Processing took 7.94 seconds in the first run. The subsequent gray
clip processed six frames in 2.02 seconds with zero detections and incidents. SQLite
review persisted across a fresh application session, JPEG bytes matched their recorded
SHA-256, and the incident CSV matched the stored row. Both clips are short integration
fixtures. Their timings include different first-call costs and their results do not
estimate accuracy or sustained throughput.

The final headless Chrome test used stride 2 at the same confidence and image size.
It decoded six frames and analyzed source indices 0, 2, and 4. Three Handgun
predictions produced one incident at frame 4 (0.667 s) in 6.35 seconds. Chrome
played the resulting VP8 WebM and rendered the saved evidence in Incident History.
The gray upload analyzed three frames in 1.19 seconds and added zero incidents.

## Automatic full-screen player

Run this from the repository root with the verified checkpoint under `models/`.

```bash
YOLO_OFFLINE=true uv run streamlit run demo_app.py
```

Open the URL printed by Streamlit and press **F11** in Chrome. The demo starts on
page load. The source name, current-frame detection status and rolling analyzed FPS
appear below a single display. Red boxes show the model's actual class label and
confidence. The player uses real CPU inference on every analyzed observation and
keeps the previous frame visible until the next analyzed frame arrives.

Copy permitted 10–30-second clips into `datasets/demo_playlist/`. Name them with
numeric prefixes such as `01_ordinary_scene.mp4` and `02_permitted_example.avi`
to control their lexicographic order. Use ordinary scenes and permitted firearm
examples. Supported extensions are MP4, AVI, MOV, MKV and WebM. Actual playback
depends on OpenCV decoding. Files are discovered on page load. A clip that
fails is logged on the server and skipped for the rest of that session. If no clip
works, the player displays an error. The directory is tracked with a marker and
the videos are ignored by Git.

The existing local gray and repeated-photo smoke clips were copied into the playlist
for verification. They are short integration fixtures. Replace them with your approved
presentation clips when ready. For a fresh checkout you can build the fixtures using
the script above, then copy them into the playlist.

| Environment variable | Default |
| --- | --- |
| `WD_DEMO_PLAYLIST` | Repository `datasets/demo_playlist/`. Changing this is optional. |
| `WD_DEMO_IMGSZ` | `416`. Use a multiple of 32. |
| `WD_DEMO_CONFIDENCE` | `0.35`. Allowed range 0–1. |
| `WD_DEMO_STRIDE` | `1`. Higher values skip source frames. |

The player emits analyzed images as they are ready. It is silent and may take
longer than the source clip duration. The FPS indicator measures recent server-side
frame delivery, including inference and frame encoding. Headless Chrome verified
automatic gray → positive → gray switching, both status messages, red box pixels,
one viewport, and refresh. It displayed 2.1–3.6 FPS on these short clips across two
Chrome runs at image size 416. In a separate decode-and-inference comparison, the second pass over the
positive clip ran at 1.89 FPS for 320 and 1.99 FPS for 416. These short results do
not establish sustained performance on longer footage. The presentation player
does not persist runs, evidence, or incident records.

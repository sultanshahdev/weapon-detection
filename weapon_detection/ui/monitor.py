"""Upload or select a video and present results from the analysis coordinator."""

import logging
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import cv2
import pandas as pd
import streamlit as st

from weapon_detection.detection.model import YoloDetector, default_checkpoint
from weapon_detection.incidents.alert_engine import AlertRules
from weapon_detection.incidents.incident_manager import AnalysisResult, analyze_video
from weapon_detection.storage.evidence import EvidenceStore
from weapon_detection.storage.repository import IncidentRepository
from weapon_detection.utils.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


class LockedDetector:
    """Serialize calls into a shared Ultralytics predictor."""

    def __init__(self, detector: YoloDetector) -> None:
        self.detector = detector
        self.lock = Lock()

    def predict(self, frame):
        with self.lock:
            return self.detector.predict(frame)


@st.cache_resource(max_entries=3, show_spinner=False)
def cached_detector(
    path: Path, revision: tuple[int, int, int], confidence: float, image_size: int
) -> LockedDetector:
    return LockedDetector(YoloDetector(path, confidence=confidence, imgsz=image_size))


def approved_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*")
        if path.suffix.lower() in extensions and not path.is_symlink() and path.is_file()
    )


@contextmanager
def local_input(upload, sample: Path | None, settings: Settings):
    if upload is None:
        if sample is None:
            raise ValueError("Upload a video or choose a local sample.")
        yield sample, sample.name, "sample"
        return
    size = upload.size
    if not 0 < size <= settings.max_upload_mb * 1024**2:
        raise ValueError(f"Upload must be nonempty and at most {settings.max_upload_mb} MiB.")
    name = Path(upload.name.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError("Choose an MP4, AVI, MOV, MKV or WebM file.")
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="video-", dir=settings.input_dir) as directory:
        path = Path(directory) / f"input{suffix}"
        upload.seek(0)
        with path.open("xb") as stream:
            shutil.copyfileobj(upload, stream)
        yield path, name, "upload"


def render_monitor(settings: Settings, repository: IncidentRepository, evidence: EvidenceStore):
    models = approved_files(PROJECT_ROOT / "models", (".pt",))
    if settings.model_path is None:
        st.caption("No local model configured through WD_MODEL_PATH. Choose a local checkpoint.")
    elif settings.model_path.is_file() and settings.model_path.suffix.lower() == ".pt":
        models = [settings.model_path, *(p for p in models if p != settings.model_path)]
    else:
        st.warning("The configured model file is unavailable. Check WD_MODEL_PATH.")
    # WD_MODEL_PATH wins; otherwise the fine-tuned model, then the OIV7 fallback.
    preferred = (
        settings.model_path if settings.model_path in models else default_checkpoint(PROJECT_ROOT)
    )
    if not models:
        st.warning(
            "Place a trusted firearm-capable .pt checkpoint in models/ or set WD_MODEL_PATH."
        )
    mode = st.radio("Video source", ["Upload", "Local sample"], horizontal=True)
    upload = None
    sample = None
    if mode == "Upload":
        upload = st.file_uploader(
            "Upload a prerecorded video",
            type=[value[1:] for value in VIDEO_SUFFIXES],
            help=f"Local processing. Maximum upload size {settings.max_upload_mb} MiB.",
            max_upload_size=settings.max_upload_mb,
        )
    else:
        samples = approved_files(PROJECT_ROOT / "datasets" / "samples", VIDEO_SUFFIXES)
        sample = st.selectbox("Approved sample", samples, format_func=lambda path: path.name)
        if not samples:
            st.caption("Add an approved test clip under datasets/samples/.")

    if upload is not None:
        filename = Path(upload.name.replace("\\", "/")).name
        st.caption(f"Ready to analyze {filename}")

    with st.form("analysis"):
        model = st.selectbox(
            "Checkpoint",
            models,
            index=models.index(preferred) if preferred in models else 0,
            format_func=lambda path: path.name,
        )
        left, middle, right = st.columns(3)
        confidence = left.number_input(
            "Confidence threshold", 0.0, 1.0, settings.confidence_threshold, step=0.05
        )
        image_size = middle.selectbox("Inference image size", (416, 640), index=1)
        stride = right.number_input("Frame stride", min_value=1, value=2, step=1)
        max_frames = left.number_input(
            "Maximum processed frames", min_value=1, value=settings.max_video_frames
        )
        confirmation = middle.number_input(
            "Positive observations", min_value=1, value=settings.alert_confirmation_frames
        )
        window = right.number_input("Confirmation window", min_value=1, value=settings.alert_window)
        reset = left.number_input(
            "Misses to reset", min_value=1, value=settings.alert_reset_misses, step=1
        )
        gap = middle.number_input(
            "Maximum gap (s)", min_value=0.01, value=settings.alert_max_gap_seconds
        )
        cooldown = right.number_input(
            "Cooldown (s)", min_value=0.0, value=settings.alert_cooldown_seconds
        )
        st.caption("CPU · The alert window counts processed observations. Timing uses video time.")
        started = st.form_submit_button(
            "Start Analysis",
            type="primary",
            disabled=not models or (upload is None and sample is None),
        )
    if started:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_classes", None)
        try:
            rules = AlertRules(confidence, confirmation, window, gap, reset, cooldown)
            with local_input(upload, sample, settings) as (path, source, source_type):
                revision = model.stat()
                with st.spinner("Loading checkpoint on CPU…"):
                    detector = cached_detector(
                        model,
                        (revision.st_mtime_ns, revision.st_size, revision.st_ino),
                        confidence,
                        image_size,
                    )
                classes = ", ".join(detector.detector.firearm_classes.values())
                st.session_state.analysis_classes = (
                    f"{model.name} · CPU · {classes} · confidence ≥ {confidence:.2f}"
                )
                st.session_state.analysis_result = analyze_video(
                    path,
                    detector,
                    repository,
                    evidence,
                    settings.output_dir,
                    source_name=source,
                    source_type=source_type,
                    model=model.name,
                    image_size=image_size,
                    rules=rules,
                    stride=stride,
                    max_frames=max_frames,
                    progress=progress_display(),
                )
        except (OSError, ValueError, RuntimeError, sqlite3.Error, cv2.error) as exc:
            logger.exception("Analysis failed")
            if isinstance(exc, (ValueError, RuntimeError)):
                st.error(f"Analysis failed. {exc}")
            else:
                st.error(
                    "Analysis failed while reading, encoding or saving data. Check server logs."
                )
    if classes := st.session_state.get("analysis_classes"):
        st.caption(classes)
    if result := st.session_state.get("analysis_result"):
        render_result(result)


def progress_display():
    status = st.empty()
    bar = st.empty()
    preview = st.empty()
    running_latency = 0.0

    def show(frame, stats, incidents, incident, elapsed):
        nonlocal running_latency
        running_latency += frame.inference_latency_ms
        mean = running_latency / stats.processed_frames
        total = stats.reported_total_frames
        source = f"{stats.decoded_frames}/{total}" if total else str(stats.decoded_frames)
        status.text(
            f"Source frames decoded {source} · {stats.processed_frames} analyzed · "
            f"{stats.detection_count} detections · {incidents} incidents · "
            f"{elapsed:.1f} s elapsed · {mean:.0f} ms mean inference"
        )
        if total:
            bar.progress(min(1.0, stats.decoded_frames / total))
        if stats.processed_frames == 1 or stats.processed_frames % 5 == 0 or incident:
            image = frame.annotated_frame
            if image.shape[1] > 720:
                image = cv2.resize(
                    image, (720, max(1, round(image.shape[0] * 720 / image.shape[1])))
                )
            preview.image(image, channels="BGR", caption=f"Source frame {frame.frame_index}")
        if incident:
            st.warning(
                f"Multi-frame rule confirmed incident {incident['id']} for "
                f"{incident['source_name']} at {incident['timestamp_seconds']:.2f} s · "
                f"{incident['confirmation_count']} positive observations · "
                f"confidence {incident['confidence']:.2f} · evidence saved · new"
            )

    return show


def render_result(result: AnalysisResult) -> None:
    stats = result.stats
    st.subheader("Analysis results")
    st.caption(f"{result.source_name} · run {result.run_id}")
    columns = st.columns(4)
    columns[0].metric("Frames analyzed", stats["processed_frames"])
    columns[1].metric("Frame detections", stats["detection_count"])
    columns[2].metric("Confirmed incidents", result.incident_count)
    columns[3].metric("Processing time", f"{stats['elapsed_seconds']:.2f} s")
    total = stats["reported_total_frames"]
    st.caption(
        f"Decoded {stats['decoded_frames']} of {total if total is not None else 'unknown'} "
        f"source frames · {stats['source_fps']:.2f} source FPS · "
        f"{stats['inference_mean_ms']:.1f} ms mean inference · "
        f"{stats['processed_frames_per_second']:.2f} analyzed frames/s on this run."
    )
    if stats["stopped_at_limit"]:
        st.caption("Stopped at the selected processed-frame limit.")
    if not result.incident_count:
        st.info(
            "No detections met the multi-frame rule."
            if stats["detection_count"]
            else "The model returned no qualifying firearm detections."
        )
    if not result.video.is_file() or not result.detections_csv.is_file():
        st.warning("Local analysis files are unavailable. Incident history remains accessible.")
        return
    if st.checkbox("Open annotated video player", key=f"player-{result.run_id}"):
        st.video(str(result.video), format="video/webm")
    st.caption("The WebM contains annotated frames without audio.")
    st.download_button(
        "Download annotated video",
        result.video.read_bytes,
        "annotated.webm",
        "video/webm",
        on_click="ignore",
    )
    st.download_button(
        "Download detection CSV",
        result.detections_csv.read_bytes,
        "detections.csv",
        "text/csv",
        on_click="ignore",
    )
    table = pd.read_csv(result.detections_csv, nrows=250)
    if not table.empty:
        st.caption("First 250 frame detections. The CSV contains all detections.")
        st.dataframe(
            table[
                [
                    "frame_index",
                    "timestamp_seconds",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
            ],
            hide_index=True,
        )

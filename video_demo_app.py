"""Video demo: Mendeley firearm-action footage with dataset (green) and model (red) boxes.

Run with:  uv run streamlit run video_demo_app.py
CPU only. Model boxes come from outputs/demo_cache/ (scripts/precompute_demo.py); the
confidence slider only filters those cached boxes. Rendered videos are cached in
outputs/demo_videos/. The Mendeley videos are a demo source only; they are not used for
training, threshold selection or any metric.
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from weapon_detection.data.mendeley_videos import (
    DATASET_ROOT,
    list_videos,
    load_boxes,
    read_frame,
    video_info,
)
from weapon_detection.detection.annotated_video import (
    DEFAULT_CONFIDENCE,
    cache_name,
    draw_overlays,
    render_annotated_video,
)
from weapon_detection.detection.demo_cache import (
    CACHE_DIR,
    DETECTION_FLOOR,
    IMGSZ,
    SUMMARY_HEADER,
    cache_path,
    detect_video,
    filter_boxes,
    read_cache,
    save_summary,
    signature,
    summary_rows,
    threshold_facts,
    write_cache,
)
from weapon_detection.detection.model import YoloDetector
from weapon_detection.ml.common import SEED
from weapon_detection.utils.config import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "outputs" / "demo_videos"
SELECTION = CACHE_DIR / "demo_selection.json"
DEMO_SET = "Demo selection"
# First existing checkpoint wins: my fine-tuned model, then the pretrained OIV7 fallback.
CHECKPOINTS = [
    PROJECT_ROOT / "models/firearm_yolov8n_best.pt",
    PROJECT_ROOT / "models/yolov8n-oiv7.pt",
]

CAPTION = (
    "Demo footage from the Mendeley dataset (Ruiz-Santaquiteria et al., DOI "
    "10.17632/bbzpxhd22j.2). Green boxes are the dataset's own annotations. Red boxes are "
    "predictions of my model, which was trained on a different dataset. This footage was "
    "not used for training or for any reported result."
)
# Copied from https://data.mendeley.com/datasets/bbzpxhd22j/2 (Licence and Description).
LICENCE = (
    "Licence: CC BY NC 3.0. \"The dataset is licensed under CC BY NC 3.0. However, we do "
    "allow commercial use, but only if prior permission is obtained.\" Used here for "
    "non-commercial coursework demonstration only."
)
FACT_COLUMNS = (
    "frames_with_model_box: frames with at least one model box at this threshold. "
    "model_boxes: model boxes at this threshold. gt_frames: frames with a dataset box. "
    "gt_frames_matched_iou50: GT frames where a model box overlaps a GT box with IoU >= 0.5."
)


@st.cache_data
def cached_videos(root: str) -> list:
    return list_videos(Path(root))


@st.cache_data
def load_cache(path: str, mtime_ns: int, expected: str) -> dict | None:
    """Read one detection cache; mtime_ns is only the cache key, so edits are picked up."""
    return read_cache(Path(path), json.loads(expected))


def cache_for(record, expected: dict | None) -> dict | None:
    if expected is None:
        return None
    path = cache_path(record.class_name, record.name)
    if not path.exists():
        return None
    return load_cache(str(path), path.stat().st_mtime_ns, json.dumps(expected))


@st.cache_resource
def load_detector(path: str) -> YoloDetector:
    """Load the checkpoint on CPU through the repo's adapter (checks its class names)."""
    os.environ.setdefault("YOLO_OFFLINE", "true")  # never download anything
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return YoloDetector(Path(path), confidence=DETECTION_FLOOR, imgsz=IMGSZ)


def selection_keys() -> list[str]:
    if not SELECTION.exists():
        return []
    return [f"{v['class_name']}/{v['name']}" for v in json.loads(SELECTION.read_text())["videos"]]


def facts_table(record, info, gt, model) -> pd.DataFrame:
    """Plain facts about the selected video; no interpretation."""
    categories = sorted({box[4] for boxes in gt.values() for box in boxes})
    rows = {
        "class folder": record.class_name,
        "resolution": f"{info['width']}x{info['height']}",
        "fps": f"{info['fps']:.2f}",
        "frames": info["frames"],
        "duration (s)": f"{info['duration']:.2f}",
        "frames with GT boxes": len(gt),
        "total GT boxes": sum(len(b) for b in gt.values()),
        "GT categories": ", ".join(categories) or "none",
    }
    if model is not None:
        rows["frames with a model box"] = len(model)
        rows["total model boxes"] = sum(len(v) for v in model.values())
        rows["first model detection (frame)"] = min(model) if model else "none"
    return pd.DataFrame({"value": [str(v) for v in rows.values()]}, index=list(rows))


def step(keys: list[str], delta: int) -> None:
    """Previous/Next button callback: move the remembered selection by delta."""
    current = st.session_state.get("video")
    index = keys.index(current) if current in keys else 0
    st.session_state.video = keys[(index + delta) % len(keys)]


def main() -> None:
    st.set_page_config(page_title="Firearm video demo", layout="wide")
    st.title("Firearm video demo - dataset annotations vs model detections")
    st.caption(CAPTION)
    st.caption(LICENCE)

    records = cached_videos(str(DATASET_ROOT)) if DATASET_ROOT.is_dir() else []
    if not records:
        st.error(
            f"No videos found. Expected the extracted dataset in:\n\n`{DATASET_ROOT}`\n\n"
            "Run `uv run python scripts/extract_mendeley_videos.py` (see README, Video demo)."
        )
        return
    checkpoint = next((p for p in CHECKPOINTS if p.is_file()), None)
    expected = signature(checkpoint) if checkpoint else None
    by_key = {f"{r.class_name}/{r.name}": r for r in records}
    caches = {key: cache_for(r, expected) for key, r in by_key.items()}

    with st.sidebar:
        demo_keys = [k for k in selection_keys() if k in by_key]
        classes = sorted({r.class_name for r in records})
        filters = ([DEMO_SET] if demo_keys else []) + ["All", *classes]
        chosen = st.selectbox("Class", filters, key="class_filter")
        if chosen == DEMO_SET:
            keys = demo_keys
        else:
            keys = [k for k, r in by_key.items() if chosen in ("All", r.class_name)]
        # Remember the video across reruns; only reset it if it is not in this list.
        if st.session_state.get("video") not in keys:
            st.session_state.video = keys[0]
        left, right = st.columns(2)
        left.button("Previous", on_click=step, args=(keys, -1), width="stretch")
        right.button("Next", on_click=step, args=(keys, 1), width="stretch")
        key = st.selectbox(
            "Video",
            keys,
            key="video",
            format_func=lambda k: f"{k} [{'model ready' if caches[k] else 'annotations only'}]",
        )
        ready = sum(c is not None for c in caches.values())
        st.caption(f"Model detections precomputed: {ready} of {len(caches)} videos")
        cache = caches[key]
        show_gt = st.checkbox("Show dataset annotations (green)", value=True)
        show_model = st.checkbox(
            "Show model detections (red)", value=False, disabled=cache is None
        )
        conf = st.slider("Model confidence threshold", 0.05, 0.95, DEFAULT_CONFIDENCE, 0.05)
        run_now = False
        if checkpoint is None:
            st.caption("No model checkpoint found in models/; only dataset annotations are shown.")
        elif cache is None:
            st.caption("No precomputed detections for this video; model overlay is off.")
            run_now = st.button("Run model on this video now")
        else:
            st.caption(f"Model checkpoint: {checkpoint.name}")

    record = by_key[key]
    gt = load_boxes(record.label_path)
    if run_now:
        bar = st.progress(0.0, text="Running the model on CPU (slow)...")
        result = detect_video(
            load_detector(str(checkpoint)),
            record.video_path,
            progress=lambda f: bar.progress(f, text=f"Running the model on CPU... {f:.0%}"),
        )
        write_cache(cache_path(record.class_name, record.name), {**expected, **result})
        st.rerun()

    use_model = show_model and cache is not None
    model = filter_boxes(cache["boxes"], conf) if use_model else None
    out_path = OUT_DIR / cache_name(
        f"{record.class_name}_{record.name}", show_gt, checkpoint.name if use_model else None, conf
    )
    if not out_path.exists():
        with st.spinner("Rendering from cached boxes..."):
            render_annotated_video(
                record.video_path,
                gt if show_gt else None,
                None,
                out_path,
                conf,
                cached_boxes=cache["boxes"] if use_model else None,
            )
    st.video(str(out_path))

    info = video_info(record.video_path)
    st.subheader(key)
    st.table(facts_table(record, info, gt, model))

    with st.expander("Frame browser"):
        index = st.slider("Frame number (0-based)", 0, max(0, info["frames"] - 1), 0)
        frame = read_frame(record.video_path, index)
        if frame is None:
            st.error("This frame could not be read.")
        else:
            image = draw_overlays(
                frame,
                index,
                gt.get(index, []) if show_gt else None,
                model.get(index, []) if model is not None else None,
            )
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"frame {index}")

    with st.expander("Threshold facts for this video"):
        if cache is None:
            st.write("No precomputed detections for this video.")
        else:
            st.table(pd.DataFrame(threshold_facts(gt, cache["boxes"], cache["n_frames"])))
            st.caption(FACT_COLUMNS)

    with st.expander("All precomputed videos (demo footage statistics, not an evaluation)"):
        entries = [
            (by_key[k].class_name, load_boxes(by_key[k].label_path), c)
            for k, c in caches.items()
            if c is not None
        ]
        st.markdown(SUMMARY_HEADER)
        st.write(f"Videos included: {len(entries)}")
        if entries:
            rows = summary_rows(entries)
            save_summary(rows)  # outputs/demo_cache/threshold_summary.csv and .md
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(
                "frac_frames_with_model_box: mean over videos of frames with a model box / "
                "frames. frac_gt_frames_matched_iou50: mean over videos of GT frames with a "
                "model box at IoU >= 0.5 / GT frames (Handgun and Machine_Gun only)."
            )

    counts = {"ground-truth boxes": [len(gt.get(i, [])) for i in range(info["frames"])]}
    if model is not None:
        counts["model boxes"] = [len(model.get(i, [])) for i in range(info["frames"])]
    st.markdown("**Boxes per frame**")
    st.line_chart(pd.DataFrame(counts), x_label="frame number", y_label="boxes in frame")


main()

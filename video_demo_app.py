"""Video demo: Mendeley firearm-action footage with dataset (green) and model (red) boxes.

Run with:  uv run streamlit run video_demo_app.py
CPU only. Rendered videos are cached in outputs/demo_videos/ (git-ignored).
The Mendeley videos are a demo source only; they are not used for training or any metric.
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
from weapon_detection.detection.model import YoloDetector
from weapon_detection.ml.common import SEED
from weapon_detection.utils.config import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "outputs" / "demo_videos"
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


@st.cache_data
def cached_videos(root: str) -> list:
    return list_videos(Path(root))


@st.cache_resource
def load_detector(path: str, conf: float) -> YoloDetector:
    """Load the checkpoint on CPU through the repo's adapter (checks its class names)."""
    os.environ.setdefault("YOLO_OFFLINE", "true")  # never download anything
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return YoloDetector(Path(path), confidence=conf)


def facts_table(record, info, gt, sidecar) -> pd.DataFrame:
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
    if sidecar and sidecar["model_on"]:
        model = {int(k): v for k, v in sidecar["model_boxes"].items()}
        rows["frames with a model box"] = len(model)
        rows["total model boxes"] = sum(len(v) for v in model.values())
        rows["first model detection (frame)"] = min(model) if model else "none"
        rows["model confidence threshold"] = sidecar["conf"]
    return pd.DataFrame({"value": [str(v) for v in rows.values()]}, index=list(rows))


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

    with st.sidebar:
        classes = sorted({r.class_name for r in records})
        chosen_class = st.selectbox("Class", ["All", *classes])
        shown = [r for r in records if chosen_class in ("All", r.class_name)]
        record = st.selectbox(
            "Video",
            shown,
            format_func=lambda r: f"{r.class_name} / {r.name}" if chosen_class == "All" else r.name,
        )
        show_gt = st.checkbox("Show dataset annotations (green)", value=True)
        show_model = st.checkbox(
            "Show model detections (red)",
            value=False,
            disabled=checkpoint is None,
            help="Disabled: no checkpoint in models/." if checkpoint is None else None,
        )
        if checkpoint is None:
            st.caption("No model checkpoint found in models/; only dataset annotations are shown.")
        else:
            st.caption(f"Model checkpoint: {checkpoint.name}")
        conf = st.slider("Model confidence threshold", 0.05, 0.95, DEFAULT_CONFIDENCE, 0.05)
        prepare = st.button("Prepare video", type="primary")

    use_model = show_model and checkpoint is not None
    video_id = f"{record.class_name}_{record.name}"
    out_path = OUT_DIR / cache_name(video_id, show_gt, checkpoint.name if use_model else None, conf)
    sidecar_path = out_path.with_suffix(".json")
    gt = load_boxes(record.label_path)

    if prepare and not (out_path.exists() and sidecar_path.exists()):
        bar = st.progress(0.0, text="Rendering on CPU...")
        detector = load_detector(str(checkpoint), conf) if use_model else None
        render_annotated_video(
            record.video_path,
            gt if show_gt else None,
            detector,
            out_path,
            conf,
            progress=lambda f: bar.progress(f, text=f"Rendering on CPU... {f:.0%}"),
        )
        bar.empty()

    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None
    if out_path.exists():
        st.video(str(out_path))
    else:
        st.info("Press **Prepare video** in the sidebar to render this video with these settings.")

    info = video_info(record.video_path)
    st.subheader(f"{record.class_name} / {record.name}")
    st.table(facts_table(record, info, gt, sidecar))

    model = {}
    if sidecar and sidecar["model_on"]:
        model = {int(k): v for k, v in sidecar["model_boxes"].items()}

    with st.expander("Frame browser"):
        last = max(0, info["frames"] - 1)
        index = st.slider("Frame number (0-based)", 0, last, 0)
        frame = read_frame(record.video_path, index)
        if frame is None:
            st.error("This frame could not be read.")
        else:
            image = draw_overlays(
                frame,
                index,
                gt.get(index, []) if show_gt else None,
                model.get(index, []) if sidecar and sidecar["model_on"] else None,
            )
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"frame {index}")

    counts = {"ground-truth boxes": [len(gt.get(i, [])) for i in range(info["frames"])]}
    if sidecar and sidecar["model_on"]:
        counts["model boxes"] = [len(model.get(i, [])) for i in range(info["frames"])]
    st.markdown("**Boxes per frame**")
    st.line_chart(pd.DataFrame(counts), x_label="frame number", y_label="boxes in frame")


main()

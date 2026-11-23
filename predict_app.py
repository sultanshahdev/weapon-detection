"""Audience-input demo: pick or upload an image, get a live firearm prediction.

Run with:  uv run streamlit run predict_app.py
CPU only. Nothing is written to disk.
"""

import json
import os

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from weapon_detection.ml.common import MANIFEST_PATH, OUTPUT_DIR, REPO_ROOT, SEED
from weapon_detection.ml.predict import (
    CLASSICAL_DIR,
    CLASSICAL_MODELS,
    YOLO_MODELS,
    classical_predict,
    decode_image,
    label_text,
    model_menu,
    yolo_predict,
)

CAPTION = (
    "Research prototype for a university project; not a safety system; "
    "predictions require human review."
)
SETTINGS_PATH = OUTPUT_DIR / "chosen_settings.json"


@st.cache_resource
def load_classical(key: str):
    return joblib.load(CLASSICAL_DIR / f"{key}.joblib")


@st.cache_resource
def load_yolo(path: str):
    os.environ.setdefault("YOLO_OFFLINE", "true")  # never download anything
    from ultralytics import YOLO

    return YOLO(path)


@st.cache_data
def load_settings() -> dict:
    return json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}


@st.cache_data
def test_images() -> pd.DataFrame:
    """TEST-split images only, in a fixed random order (seed 42)."""
    manifest = pd.read_csv(MANIFEST_PATH)
    test = manifest[manifest["split"] == "test"][["image_path", "firearm_present"]]
    return test.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Firearm image check", layout="wide")
    st.title("Firearm image check - live prediction")
    st.caption(CAPTION)

    settings = load_settings()
    names, notes = model_menu(settings)
    if not names:
        st.error("No models are available. Run the ML pipeline first (see README).")
        return

    left, right = st.columns(2)
    with left:
        source = st.radio("Input", ["Test-split image", "Upload an image"], horizontal=True)
        image_bgr, true_label, image_name = None, None, None
        if source == "Test-split image":
            if not MANIFEST_PATH.exists():
                st.error("manifest.csv not found. Run prepare_data first.")
                return
            test = test_images()
            if "pick" not in st.session_state:
                st.session_state.pick = 0
            if st.button("Show random test image"):
                rng = np.random.default_rng(SEED + st.session_state.pick + 1)
                st.session_state.pick = int(rng.integers(len(test)))
            labels = [p.split("/")[-1] for p in test["image_path"]]
            index = st.selectbox(
                "Test image",
                range(len(test)),
                index=st.session_state.pick,
                format_func=lambda i: labels[i],
            )
            row = test.iloc[index]
            image_bgr = cv2.imread(str(REPO_ROOT / row["image_path"]))
            true_label, image_name = int(row["firearm_present"]), labels[index]
            if image_bgr is None:
                st.error("This image could not be read.")
                return
        else:
            upload = st.file_uploader("Upload a jpg or png", type=["jpg", "jpeg", "png"])
            if upload is None:
                st.info("Upload an image to get a prediction.")
                return
            image_bgr = decode_image(upload.getvalue())
            image_name = upload.name
            if image_bgr is None:
                st.error("This file is not a readable image. Please try another jpg or png.")
                return

        model_name = st.selectbox("Model", names)
        for note in notes:
            st.caption(note)

    try:
        if model_name in CLASSICAL_MODELS:
            key, feature_set = CLASSICAL_MODELS[model_name]
            threshold = settings[key]["threshold"]
            result = classical_predict(load_classical(key), image_bgr, feature_set, threshold)
        else:
            key, path = YOLO_MODELS[model_name]
            threshold = settings[key]["threshold"]
            imgsz = int(settings[key].get("imgsz", 640))
            result = yolo_predict(load_yolo(str(path)), image_bgr, threshold, imgsz)
    except FileNotFoundError as error:
        st.error(f"Model file missing: {error}")
        return

    with right:
        prediction = label_text(result["is_firearm"])
        if result["is_firearm"]:
            st.warning(f"Prediction: **{prediction}**")
        else:
            st.info(f"Prediction: **{prediction}**")
        threshold_text = "none (always the majority class)" if threshold is None else threshold
        st.write(
            f"Score: {result['score']:.4f} | threshold: {threshold_text} | "
            f"latency: {result['ms']:.1f} ms (CPU)"
        )
        if true_label is not None:
            correct = int(result["is_firearm"]) == true_label
            truth = "firearm" if true_label else "no firearm"
            marker = "CORRECT" if correct else "INCORRECT"
            (st.success if correct else st.error)(f"{marker} - true label: {truth}")
        shown = result["image_rgb"]
        if shown is None:
            shown = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(shown, caption=image_name, width="stretch")
        st.caption(CAPTION)


main()

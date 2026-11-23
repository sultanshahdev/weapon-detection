"""Streamlit entry point for the local academic prototype."""

import logging
import os
import sqlite3
from importlib.metadata import version

import streamlit as st

from weapon_detection.storage.evidence import EvidenceStore
from weapon_detection.storage.repository import IncidentRepository
from weapon_detection.ui.monitor import render_monitor
from weapon_detection.ui.review import render_analytics, render_history
from weapon_detection.utils.config import ConfigurationError, load_settings

PROJECT_TITLE = "Real-Time Weapon Detection and AI-Powered Threat Intelligence System"
logger = logging.getLogger(__name__)


def main() -> None:
    st.set_page_config(
        page_title="Real-Time Weapon Detection & AI-Powered Threat Intelligence", layout="wide"
    )
    st.title(PROJECT_TITLE)
    st.info(
        "Research prototype. The local foundation is running. A prediction alone cannot establish "
        "that a firearm is present. Human review is required."
    )
    os.environ.setdefault("YOLO_OFFLINE", "true")
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        st.error(f"Configuration error. {exc}")
        return
    try:
        repository = IncidentRepository(settings.database_path)
        evidence = EvidenceStore(settings.evidence_dir)
    except (OSError, sqlite3.Error, ValueError):
        logger.exception("Could not initialize incident storage")
        st.error("Incident storage could not open. Check the configured database and its schema.")
        return

    monitoring, history, analytics, system = st.tabs(
        ["Video Monitoring", "Incident History", "Analytics", "System Information"]
    )
    try:
        with monitoring:
            render_monitor(settings, repository, evidence)
        with history:
            render_history(repository, evidence)
        with analytics:
            render_analytics(repository)
    except (OSError, sqlite3.Error):
        logger.exception("Could not read dashboard data")
        st.error("A local file or database could not be read. Check the server logs.")
    with system:
        st.subheader("System information")
        st.write(
            f"CPU inference · PyTorch {version('torch')} · Ultralytics {version('ultralytics')}"
        )
        st.caption("Checkpoint labels and settings are shown in Video Monitoring after loading.")
        st.write(
            "Temporal confirmation reduces isolated detections. The rule describes a source-level "
            "episode without object tracking. Accuracy on representative surveillance "
            "footage has yet to be measured."
        )
        st.caption(
            "Analysis runs synchronously. Use the processed-frame limit to bound work. "
            "Uploads are temporary. Run outputs and evidence stay on local disk."
        )


if __name__ == "__main__":
    main()

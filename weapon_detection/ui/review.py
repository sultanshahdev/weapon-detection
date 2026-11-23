"""Incident history, evidence review, and repository-backed analytics."""

import hashlib
import logging
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from weapon_detection.storage.evidence import EvidenceStore
from weapon_detection.storage.reports import incident_csv
from weapon_detection.storage.repository import REVIEW_STATUSES, IncidentRepository

logger = logging.getLogger(__name__)
DISPLAY_FIELDS = {
    "id": "Incident",
    "created_at_utc": "Created (UTC)",
    "source_name": "Video",
    "timestamp_seconds": "Video time (s)",
    "normalized_class": "Class",
    "confidence": "Confidence",
    "confirmation_count": "Positive observations",
    "review_status": "Review status",
}


def render_history(repository: IncidentRepository, evidence: EvidenceStore) -> None:
    st.subheader("Incident history")
    if message := st.session_state.pop("review_feedback", None):
        st.success(message)
    all_incidents = repository.list_incidents()
    left, middle, right = st.columns(3)
    status = left.selectbox(
        "Review status filter",
        (None, *REVIEW_STATUSES),
        format_func=lambda value: value or "All statuses",
    )
    source = middle.selectbox(
        "Video source filter",
        [None, *sorted({r["source_name"] for r in all_incidents})],
        format_func=lambda value: value or "All videos",
    )
    confidence = right.slider("Minimum confidence", 0.0, 1.0, 0.0, step=0.05)
    visible = repository.list_incidents(status=status, source=source, confidence=confidence)
    st.download_button(
        "Export filtered incidents",
        incident_csv(visible),
        "incidents_filtered.csv",
        "text/csv",
        on_click="ignore",
    )
    st.download_button(
        "Export all incidents",
        incident_csv(all_incidents),
        "incidents.csv",
        "text/csv",
        on_click="ignore",
    )
    if not visible:
        st.info("No incidents match the current filters.")
        return
    table = pd.DataFrame(visible)[list(DISPLAY_FIELDS)].rename(columns=DISPLAY_FIELDS)
    st.dataframe(table, hide_index=True)
    incidents = {row["id"]: row for row in visible}
    selected = st.selectbox(
        "Inspect incident",
        list(incidents),
        format_func=lambda key: (
            f"{incidents[key]['source_name']} · "
            f"{incidents[key]['timestamp_seconds']:.2f} s · {key[-12:]}"
        ),
    )
    render_incident(incidents[selected], repository, evidence)


def render_incident(
    incident: dict,
    repository: IncidentRepository,
    evidence: EvidenceStore,
) -> None:
    left, right = st.columns([3, 2])
    try:
        image = evidence.read(incident["snapshot_path"])
    except (ValueError, OSError):
        logger.exception("Unable to read incident evidence")
        image = None
    if image is None:
        left.warning("Evidence snapshot is missing or unreadable.")
    elif hashlib.sha256(image).hexdigest() != incident["snapshot_sha256"]:
        left.warning("Evidence bytes differ from the saved SHA-256.")
    else:
        left.image(image, caption="Annotated frame at temporal confirmation")
    box = [format(incident[f"bbox_{name}"], ".1f") for name in ("x1", "y1", "x2", "y2")]
    right.write(
        f"Model label: {incident['class_name']} · confidence {incident['confidence']:.2f}\n\n"
        f"Source frame {incident['source_frame_index']} · "
        f"{incident['timestamp_seconds']:.2f} s · {incident['detection_count']} qualifying boxes"
    )
    right.caption(f"Representative XYXY pixels ({', '.join(box)})")
    right.caption(f"Run {incident['run_id']}")
    with right.expander("Snapshot SHA-256"):
        st.code(incident["snapshot_sha256"])
    with right.form(f"review-{incident['id']}"):
        review_status = st.selectbox(
            "Review decision",
            REVIEW_STATUSES,
            index=REVIEW_STATUSES.index(incident["review_status"]),
        )
        notes = st.text_area("Review notes", value=incident["review_notes"], max_chars=4000)
        save = st.form_submit_button("Save review", type="primary")
    if save:
        try:
            repository.review(incident["id"], review_status, notes)
        except (ValueError, sqlite3.Error):
            logger.exception("Incident review failed")
            right.error("The review could not be saved. Check the database and try again.")
        else:
            st.session_state.review_feedback = "Review saved."
            st.rerun()


def render_analytics(repository: IncidentRepository) -> None:
    st.subheader("Analytics")
    runs = repository.list_runs()
    incidents = repository.list_incidents()
    completed = [run for run in runs if run["status"] == "completed"]
    columns = st.columns(3)
    columns[0].metric("Processing runs", len(runs))
    columns[1].metric("Completed analyses", len(completed))
    columns[2].metric("Confirmed incidents", len(incidents))
    for column, status in zip(st.columns(3), REVIEW_STATUSES, strict=True):
        column.metric(status.title(), sum(row["review_status"] == status for row in incidents))
    if incidents:
        table = pd.DataFrame(incidents)
        table["day_utc"] = pd.to_datetime(table["created_at_utc"], utc=True).dt.date
        daily = table.groupby("day_utc").size().reset_index(name="Incidents")
        counts = table.groupby("review_status").size().reset_index(name="Incidents")
        left, right = st.columns(2)
        left.plotly_chart(px.bar(daily, x="day_utc", y="Incidents", title="Incidents by UTC date"))
        right.plotly_chart(px.bar(counts, x="review_status", y="Incidents", title="Review status"))
    else:
        st.info("Charts will appear after the first stored incident.")
    if runs:
        st.subheader("Run history")
        st.caption("A run left running after a server exit has an unknown final outcome.")
        st.dataframe(
            pd.DataFrame(runs)[
                [
                    "source_name",
                    "status",
                    "started_at_utc",
                    "model_identifier",
                    "image_size",
                    "frame_stride",
                    "processed_frames",
                    "mean_inference_latency_ms",
                    "processing_duration_seconds",
                    "incident_count",
                ]
            ],
            hide_index=True,
        )
    st.caption("Detection accuracy on representative surveillance footage: Not yet evaluated.")

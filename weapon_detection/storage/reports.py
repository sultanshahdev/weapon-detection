"""Export persisted incident records as spreadsheet-safe UTF-8 CSV."""

import csv
from collections.abc import Iterable
from io import StringIO

from weapon_detection.utils.csv import spreadsheet_safe

INCIDENT_FIELDS = (
    "incident_id",
    "created_at_utc",
    "source_name",
    "source_frame_index",
    "timestamp_seconds",
    "class_name",
    "normalized_class",
    "confidence",
    "confirmation_count",
    "detection_count",
    "review_status",
    "review_notes",
    "reviewed_at_utc",
    "snapshot_path",
    "snapshot_sha256",
    "run_id",
)


def incident_csv(records: Iterable[dict]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INCIDENT_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                key: spreadsheet_safe(record["id" if key == "incident_id" else key])
                for key in INCIDENT_FIELDS
            }
        )
    return stream.getvalue().encode("utf-8")

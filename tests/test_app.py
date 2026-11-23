import importlib
import runpy
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from weapon_detection.utils import config

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_package_and_app_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_call(*args: object, **kwargs: object) -> None:
        pytest.fail("Import called application startup code")

    monkeypatch.setattr(st, "set_page_config", unexpected_call)
    monkeypatch.setattr(config, "load_settings", unexpected_call)
    for module in ("detection", "incidents", "storage", "ui", "utils"):
        importlib.import_module(f"weapon_detection.{module}")
    assert callable(runpy.run_path(str(APP_PATH))["main"])


@pytest.mark.parametrize(
    ("overrides", "element_type", "message"),
    [
        ({}, "caption", "No local model configured"),
        ({"WD_MODEL_PATH": "missing.pt"}, "warning", "model file is unavailable"),
        ({"WD_CONFIDENCE_THRESHOLD": "-1"}, "error", "WD_CONFIDENCE_THRESHOLD"),
    ],
)
def test_app_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, str],
    element_type: str,
    message: str,
) -> None:
    def isolated_settings() -> config.Settings:
        return original_load_settings(tmp_path, environ=overrides)

    original_load_settings = config.load_settings
    monkeypatch.setattr(config, "load_settings", isolated_settings)
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    assert app.title[0].value == (
        "Real-Time Weapon Detection and AI-Powered Threat Intelligence System"
    )
    assert "foundation is running" in app.info[0].value
    assert message in getattr(app, element_type)[0].value

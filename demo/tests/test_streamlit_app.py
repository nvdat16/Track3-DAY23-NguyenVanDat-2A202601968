"""Smoke test for the separate Streamlit demo application."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="Streamlit demo extra is not installed",
)


def test_streamlit_app_renders():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "LangGraph Support Console"
    assert [tab.label for tab in app.tabs] == ["Agent", "Scenarios", "Architecture", "Report"]

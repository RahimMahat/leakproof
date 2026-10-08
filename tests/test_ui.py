"""The dashboard renders from the committed results without raising."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "ui" / "app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    assert len(app.metric) >= 10

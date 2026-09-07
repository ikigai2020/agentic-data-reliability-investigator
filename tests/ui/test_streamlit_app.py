"""The Streamlit app itself (M5.3).

Streamlit executes the app as ``__main__`` with no parent package, and only when a
session connects — so an import error inside it does not show up when the server starts,
or when anything imports the view functions. It shows up in front of an audience.
``AppTest`` runs the real script through Streamlit's own script runner, which is the only
check that catches that class of failure.

Skipped when the optional ``ui`` extra is not installed: the UI is not required to run
the system, and the rest of the suite must not depend on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from investigator.app import run_investigation

from ..conftest import DIAGNOSED

pytest.importorskip("streamlit", reason="UI extra not installed (uv sync --extra ui)")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "src" / "investigator" / "ui" /
          "streamlit_app.py")


async def _app() -> AppTest:
    # A run has to exist for the picker to have anything to show.
    await run_investigation(DIAGNOSED)
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    return app


async def test_the_app_renders_without_raising() -> None:
    app = await _app()
    assert not app.exception, app.exception


async def test_the_project_is_named_at_the_top() -> None:
    app = await _app()
    assert "Agentic Data Reliability Investigator" in [t.value for t in app.title]


async def test_the_diagnosis_is_stated_before_the_detail_that_supports_it() -> None:
    """Title, then what was concluded, then the numbers — in that order."""
    app = await _app()
    headline = app.markdown[0].value

    assert "diagnosed" in headline
    assert "transformation_logic" in headline, "the cause itself, not just the outcome word"


async def test_every_tab_is_present() -> None:
    app = await _app()
    assert len(app.tabs) == 9


async def test_the_header_carries_the_verdict() -> None:
    app = await _app()
    metrics = {m.label: m.value for m in app.metric}

    assert metrics["Confidence"] in {"moderate", "high", "low", "not_applicable", "—"}
    assert metrics["Release"] == "released"
    assert "/" in metrics["Calls"], "calls are shown against the budget, not alone"


async def test_the_run_picker_offers_journalled_runs() -> None:
    app = await _app()
    assert app.sidebar.selectbox, "no run picker rendered"
    assert app.sidebar.radio[0].options == ["Replay", "Run live"]

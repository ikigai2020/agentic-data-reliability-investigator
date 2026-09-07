"""Investigation UI (M5.3).

Split deliberately: :mod:`loader` and :mod:`views` are pure functions over persisted runs
and carry the whole of the UI's logic, so they are testable without importing Streamlit.
``streamlit_app`` is a thin rendering layer over them. A UI whose logic can only be
exercised by clicking it is a UI nobody can trust in front of an audience.
"""

from .loader import RunView, latest_run, list_runs, load_run
from .views import (
    agent_cards,
    beam_rounds,
    critic_rounds,
    evidence_rows,
    header,
    investigation,
    memory_view,
    orchestration_dot,
    safety,
    traces,
)

__all__ = [
    "RunView",
    "agent_cards",
    "beam_rounds",
    "critic_rounds",
    "evidence_rows",
    "header",
    "investigation",
    "latest_run",
    "list_runs",
    "load_run",
    "memory_view",
    "orchestration_dot",
    "safety",
    "traces",
]

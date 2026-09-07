"""Conditional routing for the parent graph (FR-801).

Routing reads decisions that deterministic nodes have already made and written to state;
it never makes one itself. In particular ``route_after_stop`` only reads the
``should_continue`` flag computed by the stopping evaluator, so stop authority stays in
one place (FR-903).
"""

from __future__ import annotations

from ..schemas.state import GlobalInvestigationState


def route_after_parse(state: GlobalInvestigationState) -> str:
    """parse failure -> invalid_input; otherwise proceed to verification."""
    return "invalid" if state.get("parse_error") else "ok"


def route_after_verify(state: GlobalInvestigationState) -> str:
    """verified -> commander planning; otherwise -> deterministic stop evaluation."""
    return "verified" if state.get("verification_status") == "verified" else "stop"


def route_after_stop(state: GlobalInvestigationState) -> str:
    """continue -> next commander round; otherwise -> report and validation."""
    return "continue" if state.get("should_continue") else "stop"

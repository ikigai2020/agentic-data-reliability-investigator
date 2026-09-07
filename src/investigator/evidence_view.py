"""Read-only queries over collected evidence (AD-005).

Shared by the stopping evaluator, the branch scorer, and the Evidence Critic so all
three answer "what does current evidence say about this hypothesis?" identically.

Every helper here filters to ``current_operational`` evidence. Retrieved incidents and
runbooks are deliberately invisible to these queries: memory may reorder an
investigation but never counts toward a diagnosis (AD-005, FR-604).
"""

from __future__ import annotations

from .schemas.evidence import Evidence


def current(evidence: list[Evidence]) -> list[Evidence]:
    return [e for e in evidence if e.source_kind == "current_operational"]


def supports(hyp_id: str, evidence: list[Evidence]) -> list[Evidence]:
    return [e for e in current(evidence) if hyp_id in e.supports]


def contradictions(hyp_id: str, evidence: list[Evidence]) -> list[Evidence]:
    return [e for e in current(evidence) if hyp_id in e.contradicts]


def independent_support_tools(hyp_id: str, evidence: list[Evidence]) -> set[str]:
    """Distinct tools producing supporting current observations.

    Independence is measured by distinct tool, so two readings from the same tool never
    count as two independent supports (FR-900.2).
    """
    return {e.tool_name for e in supports(hyp_id, evidence) if e.tool_name}


def has_discriminating_support(hyp_id: str, evidence: list[Evidence]) -> bool:
    return any(e.provenance.get("discriminating") for e in supports(hyp_id, evidence))


def has_critical_contradiction(hyp_id: str, evidence: list[Evidence]) -> bool:
    """A contradiction is *critical* when it comes from a discriminating observation."""
    return any(e.provenance.get("discriminating") for e in contradictions(hyp_id, evidence))


def was_tested(hyp_id: str, evidence: list[Evidence]) -> bool:
    """True if any current observation references the hypothesis or came from its task."""
    for e in current(evidence):
        if hyp_id in e.supports or hyp_id in e.contradicts or e.task_id.endswith(hyp_id):
            return True
    return False


def had_discriminating_check(hyp_id: str, evidence: list[Evidence]) -> bool:
    """True if a discriminating current observation has been taken for this hypothesis.

    Used by the FR-704 diversity rule, which protects a hypothesis from pruning until it
    has actually been given a fair test.
    """
    for e in current(evidence):
        if not e.provenance.get("discriminating"):
            continue
        if hyp_id in e.supports or hyp_id in e.contradicts or e.task_id.endswith(hyp_id):
            return True
    return False


def tools_used_for(hyp_id: str, evidence: list[Evidence]) -> set[str]:
    """Every tool already run on this hypothesis's behalf, whatever the result."""
    used: set[str] = set()
    for e in current(evidence):
        if e.tool_name is None:
            continue
        if hyp_id in e.supports or hyp_id in e.contradicts or e.task_id.endswith(hyp_id):
            used.add(e.tool_name)
    return used

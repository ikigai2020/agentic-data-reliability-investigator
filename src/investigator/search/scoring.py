"""Branch scoring rubric (FR-703).

The rubric is deterministic and fully itemized — every point a branch receives carries a
reason string, so a score can be audited rather than trusted.

| Criterion                                     | Points |
|-----------------------------------------------|-------:|
| Consistency with current operational evidence |    0-4 |
| Discriminating value of next action           |    0-2 |
| Independence and quality of support           |    0-2 |
| Applicable retrieved context                  |    0-1 |
| Expected cost and latency efficiency          |    0-1 |

A critical current contradiction applies a three-point penalty.

Only current operational evidence moves the first three criteria. Retrieved context is
worth at most one point and can never carry a branch over the pruning threshold on its
own (AD-005).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import evidence_view
from ..schemas.branch import InvestigationBranch
from ..schemas.enums import RootCauseCategory
from ..schemas.evidence import Evidence
from ..schemas.hypothesis import Hypothesis
from .thought import Thought

MAX_SCORE = 10
CONTRADICTION_PENALTY = -3


@dataclass(frozen=True)
class BranchScore:
    """Itemized FR-703 score for one branch."""

    branch_id: str
    hypothesis_id: str
    consistency: int
    discriminating_value: int
    support_independence: int
    retrieved_context: int
    cost_efficiency: int
    contradiction_penalty: int
    reasons: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.consistency
            + self.discriminating_value
            + self.support_independence
            + self.retrieved_context
            + self.cost_efficiency
            + self.contradiction_penalty
        )


def _consistency(hyp_id: str, evidence: list[Evidence], reasons: list[str]) -> int:
    """0-4: how well current evidence fits the hypothesis."""
    if evidence_view.contradictions(hyp_id, evidence):
        reasons.append("consistency 0: current evidence contradicts this hypothesis")
        return 0
    n = len(evidence_view.independent_support_tools(hyp_id, evidence))
    points = 1 + min(3, n)
    reasons.append(f"consistency {points}: {n} independent supporting tool(s), no contradiction")
    return points


def _support_independence(hyp_id: str, evidence: list[Evidence], reasons: list[str]) -> int:
    """0-2: distinct supporting tools, capped at two (FR-900.2 independence)."""
    tools = evidence_view.independent_support_tools(hyp_id, evidence)
    points = min(2, len(tools))
    reasons.append(f"independence {points}: support from {sorted(tools) or 'no tools'}")
    return points


def score_branch(
    branch: InvestigationBranch,
    hypothesis: Hypothesis,
    evidence: list[Evidence],
    *,
    next_thought: Thought | None,
    retrieval_categories: set[RootCauseCategory] | None = None,
    calls_remaining: int,
) -> BranchScore:
    """Apply the FR-703 rubric to one branch."""
    reasons: list[str] = []

    consistency = _consistency(hypothesis.hypothesis_id, evidence, reasons)
    independence = _support_independence(hypothesis.hypothesis_id, evidence, reasons)

    if next_thought is None:
        discriminating = 0
        reasons.append("discriminating 0: no candidate action remains")
    else:
        discriminating = next_thought.discriminating_value
        reasons.append(
            f"discriminating {discriminating}: next action {next_thought.proposed_action}"
        )

    retrieved = 1 if hypothesis.category in (retrieval_categories or set()) else 0
    if retrieved:
        reasons.append("retrieved_context 1: accepted memory matches this category")

    # Affordable next step within the remaining global operational budget (FR-702).
    if next_thought is not None and next_thought.estimated_cost <= calls_remaining:
        cost = 1
        reasons.append("cost_efficiency 1: next action affordable within budget")
    else:
        cost = 0
        reasons.append("cost_efficiency 0: no affordable next action")

    penalty = 0
    if evidence_view.has_critical_contradiction(hypothesis.hypothesis_id, evidence):
        penalty = CONTRADICTION_PENALTY
        reasons.append(f"penalty {CONTRADICTION_PENALTY}: critical current contradiction")

    return BranchScore(
        branch_id=branch.branch_id,
        hypothesis_id=hypothesis.hypothesis_id,
        consistency=consistency,
        discriminating_value=discriminating,
        support_independence=independence,
        retrieved_context=retrieved,
        cost_efficiency=cost,
        contradiction_penalty=penalty,
        reasons=reasons,
    )

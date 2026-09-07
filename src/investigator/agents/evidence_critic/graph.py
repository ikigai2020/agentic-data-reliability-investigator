"""Evidence Critic agent (FR-130).

A distinct agent under §6.1: its own goal and role prompt, bounded input/output
contracts (investigation state in, one :class:`CriticReview` out), an empty tool set
(``critic`` maps to no MCP tools in :mod:`investigator.mcp_client.permissions`), local
working state in its own subgraph, authority to choose which reviews to perform, a
review loop, and explicit completion and failure conditions.

The Critic is adversarial by design and advisory by contract. It recommends; the
deterministic controller in :mod:`investigator.search.beam` and
:mod:`investigator.graph.stopping` decides (FR-130, FR-903).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ... import evidence_view
from ...graph.reducers import extend_unique
from ...schemas.branch import InvestigationBranch
from ...schemas.critic import CriticReview
from ...schemas.enums import StopRecommendation
from ...schemas.evidence import Evidence
from ...schemas.finding import AgentFinding
from ...schemas.hypothesis import Hypothesis
from ..prompts import PROMPT_VERSION, load_prompt

ROLE = "critic"
AGENT_NAME = "evidence_critic"
PROMPT = load_prompt("evidence_critic")
PROMPT_ID = f"evidence_critic:{PROMPT_VERSION}"

# Provenance keys every current observation must carry to be auditable (FR-1002).
REQUIRED_PROVENANCE = ("request_id",)


class CriticState(TypedDict, total=False):
    """The Critic's local working state (FR-202) — never written to global state."""

    # inputs
    round_number: int
    hypotheses: list[Hypothesis]
    evidence: list[Evidence]
    findings: list[AgentFinding]
    branches: list[InvestigationBranch]
    branch_scores: dict[str, float]
    retrieval_influence: Any
    min_supporting: int
    prune_threshold: int
    # Injected LLM judgment layer (optional); ``None`` keeps the review fully rule-based.
    critic_llm: Any
    # working
    unsupported_claims: Annotated[list[str], extend_unique]
    evidence_gaps: Annotated[list[str], extend_unique]
    independence_notes: Annotated[list[str], extend_unique]
    bias_challenges: Annotated[list[str], extend_unique]
    strongest_hypothesis_id: str | None
    strongest_competitor_id: str | None
    model_recommendation: str | None
    model_rationale: str
    # output
    review: CriticReview | None


def _active(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    return [h for h in hypotheses if h.status in {"active", "supported"}]


async def review_claims(state: CriticState, config: RunnableConfig) -> dict:
    """FR-130.2/130.3: unsupported claims, missing provenance, contradictions, reuse."""
    evidence = state.get("evidence", [])
    findings = state.get("findings", [])
    hypotheses = state.get("hypotheses", [])
    known_ids = {e.evidence_id for e in evidence}

    claims: list[str] = []
    gaps: list[str] = []

    # A finding that asserts support must cite evidence that exists and actually says so.
    for finding in findings:
        dangling = [eid for eid in finding.evidence_ids if eid not in known_ids]
        if dangling:
            claims.append(f"{finding.task_id}: cites evidence not in state {sorted(dangling)}")
        for hyp_id in finding.supports:
            if not evidence_view.supports(hyp_id, evidence):
                claims.append(
                    f"{finding.task_id}: claims support for {hyp_id} with no current observation"
                )
        if finding.completion_status == "failed" and finding.supports:
            claims.append(f"{finding.task_id}: failed task still asserts support")

    # Provenance completeness on current observations.
    for e in evidence_view.current(evidence):
        missing = [k for k in REQUIRED_PROVENANCE if not e.provenance.get(k)]
        if missing:
            gaps.append(f"{e.evidence_id}: missing provenance {missing}")

    # Contradictions held simultaneously with support.
    for h in hypotheses:
        hid = h.hypothesis_id
        if evidence_view.supports(hid, evidence) and evidence_view.contradictions(hid, evidence):
            gaps.append(f"{hid}: current evidence both supports and contradicts this hypothesis")

    # Evidence reuse: one observation propping up more than one hypothesis.
    for e in evidence_view.current(evidence):
        if len(set(e.supports)) > 1:
            gaps.append(
                f"{e.evidence_id}: reused as support for {sorted(set(e.supports))} — "
                "not independent corroboration"
            )

    # Untested hypotheses are gaps, not weaknesses.
    for h in _active(hypotheses):
        if not evidence_view.was_tested(h.hypothesis_id, evidence):
            gaps.append(f"{h.hypothesis_id}: never tested by a current observation")

    return {"unsupported_claims": claims, "evidence_gaps": gaps}


async def assess_independence(state: CriticState, config: RunnableConfig) -> dict:
    """FR-130.4: are the sources behind each hypothesis genuinely independent?"""
    evidence = state.get("evidence", [])
    notes: list[str] = []
    gaps: list[str] = []

    for h in state.get("hypotheses", []):
        hid = h.hypothesis_id
        supporting = evidence_view.supports(hid, evidence)
        if not supporting:
            continue
        tools = {e.tool_name for e in supporting if e.tool_name}
        systems = {e.source_name for e in supporting if e.source_name}
        notes.append(
            f"{hid}: {len(supporting)} observation(s) from {len(tools)} tool(s), "
            f"{len(systems)} source system(s)"
        )
        if len(supporting) > len(tools):
            gaps.append(
                f"{hid}: {len(supporting)} supporting observations but only {len(tools)} "
                "distinct tool(s) — repeated readings are one source"
            )
        if not evidence_view.has_discriminating_support(hid, evidence):
            gaps.append(f"{hid}: supported only by non-discriminating observations")

    return {"independence_notes": notes, "evidence_gaps": gaps}


async def rank_hypotheses(state: CriticState, config: RunnableConfig) -> dict:
    """FR-130.6: identify the leading hypothesis and its strongest competitor."""
    evidence = state.get("evidence", [])
    hypotheses = state.get("hypotheses", [])

    def strength(h: Hypothesis) -> tuple[int, int, int]:
        return (
            len(evidence_view.independent_support_tools(h.hypothesis_id, evidence)),
            1 if evidence_view.has_discriminating_support(h.hypothesis_id, evidence) else 0,
            -h.rank,
        )

    ranked = sorted(
        [h for h in hypotheses if h.status != "rejected"], key=strength, reverse=True
    )
    leading = ranked[0].hypothesis_id if ranked else None
    competitor = ranked[1].hypothesis_id if len(ranked) > 1 else None
    return {"strongest_hypothesis_id": leading, "strongest_competitor_id": competitor}


async def challenge_bias(state: CriticState, config: RunnableConfig) -> dict:
    """FR-130.8: push back on the Commander's ordering and on premature confidence."""
    evidence = state.get("evidence", [])
    hypotheses = state.get("hypotheses", [])
    leading_id = state.get("strongest_hypothesis_id")
    competitor_id = state.get("strongest_competitor_id")
    challenges: list[str] = []

    by_id = {h.hypothesis_id: h for h in hypotheses}
    leading = by_id.get(leading_id) if leading_id else None

    if competitor_id and not evidence_view.had_discriminating_check(competitor_id, evidence):
        challenges.append(
            f"strongest competitor {competitor_id} has had no discriminating check — "
            "the lead is unearned until it does"
        )

    if leading is not None and leading.rank == 1:
        others_tested = [
            h.hypothesis_id
            for h in hypotheses
            if h.hypothesis_id != leading.hypothesis_id
            and evidence_view.was_tested(h.hypothesis_id, evidence)
        ]
        if not others_tested:
            challenges.append(
                f"leading hypothesis {leading.hypothesis_id} is also the initially "
                "top-ranked one and no alternative has been tested — confirmation bias risk"
            )

    if leading is not None and leading.origin == "retrieval_influenced":
        challenges.append(
            f"{leading.hypothesis_id} leads partly because retrieved memory promoted it; "
            "confirm the diagnosis rests on current observations only"
        )

    # Retrieval must not be doing evidentiary work.
    for e in evidence:
        if e.source_kind != "current_operational" and (e.supports or e.contradicts):
            challenges.append(
                f"{e.evidence_id}: historical evidence carrying support/contradiction"
            )

    # FR-130.8 is judgment, so the model gets it — layered on top of the rules above,
    # never in place of them.
    critic_llm = state.get("critic_llm")
    if critic_llm is None:
        return {"bias_challenges": challenges}

    judgment = await critic_llm.judge(_review_context(state))
    if not judgment.consulted:
        return {"bias_challenges": challenges}

    return {
        "bias_challenges": [*challenges, *judgment.bias_challenges],
        "evidence_gaps": judgment.evidence_gaps,
        "model_recommendation": judgment.stop_recommendation,
        "model_rationale": judgment.rationale,
    }


def _review_context(state: CriticState) -> dict[str, Any]:
    """The round, flattened for review. Payloads are summaries, never raw tool output."""
    evidence = state.get("evidence", [])
    leading_id = state.get("strongest_hypothesis_id")
    competitor_id = state.get("strongest_competitor_id")
    return {
        "round_number": state.get("round_number", 0),
        "leading_hypothesis_id": leading_id,
        "strongest_competitor_id": competitor_id,
        "hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "category": h.category.value,
                "statement": h.statement,
                "rank": h.rank,
                "status": h.status,
                "origin": h.origin,
                "independent_supporting_tools": sorted(
                    evidence_view.independent_support_tools(h.hypothesis_id, evidence)
                ),
                "contradicted_by": [
                    e.tool_name for e in evidence_view.contradictions(h.hypothesis_id, evidence)
                ],
                "had_discriminating_check": evidence_view.had_discriminating_check(
                    h.hypothesis_id, evidence
                ),
            }
            for h in state.get("hypotheses", [])
        ],
        "current_observations": [
            {
                "evidence_id": e.evidence_id,
                "tool": e.tool_name,
                "summary": e.summary,
                "supports": e.supports,
                "contradicts": e.contradicts,
                "discriminating": bool(e.provenance.get("discriminating")),
            }
            for e in evidence_view.current(evidence)
        ],
        "historical_context": [
            {"evidence_id": e.evidence_id, "summary": e.summary}
            for e in evidence
            if e.source_kind != "current_operational"
        ],
        "branch_scores": state.get("branch_scores", {}),
        "deterministic_findings": {
            "unsupported_claims": state.get("unsupported_claims", []),
            "evidence_gaps": state.get("evidence_gaps", []),
            "independence_notes": state.get("independence_notes", []),
        },
    }


async def recommend(state: CriticState, config: RunnableConfig) -> dict:
    """FR-130.5/130.7: score-driven pruning, reopening, and a stop recommendation."""
    evidence = state.get("evidence", [])
    branches = state.get("branches", [])
    scores = state.get("branch_scores", {})
    threshold = state.get("prune_threshold", 5)
    min_supporting = state.get("min_supporting", 2)
    leading_id = state.get("strongest_hypothesis_id")

    pruning: list[str] = []
    reopening: list[str] = []
    for branch in branches:
        score = scores.get(branch.branch_id)
        if branch.status in {"active", "selected", "reopened"}:
            if evidence_view.has_critical_contradiction(branch.hypothesis_id, evidence):
                pruning.append(f"{branch.branch_id}: falsified by a discriminating observation")
            elif score is not None and score < threshold:
                pruning.append(f"{branch.branch_id}: score {score:g} below threshold {threshold}")
        elif branch.status == "pruned" and evidence_view.supports(
            branch.hypothesis_id, evidence
        ):
            reopening.append(
                f"{branch.branch_id}: new current evidence supports {branch.hypothesis_id}"
            )

    # Stop recommendation — advisory only; the deterministic evaluator decides (FR-903).
    recommendation: StopRecommendation = "continue"
    if leading_id is None:
        recommendation = "inconclusive"
    else:
        independent = evidence_view.independent_support_tools(leading_id, evidence)
        discriminating = evidence_view.has_discriminating_support(leading_id, evidence)
        contradicted = bool(evidence_view.contradictions(leading_id, evidence))
        if len(independent) >= min_supporting and discriminating and not contradicted:
            recommendation = "diagnosed"
        elif not any(
            b.status in {"active", "selected", "reopened"} for b in branches
        ):
            recommendation = "inconclusive"

    claims = state.get("unsupported_claims", [])
    gaps = state.get("evidence_gaps", [])
    challenges = state.get("bias_challenges", [])
    if challenges:
        # A bias challenge is not a veto, but it is always recorded — an objection that
        # only survives when the Critic happens to be recommending a diagnosis is an
        # objection nobody ever reads.
        gaps = [*gaps, *challenges]

    # The model's recommendation is advisory *to an advisory agent* — recorded, and
    # allowed to be more cautious than the rules, never less. It can move a review from
    # `diagnosed` to `continue`/`inconclusive`, but it cannot manufacture a diagnosis.
    model_recommendation = state.get("model_recommendation")
    model_note = ""
    if model_recommendation and model_recommendation != recommendation:
        if recommendation == "diagnosed":
            model_note = (
                f" model recommended '{model_recommendation}' over 'diagnosed'; "
                "deferring to the more cautious view"
            )
            recommendation = model_recommendation  # type: ignore[assignment]
        else:
            model_note = (
                f" model recommended '{model_recommendation}'; rules kept "
                f"'{recommendation}' (a model may not upgrade caution to a diagnosis)"
            )

    rationale = (
        f"round {state.get('round_number', 0)}: "
        f"{len(claims)} unsupported claim(s), {len(gaps)} evidence gap(s), "
        f"{len(pruning)} prune / {len(reopening)} reopen recommendation(s); "
        f"advisory stop recommendation={recommendation}.{model_note}"
    )
    if state.get("model_rationale"):
        rationale = f"{rationale} Critic judgment: {state['model_rationale']}"

    review = CriticReview(
        review_id=f"cr_r{state.get('round_number', 0)}",
        round_number=state.get("round_number", 0),
        branch_scores=dict(scores),
        strongest_hypothesis_id=leading_id,
        strongest_competitor_id=state.get("strongest_competitor_id"),
        unsupported_claims=claims,
        evidence_gaps=gaps,
        pruning_recommendations=pruning,
        reopening_recommendations=reopening,
        stop_recommendation=recommendation,
        rationale_summary=rationale,
    )
    return {"review": review}


def build_evidence_critic_subgraph():
    """Compile the Evidence Critic review loop."""
    g = StateGraph(CriticState)
    g.add_node("review_claims", review_claims)
    g.add_node("assess_independence", assess_independence)
    g.add_node("rank_hypotheses", rank_hypotheses)
    g.add_node("challenge_bias", challenge_bias)
    g.add_node("recommend", recommend)

    g.add_edge(START, "review_claims")
    g.add_edge("review_claims", "assess_independence")
    g.add_edge("assess_independence", "rank_hypotheses")
    g.add_edge("rank_hypotheses", "challenge_bias")
    g.add_edge("challenge_bias", "recommend")
    g.add_edge("recommend", END)
    return g.compile()


def build():
    """Return the compiled Evidence Critic subgraph."""
    return build_evidence_critic_subgraph()

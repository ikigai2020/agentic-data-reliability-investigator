"""Beam search over investigation branches (FR-701, FR-702, FR-704, FR-705, FR-706).

A *node* is an investigation snapshot after an action, a *branch* is the sequence of
actions testing one hypothesis, *depth* is the number of evidence-gathering cycles that
branch has been through, and the *root* is the verified incident (FR-701).

Everything in this module is deterministic. The Evidence Critic may recommend pruning
and reopening, but only the functions here decide, and they decide from evidence and
budgets alone (FR-130, FR-903).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import evidence_view
from ..schemas.branch import InvestigationBranch
from ..schemas.enums import RootCauseCategory
from ..schemas.evidence import Evidence
from ..schemas.hypothesis import Hypothesis
from .scoring import BranchScore, score_branch
from .thought import Thought, candidate_tools, propose

# Branch statuses that are still in play at the start of a policy pass.
LIVE_STATUSES = frozenset({"active", "selected", "reopened"})

# Large enough to ask ``candidate_tools`` for a category's whole plan rather than one
# round's worth of it — used only to work out which tools a branch was denied.
_WHOLE_PLAN = 100


@dataclass(frozen=True)
class BeamPolicy:
    """FR-702 configurable defaults."""

    beam_width: int = 3
    max_branch_depth: int = 4
    max_candidate_actions: int = 2
    prune_threshold: int = 5
    min_diverse_branches: int = 2


@dataclass
class BeamDecision:
    branches: list[InvestigationBranch]
    scores: dict[str, BranchScore] = field(default_factory=dict)
    thoughts: dict[str, Thought] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    pruned: dict[str, str] = field(default_factory=dict)
    reopened: dict[str, str] = field(default_factory=dict)
    protected_ids: list[str] = field(default_factory=list)
    closed: dict[str, str] = field(default_factory=dict)


def initialize_branches(hypotheses: list[Hypothesis]) -> list[InvestigationBranch]:
    """One root-level branch per hypothesis (FR-100.4). The root is the verified incident."""
    return [
        InvestigationBranch(
            branch_id=f"B-{h.hypothesis_id}",
            hypothesis_id=h.hypothesis_id,
            depth=0,
            parent_branch_id=None,
            status="active",
        )
        for h in sorted(hypotheses, key=lambda h: h.rank)
    ]


def advance_branches(
    branches: list[InvestigationBranch], evidence: list[Evidence]
) -> list[InvestigationBranch]:
    """Fold the round's observations into each branch's history and depth (FR-701).

    Depth counts *evidence-gathering cycles*, derived from the distinct task rounds that
    produced evidence for the branch's hypothesis — not the number of tool calls.
    """
    advanced: list[InvestigationBranch] = []
    for branch in branches:
        hyp_id = branch.hypothesis_id
        attributable = [
            e
            for e in evidence_view.current(evidence)
            if hyp_id in e.supports or hyp_id in e.contradicts or e.task_id.endswith(hyp_id)
        ]
        rounds = {e.task_id.split("-", 1)[0] for e in attributable}
        advanced.append(
            branch.model_copy(
                update={
                    "evidence_ids": sorted({e.evidence_id for e in attributable}),
                    "action_history": sorted(evidence_view.tools_used_for(hyp_id, evidence)),
                    "depth": len(rounds),
                }
            )
        )
    return advanced


def _tie_key(
    score: BranchScore, thought: Thought | None, branch: InvestigationBranch
) -> tuple[int, int, int, int, str]:
    """FR-706: stronger current evidence, greater discriminating value, lower cost, stable ID."""
    return (
        -score.total,
        -score.support_independence,
        -score.discriminating_value,
        thought.estimated_cost if thought is not None else 1_000,
        branch.branch_id,
    )


def _prune_reason(
    branch: InvestigationBranch,
    score: BranchScore,
    thought: Thought | None,
    *,
    evidence: list[Evidence],
    known_evidence_ids: set[str],
    duplicate_of: str | None,
    blocked_tools: set[str],
    policy: BeamPolicy,
    budget_exhausted: bool,
) -> tuple[str, str] | None:
    """Return ``(disposition, reason)`` or ``None`` when the branch survives.

    Disposition is ``"pruned"`` for a policy prune (FR-704) or ``"closed"`` for a branch
    that simply ran out of permitted actions — an exhausted branch is not a failed one.
    """
    hyp_id = branch.hypothesis_id

    if not set(branch.evidence_ids).issubset(known_evidence_ids):
        missing = sorted(set(branch.evidence_ids) - known_evidence_ids)
        return "pruned", f"depends on evidence not in accepted state: {missing}"

    if evidence_view.has_critical_contradiction(hyp_id, evidence):
        return "pruned", "directly falsified by a discriminating current observation"

    if duplicate_of is not None:
        return "pruned", f"duplicates the tested path of branch {duplicate_of}"

    if budget_exhausted:
        return "pruned", "global operational-call budget exhausted"

    if branch.depth >= policy.max_branch_depth:
        return "pruned", f"exceeded maximum branch depth ({policy.max_branch_depth})"

    if thought is None:
        # ``blocked_tools`` is what this branch *needed* and could not reach, which is
        # not the same as what it managed to run: under a full server outage a branch
        # has no action history at all, and reporting that as "exhausted" would hide the
        # outage behind a routine disposition.
        if blocked_tools and not evidence_view.supports(hyp_id, evidence):
            return (
                "pruned",
                f"required tool unavailable without alternative: {sorted(blocked_tools)}",
            )
        return "closed", "candidate actions exhausted"

    if score.total < policy.prune_threshold:
        return "pruned", f"score {score.total} below threshold {policy.prune_threshold}"

    return None


def _duplicate_paths(branches: list[InvestigationBranch]) -> dict[str, str]:
    """Map branch_id -> the earlier branch whose tested path it duplicates."""
    seen: dict[tuple[str, ...], str] = {}
    duplicates: dict[str, str] = {}
    for branch in sorted(branches, key=lambda b: b.branch_id):
        path = tuple(branch.action_history)
        if not path:
            continue  # an untested branch duplicates nothing
        if path in seen:
            duplicates[branch.branch_id] = seen[path]
        else:
            seen[path] = branch.branch_id
    return duplicates


def apply_policy(
    branches: list[InvestigationBranch],
    hypotheses: list[Hypothesis],
    evidence: list[Evidence],
    *,
    policy: BeamPolicy,
    calls_remaining: int,
    retrieval_categories: set[RootCauseCategory] | None = None,
    unavailable_tools: set[str] | None = None,
) -> BeamDecision:
    """Score, prune, protect diversity, reopen, and select the beam.

    Returns a decision carrying updated branches plus an audit trail of what happened to
    every branch and why.
    """
    unavailable_tools = unavailable_tools or set()
    by_hyp = {h.hypothesis_id: h for h in hypotheses}
    known_evidence_ids = {e.evidence_id for e in evidence}
    budget_exhausted = calls_remaining <= 0
    duplicates = _duplicate_paths([b for b in branches if b.status in LIVE_STATUSES])

    decision = BeamDecision(branches=[])
    updated: dict[str, InvestigationBranch] = {}

    # --- score every branch, live or previously pruned (a pruned branch may reopen) ---
    for branch in branches:
        hypothesis = by_hyp.get(branch.hypothesis_id)
        if hypothesis is None:  # defensive: branch without a surviving hypothesis
            updated[branch.branch_id] = branch
            continue
        # A tool that failed this run is not an untried option. The Commander applies the
        # same exclusion when planning, so without it the beam would believe a branch is
        # expandable that the next round cannot actually dispatch — the invariant
        # ``search.thought.candidate_tools`` documents.
        tried = (
            evidence_view.tools_used_for(branch.hypothesis_id, evidence)
            | set(branch.action_history)
            | unavailable_tools
        )
        thoughts = propose(hypothesis, branch.branch_id, tried, policy.max_candidate_actions)
        thought = thoughts[0] if thoughts else None
        score = score_branch(
            branch,
            hypothesis,
            evidence,
            next_thought=thought,
            retrieval_categories=retrieval_categories,
            calls_remaining=calls_remaining,
        )
        decision.scores[branch.branch_id] = score
        if thought is not None:
            decision.thoughts[branch.branch_id] = thought
        updated[branch.branch_id] = branch.model_copy(update={"score": float(score.total)})

    # --- prune / close live branches (FR-704) ---
    survivors: list[str] = []
    for branch_id, branch in updated.items():
        if branch.status not in LIVE_STATUSES:
            continue
        branch_score = decision.scores.get(branch_id)
        if branch_score is None:
            continue
        hypothesis = by_hyp.get(branch.hypothesis_id)
        plan_tools = (
            set(candidate_tools(hypothesis.category, set(), _WHOLE_PLAN))
            if hypothesis is not None
            else set()
        )
        verdict = _prune_reason(
            branch,
            branch_score,
            decision.thoughts.get(branch_id),
            evidence=evidence,
            known_evidence_ids=known_evidence_ids,
            duplicate_of=duplicates.get(branch_id),
            blocked_tools=unavailable_tools & plan_tools,
            policy=policy,
            budget_exhausted=budget_exhausted,
        )
        if verdict is None:
            survivors.append(branch_id)
            continue
        disposition, reason = verdict
        updated[branch_id] = branch.model_copy(
            update={"status": disposition, "prune_reason": reason}
        )
        if disposition == "pruned":
            decision.pruned[branch_id] = reason
        else:
            decision.closed[branch_id] = reason

    # --- diversity protection (FR-704) ---
    survivors = _protect_diversity(updated, decision, survivors, evidence, policy)

    # --- reopening (FR-705) ---
    if not budget_exhausted:
        # Reopening costs calls. With none left there is nothing to reopen a branch
        # *for*, so an exhausted investigation stops rather than reviving work it
        # cannot do.
        survivors = _reopen(updated, decision, survivors, evidence, policy)

    # --- beam selection (FR-702 width, FR-706 ties) ---
    ordered = sorted(
        survivors,
        key=lambda bid: _tie_key(
            decision.scores[bid], decision.thoughts.get(bid), updated[bid]
        ),
    )
    decision.selected_ids = ordered[: policy.beam_width]
    for branch_id in ordered:
        if updated[branch_id].status == "reopened":
            # A branch that came back keeps saying so (FR-705). Beam membership is
            # carried by ``selected_ids``, so the status stays free to record history.
            continue
        status = "selected" if branch_id in decision.selected_ids else "active"
        updated[branch_id] = updated[branch_id].model_copy(update={"status": status})

    decision.branches = [updated[b.branch_id] for b in branches]
    return decision


def _protect_diversity(
    updated: dict[str, InvestigationBranch],
    decision: BeamDecision,
    survivors: list[str],
    evidence: list[Evidence],
    policy: BeamPolicy,
) -> list[str]:
    """FR-704: keep at least two diverse hypotheses alive until each has been tested.

    A branch pruned purely for a low score is restored when too few distinct root-cause
    categories survive and its hypothesis has not yet received a discriminating check.
    A falsified branch is never restored — it was tested, and it lost.
    """
    live_hypotheses = {updated[bid].hypothesis_id for bid in survivors}
    if len(live_hypotheses) >= policy.min_diverse_branches:
        return survivors

    candidates = [
        bid
        for bid, reason in decision.pruned.items()
        if reason.startswith("score ")
        and not evidence_view.had_discriminating_check(updated[bid].hypothesis_id, evidence)
        and decision.thoughts.get(bid) is not None
    ]
    candidates.sort(key=lambda bid: (-decision.scores[bid].total, bid))

    for bid in candidates:
        if len({updated[b].hypothesis_id for b in survivors}) >= policy.min_diverse_branches:
            break
        prior = decision.pruned.pop(bid)
        updated[bid] = updated[bid].model_copy(
            update={
                "status": "active",
                "prune_reason": None,
                "action_history": [
                    *updated[bid].action_history,
                    f"diversity hold (would have been pruned: {prior})",
                ],
            }
        )
        survivors.append(bid)
        decision.protected_ids.append(bid)
    return survivors


def _reopen(
    updated: dict[str, InvestigationBranch],
    decision: BeamDecision,
    survivors: list[str],
    evidence: list[Evidence],
    policy: BeamPolicy,
) -> list[str]:
    """FR-705: reopen a pruned branch and record the trigger and the prior prune reason.

    Two triggers: new current evidence specifically supports the pruned hypothesis, or
    every surviving branch now carries a current contradiction.
    """
    all_survivors_contradicted = bool(survivors) and all(
        evidence_view.contradictions(updated[bid].hypothesis_id, evidence) for bid in survivors
    )

    reopen_candidates: list[tuple[str, str]] = []
    for bid, branch in updated.items():
        if branch.status != "pruned":
            continue
        if bid in decision.protected_ids:
            continue
        if decision.thoughts.get(bid) is None:
            continue  # nothing left to do with it even if reopened
        if evidence_view.has_critical_contradiction(branch.hypothesis_id, evidence):
            continue  # falsified branches stay closed
        if evidence_view.supports(branch.hypothesis_id, evidence):
            reopen_candidates.append((bid, "new current evidence supports this hypothesis"))
        elif all_survivors_contradicted:
            reopen_candidates.append((bid, "all surviving branches carry a current contradiction"))

    reopen_candidates.sort(key=lambda pair: (-decision.scores[pair[0]].total, pair[0]))
    for bid, trigger in reopen_candidates[: policy.beam_width]:
        prior = updated[bid].prune_reason or decision.pruned.get(bid) or "unknown"
        updated[bid] = updated[bid].model_copy(
            update={
                "status": "reopened",
                "action_history": [
                    *updated[bid].action_history,
                    f"reopened: {trigger} (prior prune: {prior})",
                ],
            }
        )
        decision.pruned.pop(bid, None)
        decision.reopened[bid] = f"{trigger} (prior prune: {prior})"
        survivors.append(bid)
    return survivors

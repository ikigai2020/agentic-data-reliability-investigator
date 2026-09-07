"""Retrieval influence on planning + memory-as-context evidence (FR-604, AD-005).

Accepted memory may reorder hypotheses (change investigation *order*) or suggest a check,
and is recorded as ``retrieved_incident`` / ``runbook`` evidence for traceability — but it
never carries support/contradiction and never establishes a diagnosis (AD-005). The
recorded :class:`RetrievalInfluence` states exactly what retrieval did (FR-604).
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..schemas.evidence import Evidence
from ..schemas.hypothesis import Hypothesis
from ..schemas.retrieval import RetrievalInfluence, RetrievedItem, TrustDecision


def _accepted_docs(
    items: list[RetrievedItem], decisions: list[TrustDecision]
) -> list[RetrievedItem]:
    accepted_ids = {d.doc_id for d in decisions if d.accepted}
    return [i for i in items if i.document.doc_id in accepted_ids]


def apply_influence(
    hypotheses: list[Hypothesis],
    items: list[RetrievedItem],
    decisions: list[TrustDecision],
) -> tuple[list[Hypothesis], RetrievalInfluence]:
    """Reorder hypotheses from accepted memory and record the influence (FR-604)."""
    order_before = [h.hypothesis_id for h in sorted(hypotheses, key=lambda h: h.rank)]
    accepted = _accepted_docs(items, decisions)
    accepted_ids = [i.document.doc_id for i in accepted]
    rejected_ids = [d.doc_id for d in decisions if not d.accepted]

    if not items:
        return hypotheses, RetrievalInfluence(kind="not_run", detail="retrieval not run")

    if not accepted:
        return hypotheses, RetrievalInfluence(
            kind="rejected",
            detail="all retrieved items rejected by trust gate",
            rejected_doc_ids=rejected_ids,
            order_before=order_before,
            order_after=order_before,
        )

    # Categories confirmed by accepted *incidents* may reprioritize hypotheses.
    promoted_categories = {
        i.document.category
        for i in accepted
        if i.document.type == "incident" and i.document.category
    }
    by_rank = sorted(hypotheses, key=lambda h: h.rank)
    promoted = [h for h in by_rank if h.category in promoted_categories]
    rest = [h for h in by_rank if h.category not in promoted_categories]
    new_order_list = promoted + rest
    order_after = [h.hypothesis_id for h in new_order_list]

    # Reassign ordinal ranks and mark genuinely-moved hypotheses as retrieval-influenced.
    updated: list[Hypothesis] = []
    moved: list[str] = []
    for new_rank, hyp in enumerate(new_order_list, start=1):
        changed_position = order_before.index(hyp.hypothesis_id) != (new_rank - 1)
        origin = (
            "retrieval_influenced"
            if (changed_position and hyp.category in promoted_categories)
            else hyp.origin
        )
        if changed_position and hyp.category in promoted_categories:
            moved.append(hyp.hypothesis_id)
        updated.append(hyp.model_copy(update={"rank": new_rank, "origin": origin}))

    has_runbook = any(i.document.type == "runbook" for i in accepted)
    if order_after != order_before:
        influence = RetrievalInfluence(
            kind="reordered",
            detail=f"promoted {moved} from confirmed historical incident(s)",
            promoted_hypotheses=moved,
            accepted_doc_ids=accepted_ids,
            rejected_doc_ids=rejected_ids,
            order_before=order_before,
            order_after=order_after,
        )
    elif has_runbook:
        influence = RetrievalInfluence(
            kind="suggested_check",
            detail="approved runbook accepted as planning context",
            accepted_doc_ids=accepted_ids,
            rejected_doc_ids=rejected_ids,
            order_before=order_before,
            order_after=order_after,
        )
    else:
        influence = RetrievalInfluence(
            kind="no_effect",
            detail="accepted memory did not change hypothesis order",
            accepted_doc_ids=accepted_ids,
            rejected_doc_ids=rejected_ids,
            order_before=order_before,
            order_after=order_after,
        )
    return updated, influence


def build_memory_evidence(
    items: list[RetrievedItem],
    decisions: list[TrustDecision],
    *,
    investigation_id: str,
) -> list[Evidence]:
    """Record accepted memory as historical evidence (never support/contradiction)."""
    now = datetime.now(UTC)
    evidence: list[Evidence] = []
    for item in _accepted_docs(items, decisions):
        doc = item.document
        source_kind = "retrieved_incident" if doc.type == "incident" else "runbook"
        evidence.append(
            Evidence(
                evidence_id=f"ev_mem_{doc.doc_id}",
                investigation_id=investigation_id,
                task_id="retrieval",
                producing_agent="retriever",
                source_kind=source_kind,  # type: ignore[arg-type]
                source_name=doc.doc_id,
                tool_name=None,
                observed_at=doc.reviewed_at or doc.created_at,
                collected_at=now,
                payload={
                    "title": doc.title,
                    "type": doc.type,
                    "pipeline": doc.pipeline,
                    "dataset": doc.dataset,
                    "category": doc.category.value if doc.category else None,
                    "confirmation_status": doc.confirmation_status,
                    "score": item.score,
                },
                summary=f"historical {doc.type}: {doc.title}",
                supports=[],  # AD-005: memory never establishes a diagnosis
                contradicts=[],
                freshness_status="historical",
                provenance={
                    "phase": "retrieval",
                    "confirmation_status": doc.confirmation_status,
                    "resolution_status": doc.resolution_status,
                    "relevance_score": item.score,
                },
            )
        )
    return evidence

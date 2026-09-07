"""End-to-end retrieval integration over the real corpus (M2 exit criteria, §22).

Demonstrates: useful retrieval changes order; misleading/outdated retrieval is rejected;
memory never counts as proof (AD-005).
"""

from __future__ import annotations

import pytest

from investigator.app import run_investigation
from investigator.reporting.report import validate_grounding

from ..conftest import DIAGNOSED, INCONCLUSIVE

pytestmark = pytest.mark.asyncio


async def test_useful_retrieval_changes_order() -> None:
    state = await run_investigation(DIAGNOSED)
    influence = state["retrieval_influence"]
    assert influence.kind == "reordered"
    assert influence.order_before != influence.order_after
    # A confirmed source_data incident promotes that hypothesis to the front.
    assert influence.order_after[0].endswith("source_data")
    assert "INC-HIST-0007" in influence.accepted_doc_ids


async def test_misleading_and_outdated_retrieval_rejected() -> None:
    state = await run_investigation(DIAGNOSED)
    influence = state["retrieval_influence"]
    # Different-pipeline (payments) and outdated/superseded incidents are rejected.
    assert "INC-HIST-0042" in influence.rejected_doc_ids
    assert "INC-HIST-0003" in influence.rejected_doc_ids


async def test_memory_never_counts_as_proof() -> None:
    state = await run_investigation(DIAGNOSED)
    # Diagnosis is unchanged despite memory promoting a different hypothesis.
    assert state["outcome"] == "diagnosed"
    assert state["leading_hypothesis_id"].endswith("transformation_logic")

    report = state["report"]
    supporting = set(report["supporting_evidence_ids"])
    memory_ids = {
        e.evidence_id for e in state["evidence"] if e.source_kind != "current_operational"
    }
    assert memory_ids  # memory was recorded ...
    assert not (supporting & memory_ids)  # ... but never used as support (AD-005)

    # Historical context is labeled, and grounding still holds for the released report.
    assert report["historical_context"]["effect"] == "reordered"
    grounded, issues = validate_grounding(report, state["evidence"])
    assert grounded, issues


async def test_off_pipeline_memory_rejected_in_other_scenario() -> None:
    # The inconclusive scenario is a different pipeline (returns_daily); no corpus item
    # matches, so retrieval accepts nothing and the outcome is unaffected.
    state = await run_investigation(INCONCLUSIVE)
    influence = state["retrieval_influence"]
    assert influence.kind in {"rejected", "not_run"}
    assert not influence.accepted_doc_ids
    assert state["outcome"] == "inconclusive"

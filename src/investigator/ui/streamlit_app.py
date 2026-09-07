"""Investigation UI (M5.3).

    uv run investigator ui          # or: streamlit run src/investigator/ui/streamlit_app.py

A thin rendering layer. Every derivation lives in :mod:`investigator.ui.views`, which is
plain functions over a persisted run — so what this file does is choose widgets, and the
behaviour it displays is tested without a browser.

The tabs are built around an investigation moving through stages rather than around a
chat transcript, because that is what the system does: a bounded search over competing
hypotheses that sometimes ends by refusing to conclude.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import streamlit as st

# Absolute imports on purpose: Streamlit executes this file as ``__main__``, so it has no
# parent package and relative imports fail at load — invisibly, because the script only
# runs when a session connects.
from investigator.config import load_config
from investigator.ui import loader, views

st.set_page_config(page_title="Data Reliability Investigator", page_icon="🔎", layout="wide")

_TONE_COLOURS = {"ok": "#15803d", "warn": "#b45309", "info": "#1d4ed8"}


# --------------------------------------------------------------------------- #
# Sidebar — pick a run, or make one
# --------------------------------------------------------------------------- #
def _scenarios() -> list[str]:
    scenarios_dir = load_config().scenarios_dir
    return sorted(p.name for p in scenarios_dir.iterdir() if (p / "alert.json").exists())


def _run_live(scenario_id: str) -> None:
    from investigator.app import run_investigation

    with st.spinner(f"Investigating {scenario_id}…"):
        state = asyncio.run(run_investigation(scenario_id))
    st.session_state["live_state"] = state
    st.session_state["selected_run"] = state.get("investigation_id")


def _sidebar() -> loader.RunView | None:
    st.sidebar.title("🔎 Investigator")

    mode = st.sidebar.radio(
        "Mode",
        ["Replay", "Run live"],
        help="Replay reads a persisted run. Run live executes the graph now.",
    )

    if mode == "Run live":
        scenario = st.sidebar.selectbox("Scenario", _scenarios())
        if st.sidebar.button("Run investigation", type="primary", width="stretch"):
            _run_live(scenario)

    runs = loader.list_runs()
    if not runs:
        st.sidebar.warning("No runs yet. Switch to *Run live*, or run the CLI once.")
        return None

    labels = {r.investigation_id: f"{r.label}  ·  {r.investigation_id[-6:]}" for r in runs}
    selected = st.session_state.get("selected_run")
    ids = list(labels)
    index = ids.index(selected) if selected in ids else 0
    chosen = st.sidebar.selectbox("Run", ids, index=index, format_func=lambda i: labels[i])
    st.session_state["selected_run"] = chosen

    live = st.session_state.get("live_state")
    if live is not None and live.get("investigation_id") == chosen:
        # Payloads and provenance only exist in memory; a replay cannot show them.
        return loader.view_from_state(live)
    return loader.load_run(chosen)


# --------------------------------------------------------------------------- #
# Header — the verdict travels with you
# --------------------------------------------------------------------------- #
PROJECT_TITLE = "Agentic Data Reliability Investigator"
PROJECT_SUBTITLE = (
    "Read-only, multi-agent diagnosis of data-reliability incidents — "
    "it names a cause only when the evidence carries one."
)


def _title() -> None:
    """The project's name, on every view. The tabs below are one investigation."""
    st.title(PROJECT_TITLE)
    st.caption(PROJECT_SUBTITLE)


def _header(run: loader.RunView) -> None:
    head = views.header(run)
    colour = _TONE_COLOURS.get(head["tone"], "#334155")
    leading = (run.report.get("leading_hypothesis") or {}).get("category")

    # The diagnosis, stated before any of the detail that supports it.
    diagnosis = leading or head["outcome"] or "no outcome"
    st.markdown(
        f"<div style='font-size:1.35rem; font-weight:600; margin-bottom:2px'>"
        f"<span style='color:{colour}'>{diagnosis}</span>"
        f"<span style='color:#64748b; font-weight:400'> · {head['outcome'] or '—'}</span>"
        f"</div>"
        f"<div style='color:#64748b; margin-bottom:14px'>"
        f"{head['scenario_id'] or 'investigation'} · <code>{head['investigation_id']}</code>"
        f"{' · live run' if head['is_live'] else ''}</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(6)
    columns[0].metric("Confidence", head["confidence"] or "—")
    columns[1].metric("Risk tier", head["risk_tier"] or "—")
    columns[2].metric("Release", head["release_status"] or "—")
    columns[3].metric("Rounds", head["rounds_used"] or 0)
    columns[4].metric(
        "Calls", f"{head['calls_used'] or 0} / {head['calls_budget'] or '—'}"
    )
    columns[5].metric("Escalated", "yes" if head["escalated"] else "no")
    if not run.has_journal:
        st.info(
            "This run predates the run journal, so only the report is available. "
            "Orchestration, beam, memory and critic views need a journalled run."
        )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def _tab_investigation(run: loader.RunView) -> None:
    data = views.investigation(run)
    left, right = st.columns(2)

    with left:
        st.subheader("Alert as received")
        st.json(data["alert"], expanded=False)
        st.subheader("Verification")
        verification = data["verification"]
        st.write(f"**{verification.get('status')}** — "
                 f"{(verification.get('detail') or {}).get('explanation', '')}")

    with right:
        st.subheader("Verdict")
        leading = data["leading_hypothesis"]
        if leading:
            st.success(f"**{leading['category']}** — {leading['statement']}")
        else:
            st.warning(data["stop_reason"] or "no cause named")
        if data["strongest_alternative"]:
            st.caption(
                f"Strongest alternative: {data['strongest_alternative']['category']} "
                f"({data['strongest_alternative']['status']})"
            )
        if data["uncertainty"]:
            st.info(data["uncertainty"])

    st.subheader("Supporting evidence")
    if data["supporting_evidence"]:
        st.dataframe(
            [
                {"tool": e["tool_name"], "summary": e["summary"], "freshness":
                 e["freshness_status"]}
                for e in data["supporting_evidence"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No supporting evidence — the system did not name a cause.")

    if data["escalation_package"]:
        with st.expander("Human escalation package", expanded=True):
            package = data["escalation_package"]
            st.write(f"**Stop reason:** {package.get('stop_reason')}")
            for check in package.get("recommended_next_checks") or []:
                st.write(f"- {check}")
            if package.get("unavailable_sources"):
                st.error("Unavailable sources: " + ", ".join(package["unavailable_sources"]))


def _tab_orchestration(run: loader.RunView) -> None:
    rounds = run.rounds
    round_number = None
    if len(rounds) > 1:
        round_number = st.select_slider(
            "Round", options=rounds, value=rounds[-1],
            help="Highlights the stages that ran in this round.",
        )
    elif rounds:
        round_number = rounds[-1]

    st.graphviz_chart(views.orchestration_dot(run, round_number), width="stretch")

    st.subheader("Stages")
    for stage in views.stage_timeline(run, round_number):
        prefix = "▶" if stage["in_round"] else "·"
        st.markdown(
            f"{prefix} `{stage['sequence']:>2}` **{stage['node']}** "
            f"<span style='color:#64748b'>r{stage['round']} · {stage['duration_ms']} ms</span><br>"
            f"<span style='color:#475569'>{stage['summary']}</span>",
            unsafe_allow_html=True,
        )


def _tab_beam(run: loader.RunView) -> None:
    st.caption(
        "Every branch scored on the FR-703 rubric, with the reason for each point and the "
        "reason it was kept or cut. The rubric is deterministic: a score can be audited "
        "rather than trusted."
    )
    for round_data in views.beam_rounds(run):
        st.subheader(f"Round {round_data['round']} — {round_data['summary']}")
        st.dataframe(
            [
                {
                    "branch": row["branch_id"],
                    "total": row["total"],
                    **{key: row["components"].get(key) for key, _, _ in
                       views.RUBRIC_COMPONENTS},
                    "disposition": row["disposition"],
                    "reason": row["prune_reason"] or ("selected" if row["selected"] else ""),
                }
                for row in round_data["branches"]
            ],
            width="stretch",
            hide_index=True,
        )
        for row in round_data["branches"]:
            if row["reasons"]:
                with st.expander(f"Why {row['branch_id']} scored {row['total']}"):
                    for reason in row["reasons"]:
                        st.write(f"- {reason}")
        for note in round_data["disagreements"]:
            st.warning(note)


def _tab_agents(run: loader.RunView) -> None:
    st.caption(
        "Permissions are enforced in dispatch code, not in prompts. Two of these agents "
        "hold no operational tools at all."
    )
    for card in views.agent_cards(run):
        with st.container(border=True):
            st.markdown(f"**{card['agent']}** · `{card['prompt_version']}`")
            st.caption(card["purpose"])
            columns = st.columns([2, 1])
            if card["holds_no_tools"]:
                columns[0].info("Holds no operational tools")
            else:
                columns[0].write("**Permitted tools:** " + ", ".join(card["permitted_tools"]))
                columns[0].caption("Servers: " + ", ".join(card["servers"]))
            columns[1].metric("Calls made", card["calls_made"])
            if card["decisions"]:
                with st.expander("Decisions this run"):
                    for decision in card["decisions"]:
                        st.write(f"- {decision}")


def _tab_evidence(run: loader.RunView) -> None:
    current_only = st.checkbox("Current operational evidence only", value=False)
    rows = views.evidence_rows(run, current_only=current_only)
    st.caption(
        "Current versus historical is the system's central claim: retrieved memory may "
        "reorder an investigation but never proves one."
    )
    for row in rows:
        border = "#15803d" if row["is_current"] else "#b45309"
        badge = "current" if row["is_current"] else row["source_kind"]
        marks = []
        if row["discriminating"]:
            marks.append("discriminating")
        if row["supports_diagnosis"]:
            marks.append("supports the diagnosis")
        st.markdown(
            f"<div style='border-left:3px solid {border}; padding-left:10px; "
            f"margin-bottom:8px'><b>{row['tool_name'] or row['source_kind']}</b> "
            f"<span style='color:#64748b'>· {badge}"
            + (f" · {', '.join(marks)}" if marks else "")
            + f"</span><br>{row['summary']}</div>",
            unsafe_allow_html=True,
        )
        if row["payload"] is not None:
            with st.expander(f"Raw payload — {row['evidence_id']}"):
                st.json(row["payload"], expanded=False)
                st.caption(f"Provenance: {json.dumps(row['provenance'], default=str)}")
    if rows and rows[0]["payload"] is None:
        st.caption(
            "Raw payloads are available for a live run only — they are deliberately never "
            "written to disk (FR-1208 retention)."
        )


def _tab_memory(run: loader.RunView) -> None:
    memory = views.memory_view(run)
    if not memory["decisions"]:
        st.info("Retrieval did not run for this investigation.")
        return

    st.subheader("Trust gate")
    st.caption(
        "Six gates, all of which must pass. A document can be the most relevant thing in "
        "the corpus and still be refused."
    )
    st.dataframe(
        [
            {
                "document": d["doc_id"],
                "score": round(d["score"], 3) if d["score"] is not None else None,
                **{gate: ("pass" if d["gate_results"].get(gate) else "FAIL")
                   for gate in memory["gates"]},
                "accepted": d["accepted"],
            }
            for d in memory["decisions"]
        ],
        width="stretch",
        hide_index=True,
    )
    for decision in memory["decisions"]:
        if not decision["accepted"]:
            st.markdown(
                f"**{decision['doc_id']}** rejected — {'; '.join(decision['reasons'])}"
            )

    st.subheader("Influence on the investigation")
    influence = memory["influence"]
    st.write(f"**{influence.get('kind', 'unknown')}** — {influence.get('detail', '')}")
    if memory["order_before"]:
        st.write("Hypothesis order before: " + " → ".join(memory["order_before"]))
        st.write("Hypothesis order after: " + " → ".join(memory["order_after"]))
        if not memory["reordered"]:
            st.success("Order unchanged: memory did not influence this investigation.")
    if memory["note"]:
        st.caption(memory["note"])


def _tab_critic(run: loader.RunView) -> None:
    rounds = views.critic_rounds(run)
    if not rounds:
        st.info("The Critic did not review this investigation.")
        return
    st.caption(
        "The Critic is advisory. Deterministic policy owns the outcome, and every "
        "disagreement is recorded rather than silently resolved (FR-1107)."
    )
    for review in rounds:
        with st.container(border=True):
            columns = st.columns(2)
            columns[0].markdown(f"**Round {review['round']}** · `{review['review_id']}`")
            columns[0].write(f"Critic recommends: **{review['recommendation']}**")
            columns[1].write(f"Deterministic outcome: **{review['deterministic_outcome']}**")
            if review["disagrees"]:
                columns[1].warning("Disagreement — recorded, not resolved by the Critic")
            if review["rationale"]:
                st.caption(review["rationale"])
            for label, key in (
                ("Unsupported claims", "unsupported_claims"),
                ("Evidence gaps", "evidence_gaps"),
                ("Recommended prunes", "pruning_recommendations"),
                ("Recommended reopens", "reopening_recommendations"),
            ):
                if review[key]:
                    st.markdown(f"*{label}*")
                    for item in review[key]:
                        st.write(f"- {item}")

    recorded = views.disagreements(run)
    if recorded:
        st.subheader("Recorded disagreements")
        for note in recorded:
            st.write(f"- {note}")


def _tab_traces(run: loader.RunView) -> None:
    data = views.traces(run)
    trace = data["tracing"]
    if trace:
        link = trace.get("url")
        st.success(
            f"LangSmith · project `{trace.get('project')}` · run `{trace.get('run_id')}`"
        )
        if link:
            st.markdown(f"[Open this run in LangSmith]({link})")
    else:
        st.caption(
            "This run was not traced. Set LANGSMITH_TRACING and LANGSMITH_API_KEY to "
            "export traces (M5.2)."
        )

    if data["llm_calls"]:
        st.subheader(
            f"Model calls — {len(data['llm_calls'])}, {data['total_tokens']} tokens"
        )
        st.dataframe(data["llm_calls"], width="stretch", hide_index=True)
    else:
        st.caption("No model calls: this run used the deterministic engine.")

    st.subheader("Stage stream")
    st.dataframe(
        [
            {k: v for k, v in stage.items() if k != "delta_keys"}
            for stage in data["stages"]
        ],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Version attribution (FR-1206)"):
        st.json(data["versions"])


def _tab_safety(run: loader.RunView) -> None:
    data = views.safety(run)
    columns = st.columns(4)
    columns[0].metric("Risk tier", data["risk_tier"] or "—")
    columns[1].metric("Release", data["release_status"] or "—")
    columns[2].metric("Blocked tool attempts", data["blocked_tool_attempts"])
    columns[3].metric("Grounded", "yes" if data["grounded"] else "no")

    if data["risk_reasons"]:
        st.subheader("Why this tier")
        for reason in data["risk_reasons"]:
            st.write(f"- {reason}")

    if data["diagnosis_checks"]:
        st.subheader("FR-900 diagnosis criteria")
        st.dataframe(
            [{"check": k, "holds": v} for k, v in data["diagnosis_checks"].items()],
            width="stretch",
            hide_index=True,
        )

    if data["tool_failures"]:
        st.error("Tool failures: " + "; ".join(data["tool_failures"]))
    if data["reasoning_fallbacks"]:
        st.warning("Degraded to deterministic reasoning: "
                   + "; ".join(data["reasoning_fallbacks"]))

    st.subheader("Acceptance gates (FR-1305)")
    st.caption(
        "Twelve gates over the whole labelled scenario set. None is a model-quality "
        "target: a better model cannot buy a pass, a worse one cannot excuse a failure."
    )
    if st.button("Run the acceptance gates"):
        from investigator.evaluation import evaluate_scenarios

        with st.spinner("Running the labelled scenario set…"):
            report = asyncio.run(evaluate_scenarios())
        st.session_state["acceptance"] = report.as_dict()

    payload: dict[str, Any] | None = st.session_state.get("acceptance")
    if payload:
        gates = views.acceptance_snapshot(payload)
        passed = sum(1 for g in gates if g["passed"])
        (st.success if passed == len(gates) else st.error)(
            f"{passed}/{len(gates)} gates pass · sample size {payload['sample_size']}"
        )
        st.dataframe(gates, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    run = _sidebar()
    _title()
    if run is None:
        st.info(
            "No persisted runs yet. Switch the sidebar to **Run live** and pick a "
            "scenario, or run `investigator investigate --scenario ...` once."
        )
        return

    _header(run)
    tabs = st.tabs(
        [
            "Investigation",
            "Orchestration",
            "Hypotheses & beam",
            "Agents",
            "Evidence",
            "Memory",
            "Critic",
            "Traces",
            "Safety",
        ]
    )
    for tab, render in zip(
        tabs,
        (
            _tab_investigation,
            _tab_orchestration,
            _tab_beam,
            _tab_agents,
            _tab_evidence,
            _tab_memory,
            _tab_critic,
            _tab_traces,
            _tab_safety,
        ),
        strict=True,
    ):
        with tab:
            render(run)


main()

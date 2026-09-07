# Agentic Data Reliability Investigator

A read-only, multi-agent system that diagnoses data-reliability incidents — and refuses to
name a cause the evidence does not carry.

When a nightly pipeline lands 12,000 rows instead of 50,000, someone on call has to work
out why: a filter change, a late upstream feed, a failed task, a quality defect, or nothing
at all. That triage is mechanical, repetitive, and mostly a matter of asking a handful of
systems the right questions in the right order. It is also exactly the kind of work where a
confidently wrong answer is worse than no answer — it sends someone to fix the wrong
pipeline at 3am.

So the design question is not "can a language model guess the cause?" It is **what has to
be true before a diagnosis is allowed out.**

## What it does

Given an alert, the system verifies the symptom against live data, generates competing
hypotheses, dispatches specialist agents to gather evidence through read-only tools, runs
an adversarial critic over what came back, and then reaches one of four outcomes:

| Outcome | When |
|---|---|
| **diagnosed** | Two independent supporting observations from different tools, at least one discriminating, the strongest competitor tested and refuted, no unresolved contradiction |
| **inconclusive** | Any of those unmet — with a handover package naming the open questions |
| **not an incident** | The authoritative current check shows the alert condition is absent |
| **held for review** | A conclusion was reached, but sensitivity, impact, lost tool integrity, or a live objection means a person decides |

The governing principle is that **the model proposes and deterministic code disposes**. A
language model supplies judgment at four points — hypothesis generation, check selection,
observation interpretation, and critic review — and every output is validated against a
closed vocabulary before anything reads it. It cannot name a tool it was not granted, widen
a permission, set a budget, mark evidence as current, or declare an outcome.

That separation is measurable, and it was measured. Running on a real language model,
root-cause accuracy fell to 0.000 while eleven of twelve safety gates still passed and
**every diagnosis the model issued was withheld from release**. See
[Measured results](#measured-results).

## Quickstart

```bash
uv sync --python 3.12                                        # no API key required
uv run investigator investigate --scenario diagnosed_transformation_filter
```

```
Outcome       : diagnosed
Confidence    : moderate
Risk tier     : medium
Release status: released
Leading cause : transformation_logic — A transformation/filter change in orders_daily
                is dropping or altering rows for orders_fact.
Supporting ev : ['ev_T1-H1-..._compare_source_and_target',
                 'ev_T1-H1-..._get_recent_transformation_changes']
Diagnosis chk : {'verified': True, 'two_independent_current_supports': True,
                 'at_least_one_discriminating': True, 'strongest_competitor_weakened': True,
                 'no_critical_current_contradiction': True, 'leading_has_evidence': True}
```

Then try the case where it refuses to answer, and the one where it declines to find a fault:

```bash
uv run investigator investigate --scenario inconclusive_missing_evidence
uv run investigator investigate --scenario not_an_incident_business_volume
```

Everything runs offline against fixture data on a deterministic reasoning engine, so the
whole system — including the test suite — is reproducible with no key and no network.

## Reviewing this project

If you have **ten minutes**, in order:

1. `uv run investigator investigate --scenario diagnosed_transformation_filter` — a
   diagnosis, then `--scenario inconclusive_missing_evidence` — a refusal.
2. `uv run investigator journal --investigation <id> --delta` — the stage-by-stage record of
   how that run moved: every branch score, every prune with its reason, the critic's review
   beside the deterministic outcome.
3. Open [`examples/`](examples/) — four real runs and two full evaluations, committed so
   they can be read without running anything. Start with
   `diagnosis-held-for-review.report.json`: a real model named the wrong cause with high
   confidence, and the release gate withheld it.
4. `uv run investigator evaluate` — six labelled scenarios, twelve acceptance gates.

If you have **an hour**, the code worth reading is:

| File | Why |
|---|---|
| `src/investigator/graph/parent_graph.py` | The orchestration: every node, and the conditional loop back into another round |
| `src/investigator/graph/stopping.py` | The six criteria that must all hold before a cause may be named |
| `src/investigator/graph/risk.py` | The release gate — what may go out on its own and what must reach a human |
| `src/investigator/mcp_client/permissions.py` | Role → tool → server, enforced in dispatch code rather than in a prompt |
| `src/investigator/search/scoring.py` | The branch rubric, itemized so a score can be audited rather than trusted |
| `src/investigator/retrieval/trust_gate.py` | Six gates a remembered incident must clear before it may influence anything |
| `docs/threat-model.md` | Threats by layer, each with the control that answers it and what is still only partial |

## Repository layout

```
src/investigator/
  agents/        four agents: commander, two specialists, evidence critic
  graph/         parent LangGraph, stopping evaluator, risk gate, run journal
  search/        tree-of-thought beam search, branch scoring rubric
  retrieval/     corpus, embedder, FAISS index, trust gate, influence
  evaluation/    labels, metrics, acceptance gates, version attribution, pricing
  governance/    human review, memory promotion, improvement backlog
  observability/ structured logging, monitoring thresholds, retention, tracing
  reporting/     report assembly, grounding validator, escalation package
  ui/            Streamlit interface (view models are plain functions, and tested)
mcp_servers/     two independently runnable read-only MCP servers
data/scenarios/  six labelled scenarios: fixtures plus their expected outcomes
examples/        real reports, journals, and evaluation results
tests/           404 tests across 13 suites
```

## License

MIT — see [LICENSE](LICENSE).

## Development status

The four agents, both tool servers, the retrieval trust gate, beam search, the governance
layer, the evaluation harness, the run journal, tracing, and the UI are all implemented and
tested. Deferred by decision: nine further scenarios, a single-agent baseline, a scripted
demo, and live scheduling for the monitoring and retention policies. Each is written up in
[TODO.md](TODO.md) with the design work already done, and the honest limitations are in
[Stated limitations](#stated-limitations).

---

## Requirements

- Python **3.12** (fetched automatically by `uv`)
- [`uv`](https://docs.astral.sh/uv/) (tested with uv 0.10)
- macOS or Linux (NFR-007)

No API keys or network access are required **by default**: the shipped configuration uses
the deterministic reasoning engine and an offline hashing embedder + FAISS, so the whole
suite runs reproducibly with no key (NFR-001, FR-1304). Point `reasoning.engine` at `llm`
and set `OPENROUTER_API_KEY` to run on a real model — see *Reasoning engines* below.

---

## Installation

```bash
uv sync --python 3.12            # core: CLI, MCP servers, evaluation harness
uv sync --extra ui               # adds Streamlit for the investigation UI
```

This creates a `.venv`, installs runtime + dev dependencies, and writes `uv.lock`
(NFR-001 reproducibility). The UI is an optional extra: the CLI, the harness, and the
whole test suite run without it. Copy the env template if you want to customise anything:

```bash
cp .env.example .env
```

---

## Running the MCP servers standalone

Each server is **separately runnable** over stdio (FR-509). Which scenario a server
serves is chosen via the `INVESTIGATOR_SCENARIO` environment variable:

```bash
# Pipeline Operations MCP (reads data/scenarios/<scenario>/pipeline_operations.json)
INVESTIGATOR_SCENARIO=diagnosed_transformation_filter \
  uv run python -m mcp_servers.pipeline_operations.server

# Data Observability MCP (reads data/scenarios/<scenario>/data_observability.json)
INVESTIGATOR_SCENARIO=diagnosed_transformation_filter \
  uv run python -m mcp_servers.data_observability.server
```

A standalone server waits for an MCP client on stdio; the application launches both
automatically (below), so you normally don't run them by hand.

---

## Running an investigation (CLI)

```bash
# List available scenarios
uv run investigator list-scenarios

# Diagnosed vertical slice (transformation-filter regression) -> outcome: diagnosed
uv run investigator investigate --scenario diagnosed_transformation_filter

# Insufficient evidence -> outcome: inconclusive + human-escalation package
uv run investigator investigate --scenario inconclusive_missing_evidence

# Full report JSON
uv run investigator investigate --scenario diagnosed_transformation_filter --json
```

Reports are also persisted to `outputs/<investigation_id>.json`. Structured JSON logs
(FR-1200) are emitted to **stderr**.

Equivalent module form: `uv run python -m investigator.app investigate --scenario ...`.

---

## Running the tests

```bash
uv run pytest -q            # full suite (404 tests)
uv run pytest tests/unit -q
uv run pytest tests/contract -q
uv run pytest tests/search -q       # M3: thoughts, scoring, beam policy, Critic
uv run pytest tests/llm -q          # LLM adapter (stubbed transport, no key needed)
uv run pytest tests/retrieval -q    # M2: embedder, trust gate, influence, metrics
uv run pytest tests/integration -q
uv run pytest tests/scenarios -q     # M4.3: the six-scenario FR-1301 corpus
uv run pytest tests/governance -q    # M4.1: reviews, promotion, monitoring, retention
uv run pytest tests/evaluation -q    # M4.2: metrics, acceptance gates, version drift
uv run pytest tests/journal -q       # M5.1: the run journal
uv run pytest tests/observability -q # M5.2: tracing stays off unless asked
uv run pytest tests/ui -q            # M5.3: view models, and the app actually renders
uv run pytest tests/safety -q

uv run ruff check .         # lint
uv run mypy src mcp_servers # type check
```

The contract/integration/scenario/safety suites launch the real MCP servers as stdio
subprocesses against fixtures — no mocks, no network.

**The suite cannot reach a model provider**, and that is enforced rather than assumed. An
autouse fixture in `tests/conftest.py` pins every test to the deterministic engine and
removes provider keys from the environment for the duration of each test. It matters
because `load_dotenv()` runs at import: a real key on the machine is already in
`os.environ` before any test starts, so a single `engine: llm` in committed config would
turn every scenario, journal and UI test into a billed API call without a line of test
code changing. The LLM adapter tests are unaffected — they pass a literal key through an
`httpx` `MockTransport` that never opens a socket. Select a real provider per run instead,
with `--provider` / `--engine`.

---

## Architecture

```
CLI (investigator.app)
  -> Parent LangGraph (graph/parent_graph.py)  [LangGraph is the only orchestrator]
       initialize -> parse_alert -> verify_incident (via MCP)
         -> commander_generate_hypotheses (+ open one branch per hypothesis)
         -> retrieve_context -> apply_retrieval_trust_gate
         -> commander_plan_round -> dispatch_specialists (parallel) -> merge
         -> critic_review -> apply_branch_policy (beam) -> evaluate_stop (deterministic)
              |                                                   |
              +---- continue: back to commander_plan_round -------+
         -> compose_report -> validate_grounding -> classify_risk (FR-1105/1106 gate)
         -> prepare_human_escalation -> persist
  Specialists (FR-802 subgraphs): Pipeline Investigator, Data Investigator
       -> PermissionedToolProxy (role-based, FR-506)
            -> InvestigatorMCPClient (discovery + dispatch, stdio)
                 -> pipeline-operations-mcp   / data-observability-mcp
                      -> provider protocol -> fixture provider -> scenario JSON
```

Authority boundaries (Appendix C): LangGraph owns orchestration/state; MCP owns
operational access; providers own data access; **deterministic code owns validation,
permissions, budgets, and outcomes** (AD-004, FR-903). Agents never import fixtures or
providers (AD-003) — enforced by a static test.

### Key modules

| Area | Path |
|---|---|
| Contracts (FR-300..307, 400, 503) | `src/investigator/schemas/` |
| Parent graph / routing / reducers / stopping / verification / risk | `src/investigator/graph/` |
| Agents + deterministic reasoning | `src/investigator/agents/` |
| MCP client + permissions + error taxonomy | `src/investigator/mcp_client/` |
| Reporting + grounding + escalation | `src/investigator/reporting/` |
| Retrieval: embedder / FAISS index / corpus / trust gate / influence / eval | `src/investigator/retrieval/` |
| MCP servers + provider protocols + fixtures | `mcp_servers/` |
| Scenarios / incident + runbook corpus | `data/scenarios/`, `data/incidents/`, `data/runbooks/` |

---

## Architecture Decision Record

Per coding rule 11, resolved ambiguities and deviations are recorded here.

### Resolved ambiguities

1. **Verification tool mapping (FR-803).** The spec says "use a suitable MCP tool" but
   does not map symptom→tool. `graph/verification.py` defines a deterministic map
   (e.g. `volume_drop`→`get_table_metrics`, `pipeline_failure`→`get_pipeline_run_status`,
   `schema_change`→`get_schema_changes`) and a deterministic observed-vs-expected
   comparison with a configurable tolerance (`config/default.yaml`).
2. **Active scenario selection.** Providers must know which fixtures to serve. The
   scenario id is injected into each MCP server subprocess via `INVESTIGATOR_SCENARIO`,
   keeping tool inputs clean and honouring FR-505 (provider owns data access).
3. **`scenario_id` in typed results (FR-503).** Sourced from that env var.
4. **Verification executor.** The Commander has no tools (FR-506), so the deterministic
   controller performs verification through the appropriate **specialist-scoped proxy**.
5. **Transport.** `stdio` (§10.1 permits it); the launch config is the only thing that
   changes for Streamable HTTP later.

### Deviations from the full spec (by design, per the milestone plan)

- **`observed_at == collected_at`.** Fixtures represent current observations captured at
  read time, so evidence freshness is `current`.
- **Operational budget accounting.** Verification calls are tracked separately from the
  FR-702 operational-call budget (which governs the investigation/beam phase).
- **Deterministic reasoning is the default, not the only option.** Hypothesis generation
  and observation interpretation run on a real model when `reasoning.engine: llm` (see
  *Reasoning engines*); the deterministic engine remains the default so tests and
  reproducibility runs need no key (AD-004, FR-1304). Branch scoring and critic review are
  still rule-based where they are exact set arithmetic; the Critic's *judgment* half runs
  on the model (see *Reasoning engines*).
- **Milestone 5 items** (the run journal, LangSmith tracing, and the Streamlit UI) remain
  unimplemented. The operational safety dashboard FR-1204 describes is M5: M4 delivers its
  data — acceptance gates, monitoring verdicts, the scenario matrix, and the review queue —
  through the CLI and the evaluation report rather than through a screen.
- **The single-agent baseline (FR-1303) is not measured** — a stated limitation, not an
  omission. See *Stated limitations*.

### Non-goals still honoured (M1–M4)

No web UI, no production connectors, no autonomous remediation, no hidden
chain-of-thought persistence, and **no CrewAI** (AD-002). All tools are read-only
(FR-504); no mutating tool exists. Thoughts are structured contracts with typed fields
(FR-700), never free-form reasoning traces.

---

## v2.1 safety additions (risk-based autonomy) — the M1–M3 foundation

Spec v2.1 added a safety/governance layer (AD-007/008, FR-1104..1109, FR-1204..1208,
FR-1306..1312, NFR-008/9/10). The spec's own milestone map (§22) assigns almost all of
it to **Milestone 4**. Per the agreed scope, M1–M3 implement only the **foundational
deterministic pieces that touch the existing report-release path**:

- **FR-1105 — Deterministic risk classification** (`src/investigator/graph/risk.py`).
  Before a diagnosis/report is released, deterministic policy classifies the
  investigation `low | medium | high | prohibited` from severity, data sensitivity,
  customer/financial/privacy/compliance impact, action reversibility (ADRI is read-only
  → reversible), evidence sufficiency, unresolved contradictions, novelty, and tool
  integrity. **Model confidence is never the sole signal** (AD-007). Release policy:
  `low`→release; `medium`→release when the additional-validation condition holds (a
  `diagnosed` outcome already satisfies FR-900's independent-support + competitor
  refutation); `high`→hold the diagnosis and prepare a human-review escalation even when
  evidence is sufficient; `prohibited`→block the output and fail closed.
- **FR-1106 — Runtime enforcement + blocked-attempt measurement.** Permission, schema,
  and budget checks already run before every MCP call; v2.1 adds (a) the risk-tier
  **release gate** in the parent graph (`classify_risk` node, run after grounding), and
  (b) counting/logging of **blocked tool attempts** (`InvestigatorMCPClient.blocked_attempts`),
  surfaced in the report's guardrail events — measured even when no unsafe action occurs.
- **AD-007 / AD-008 reflected**: autonomy is governed by deterministic policy, not
  prompts; guardrails fail closed; enforcement is code, not prompt text.

Risk policy inputs (data sensitivity / impact flags) are supplied by the controller via
each scenario's `meta.json` `risk` block (or `run_investigation(..., risk_overrides=...)`
in tests) — they are policy inputs, **not** operational evidence, and agents never read
them. Read-only, low-impact investigations therefore stay releasable: the diagnosed
scenario (severity `high`, non-sensitive) classifies **medium** and releases; flagging it
`high_impact` reclassifies it **high** and holds it for human review.

### v2.1 items, and where they landed

Everything the v2.1 layer added is now implemented. FR-1104 (threat model), FR-1107
(disagreement routing), FR-1108 (human-decision capture), FR-1109 (memory promotion),
FR-1204..1208 (monitoring, drift, closed-loop, retention), FR-1302 (expanded metrics),
and FR-1306..1312 (holistic, robustness, confidence-band, repeated, evaluator-independent,
severity-weighted, fallback evaluation) all shipped in M4 and are described below.
NFR-008/009/010 are satisfied by the deterministic layers from M1–M3 together with the
governance layer from M4.1.

Two items are deliberately not closed here: **FR-1303** (single-agent baseline) is a
stated limitation, and **FR-1301 scenarios 7–15** are deferred to a future release — see
*Scenario corpus*, *Stated limitations*, and [TODO.md](TODO.md).

**FR-1107 is complete.** M3 gave deterministic control of hard constraints, the Critic's
advisory role on judgment, and recording of every disagreement. M4 added the routing: a
*material* disagreement — the controller about to declare `diagnosed` over a standing
objection — buys one additional discriminating check when the budget allows, and if the
objection survives, the release gate hands it to a human rather than forcing the
diagnosis.

---

## Reasoning engines (AD-004)

Judgment and policy are separate concerns. **Judgment** — what hypotheses are worth
testing, what an observation means — comes from a `ReasoningEngine`. **Policy** —
permissions, budgets, branch pruning, stopping, grounding, risk gating — is deterministic
code in every configuration. Swapping engines changes the first and nothing else.

| Engine | When | Needs |
|---|---|---|
| `deterministic` | tests, reproducibility runs, offline demos | nothing |
| `llm` | real deployments and live demos — all four agents | a provider key |

| Provider | Endpoint | Key | Notes |
|---|---|---|---|
| `openai` | `api.openai.com/v1` | `OPENAI_API_KEY` | paid, reliable — preferred for a live demo |
| `openrouter` | `openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | has a free tier that rate-limits hard |

Both speak the same chat-completions API, so only the endpoint, key, and courtesy headers
differ. Setting `base_url` points at any other OpenAI-compatible endpoint — a local proxy,
another vendor — with no code change.

> OpenRouter also *serves* OpenAI models (`openai/gpt-4o-mini`), but calling them through
> OpenRouter spends **OpenRouter** credit, not OpenAI credit, unless you configure
> OpenRouter BYOK. Go direct with `--provider openai` to spend an OpenAI balance.

```bash
# default: offline rules
uv run investigator investigate --scenario diagnosed_transformation_filter

# a real model on OpenAI
export OPENAI_API_KEY=sk-...
uv run investigator investigate --scenario diagnosed_transformation_filter \
  --provider openai --model gpt-4o-mini

# OpenRouter's free tier instead
export OPENROUTER_API_KEY=sk-or-v1-...
uv run investigator investigate --scenario diagnosed_transformation_filter \
  --provider openrouter

# what the configured key can actually call
uv run investigator models --provider openai
uv run investigator models --provider openrouter --free
```

Persist a choice in `config/default.yaml` under `reasoning:`, or set
`INVESTIGATOR_REASONING_ENGINE` / `INVESTIGATOR_LLM_PROVIDER` / `INVESTIGATOR_LLM_MODEL` /
`INVESTIGATOR_LLM_BASE_URL` in `.env`. Leaving `model` and `base_url` blank takes the
provider's defaults, so switching provider is one word.

> **Model IDs go stale.** OpenRouter's free roster in particular appears and disappears
> without notice — a committed default already went stale once during development. Any
> `investigator models` output is queried live rather than hard-coded, so trust it over
> the config file.

### What each agent decides

All four components are model-backed under `--engine llm`. Every step the spec describes
as an agent decision goes to the model:

| Agent | Decides | Tools |
|---|---|---|
| Incident Commander | hypotheses (FR-100.2); **which checks to assign** (FR-100.5) | none — coordinates only |
| Pipeline Investigator | **which check to run next, or stop early** (FR-110); what each observation means | 5 pipeline-operations tools |
| Data Investigator | same (FR-120) | 5 data-observability tools |
| Evidence Critic | bias challenges, evidence gaps, stop recommendation (FR-130) | none — reviews only |

The Commander is offered the target specialist's **whole permitted tool set**, not a
hard-coded plan, so it can pick the check it thinks is most discriminating. A specialist
picks its next check from what the Commander assigned, and may return `null` to **stop
with budget unspent** — an outcome the deterministic engine cannot produce, since it
always walks its whole plan.

### The two things a model may not move

- **Which specialist handles a hypothesis.** The role selects the permission set, so it is
  a permission boundary (FR-506), not a judgment call. Tool choice *within* that set is the
  model's; the set itself is not.
- **Tool arguments.** Built from validated alert fields. Letting a model shape MCP inputs
  would turn every alert into an injection surface (FR-1101) for no benefit — the mapping
  is mechanical.

### Why a bad model cannot produce a bad diagnosis

Every field crossing back out of an adapter is validated against a **closed vocabulary**
before anything else sees it:

- categories outside the FR-400 taxonomy are discarded; repeated ones are collapsed;
- **hypothesis IDs and ranks are generated in code**, never taken from the model — they
  key branch, task, and evidence IDs across the whole graph;
- a selected tool outside the role's permissions, or outside the task's allow-list, is
  dropped — and the permissioned MCP proxy refuses it again at dispatch and counts the
  blocked attempt (FR-1106);
- a category the model marks as both supported *and* contradicted is dropped from both —
  an incoherent signal is no signal;
- an interpretation with no supports or contradicts can never be `discriminating`;
- the Critic may make a review **more** cautious but never less: it can move `diagnosed`
  to `continue`, and cannot move `continue` to `diagnosed`;
- statements, summaries, and list lengths are bounded; unknown keys are ignored.

`tests/llm/` includes the worst case: a model whose output is fully controlled by injected
text. It gains nothing — it cannot reach a tool it was not granted, widen a permission, set
a budget, or declare an outcome, because the deterministic stopping evaluator, grounding
validator, and risk gate still hold final authority (FR-903).

### The Critic is deliberately hybrid

Detection stays rule-based; judgment goes to the model. Finding a dangling evidence
reference, a missing `request_id`, or an observation reused as support for two hypotheses
is exact set arithmetic — rules do it perfectly and identically every run, and asking a
model to re-derive them would only add a way to miss one. Whether a lead is *earned* or
merely first, and whether a competitor was defeated or just ignored, does not reduce to a
rule. The model's findings are merged with the deterministic ones, never substituted for
them.

### Failure behaviour

`reasoning.on_failure` decides what a provider outage means. `fallback` (default) degrades
that step to deterministic rules and **records it** — `reasoning_fallbacks` appears in
state, in the report's guardrail events, and in the CLI summary, so a run is never quietly
half-modelled. `fail_closed` raises instead, so the investigation abstains rather than
reasoning from nothing. Rate limits (429) and 5xx are retried with backoff first, which
matters because free models throttle aggressively.

### Observability

Each completion emits an `llm_call` trace event carrying model, latency, token counts, and
attempt count — the FR-1200 fields a deterministic engine structurally cannot populate.

---

## Milestone 2 — Retrieval and Trust (§11)

Retrieval runs between hypothesis generation and task planning, so accepted memory can
change investigation order without ever changing the diagnosis (AD-005).

- **Corpus (FR-600/601)** — human-confirmed incidents (`data/incidents/*.json`) and
  approved runbooks (`data/runbooks/*.json`) with full metadata (type, pipeline,
  dataset, category, environment, version, resolution/confirmation status, timestamps).
  Current metrics/logs/run state are never represented here.
- **Embeddings + FAISS (`src/investigator/retrieval/`)** — a deterministic, offline
  hashing embedder (`embedder.py`) behind an `Embedder` protocol, indexed by FAISS
  (`index.py`, exact cosine via `IndexFlatIP`; a NumPy fallback implements the same
  interface). Reproducible with no model download or API key; a real embedding model can
  be swapped in later.
- **Routing (FR-602)** — retrieve only when prior memory could improve prioritization.
- **Trust gate (FR-603, `trust_gate.py`)** — an item may influence planning only if it
  clears every gate: relevance ≥ threshold, permitted type, **confirmed/approved**
  (unconfirmed/superseded are never precedent — FR-1109), pipeline/dataset/environment
  match, not stale/expired, and no conflict with current evidence. Fails closed.
- **Influence (FR-604, `influence.py`)** — accepted incidents may reorder hypotheses
  (recorded as `reordered` / `suggested_check` / `no_effect` / `rejected` with
  before/after order); accepted memory is recorded as `retrieved_incident`/`runbook`
  evidence with **empty supports/contradicts** and `historical` freshness, so it is
  traceable but never counts toward diagnosis. The grounding validator and the stopping
  evaluator both independently ignore non-current evidence.
- **Evaluation (FR-605, `evaluation.py`)** — precision@k, recall@k, and rejection
  accuracy over a labeled eval set covering clearly-relevant, partially-relevant,
  keyword-similar-irrelevant, outdated, same-symptom/different-pipeline, and unconfirmed
  cases (`tests/retrieval/`).

**M2 exit demonstrated end-to-end** by the `diagnosed_transformation_filter` scenario: a
confirmed historical `source_data` incident **reorders** hypotheses (source_data promoted
first); the different-pipeline (`payments_daily`) and outdated/superseded incidents are
**rejected**; and the outcome stays `diagnosed: transformation_logic` on current evidence
alone — **memory never counts as proof**. Retrieval config lives under `retrieval:` in
`config/default.yaml`.

> **FR-1109 (memory promotion).** M2 implements the *retrieval-side* control — only
> confirmed/approved, non-superseded documents are usable as precedent. The write-side
> promotion workflow + human-decision capture (FR-1108) remain Milestone 4.

## Milestone 3 — Multi-Agent Coordination and Tree Search (§12, §13)

Milestone 3 closes the investigation loop. Where M1 ran a single round and M2 decided
what to look at first, M3 decides what to keep looking at — and when to stop.

### Evidence Critic (FR-130, `src/investigator/agents/evidence_critic/`)

A fourth agent under §6.1: its own role prompt (`config/prompts/evidence_critic.md`),
bounded input/output contracts (investigation state in, one `CriticReview` out), **no
operational MCP tools**, its own subgraph and local working state, and explicit
completion and failure conditions. It is adversarial by design and advisory by contract:

- flags **unsupported claims** — findings citing evidence that does not exist, asserting
  support with no current observation behind it, or reporting support from a failed task;
- flags **missing provenance** on current observations;
- detects **contradictions** and **evidence reuse** (one observation propping up two
  hypotheses is not independent corroboration);
- assesses **independence** — repeated readings from one tool are one source, not two;
- **scores every branch** with the same FR-703 rubric the controller uses;
- identifies the **strongest competitor**;
- recommends prune / continue / reopen / diagnose / abstain;
- **challenges confirmation bias** — an untested competitor is an untested one, not a
  defeated one, and a lead that owes its position to retrieved memory is called out.

Its recommendations carry no authority. `apply_branch_policy` logs
`critic_prunes_overruled` whenever the deterministic policy disagrees (FR-903).

### Tree-of-Thought beam search (`src/investigator/search/`)

- **Thoughts (FR-700, `thought.py`)** — a typed `Thought` carries the hypothesis, the
  proposed action, the expected supportive and weakening observations, a discriminating
  value, and an estimated cost. The contract forbids extra fields, so no unstructured
  reasoning can be persisted. Candidate actions are filtered through the *same*
  permission check the Commander applies, so the search never proposes an action the
  planner would have to drop.
- **Scoring (FR-703, `scoring.py`)** — consistency with current evidence (0-4),
  discriminating value of the next action (0-2), independence and quality of support
  (0-2), applicable retrieved context (0-1), cost efficiency (0-1), and a −3 penalty for
  a critical current contradiction. Every point carries a reason string, so a score is
  auditable rather than merely numeric. Retrieved context is capped at one point and
  contributes nothing to consistency or independence (AD-005).
- **Policy (FR-702/704/705/706, `beam.py`)** — beam width 3, max depth 4, two candidate
  actions per branch. Branches are pruned for falsification, duplicate tested paths,
  fabricated evidence, exhausted budget, excess depth, or a score below 5; a branch that
  simply runs out of permitted actions is **closed**, not pruned — exhaustion is not
  failure, and the record distinguishes them. The **diversity rule** holds at least two
  hypotheses alive until each has had a discriminating check, and never rescues a
  falsified one. **Reopening** revives a pruned branch when new current evidence supports
  it or when every survivor is contradicted, recording both the trigger and the prior
  prune reason — but never once the call budget is gone.

### Parallel dispatch (FR-203, FR-204)

`dispatch_specialists` admits tasks in priority order *before* any concurrency, giving
each a private slice of the remaining global budget, so the parallel group can never
overspend it and no two coroutines race on a shared counter. Results merge by admission
order, never completion order. A specialist that raises is isolated: the failure is
recorded, the task is marked failed, every other specialist's evidence survives, and the
round continues.

### Multi-round continuation (FR-801)

`evaluate_stop` is now a conditional router. Another round runs only when the outcome is
unresolved, the incident is verified, rounds and calls remain, **and a selected branch
still has an untried candidate action**. That last condition is the real termination
condition — the round counter is a backstop. Every continue/stop decision is logged with
the full reason set.

### Checkpointing (FR-804, `src/investigator/persistence/`)

`build_parent_graph(checkpointer=...)` attaches an `AsyncSqliteSaver`, which checkpoints
after every node — a superset of the FR-804 points, asserted against the compiled graph
in `tests/integration/test_checkpointing.py`. Resume safety comes from the reducers, not
the saver: `dedupe_by` is first-write-wins on `evidence_id`, so replaying a checkpoint
cannot duplicate accepted evidence (FR-205). `interrupt_before` exposes the pause seam
that a human-in-the-loop gate will use in M4.

```bash
uv run investigator investigate --scenario diagnosed_transformation_filter --checkpoint
uv run investigator investigate --scenario inconclusive_missing_evidence --thread-id inc-1
```

**M3 exit demonstrated end-to-end.** In `diagnosed_transformation_filter` three branches
are investigated in parallel; reconciliation **falsifies** the source-data branch, the
orchestration branch is **pruned** at score 2, the untested data-quality branch is
**held by the diversity rule** (overruling the Critic's prune recommendation), and the
run still returns `diagnosed: transformation_logic` in one round without spending the
budget. In `inconclusive_missing_evidence` the loop **runs a second round** against the
branches the beam selected, exhausts the call budget, and abstains with an escalation
package that still lists the open checks.

---

## Milestone 4 — Guardrails, Evaluation, and Scenarios (§17, §18)

Three parts: **M4.1** the governance layer that decides what the system may release on
its own and what it must hand to a person, **M4.2** the evaluation harness that says
whether any of it works, and **M4.3** the scenario corpus both are measured against.

### Threat model (FR-1104, NFR-008)

[`docs/threat-model.md`](docs/threat-model.md) maps threats across the seven MAESTRO
layers to the control that answers each, and marks what is still planned. The test applied
throughout: **if the model were fully adversarial, what would still hold?** A control that
fails that test is not counted as one — which is why prompt instructions appear in the
document but never as the only layer.

### Critic disagreement (FR-1107)

Deterministic policy owns hard constraints; the Critic is advisory on judgment. Every
conflict is recorded. A conflict is *material* only in the dangerous direction — the
controller about to declare `diagnosed` over a standing objection. That case buys one
additional discriminating check if the budget allows, and if the objection survives, the
release gate routes it to human review rather than forcing the diagnosis.

The reverse — the Critic wanting more work while the controller already abstains — is the
system being appropriately conservative and is recorded without escalating. Treating it as
material would page a human on every budget-limited run.

### Human review (FR-1108) and memory promotion (FR-1109, NFR-010)

An escalation opens a `pending` review record **at escalation time**, so an unanswered
escalation is visible rather than absent — otherwise the FR-1305 completeness metric reads
100% on an empty queue.

```bash
uv run investigator reviews                       # queue + disposition completeness
uv run investigator review --investigation inv_... \
  --reviewer alice@example.com --role data-platform-oncall \
  --decision confirmed --root-cause source_data \
  --accept "upstream shortfall" --remediation "backfilled from source"
uv run investigator promote --investigation inv_...   # into the retrieval corpus
```

Promotion is gated on the **human**, not on the machine. A diagnosis the system reached on
its own is never precedent, however confident; a human-confirmed, attributed cause is —
**including on an investigation the system abstained from**. That case is deliberately
allowed and is the most valuable memory the system can acquire: the incident it could not
solve, with the answer a person eventually supplied. Refusing it would mean the system only
ever learns what it already knew.

Every promoted document records its reviewer, and `supersede()` retires one without
deleting the audit trail — the M2 trust gate rejects superseded documents immediately.

### Monitoring (FR-1205) and retention (FR-1208)

Every threshold declares a warning level, a critical level, an **owner**, and a
**response**; a number nobody owns is not monitoring. Thresholds are typed `policy` or
`quality`, because a latency regression is worth an alert while a grounding failure is
worth halting: a critical *policy* breach sets `release_permitted` false and diagnoses stop
going out.

Retention is per record class — payloads and prompts 30 days, alerts and traces 90,
reviewer data and evaluations 365, so the audit trail outlives the traces it justifies.
Expiry **redacts content while preserving identifiers and evidence references**, so a
report stays checkable after its payloads are gone rather than leaving dangling references.

---

## Evaluation harness (M4.2)

```bash
uv run investigator evaluate                    # metrics + FR-1305 gates; exit 1 on breach
uv run investigator evaluate --repeat 3         # stability across runs (FR-1309)
uv run investigator evaluate --provider openai --repeat 3 --out evaluations/llm-sweep.json
uv run investigator backlog                     # closed-loop improvement items (FR-1207)
```

**Not all errors are equal.** FR-1302 requires false-confident diagnosis and missed
escalation to be penalised above correct abstention, and `ERROR_WEIGHTS` says so
explicitly: a system that abstains too often is annoying, while one that confidently names
the wrong cause sends someone to fix the wrong pipeline at 3am. The headline
`weighted_error` reflects that ordering and is severity-weighted on top (FR-1311), so one
critical miss outweighs several low-severity ones. An accuracy number alone would hide
both.

**Ground truth comes only from labels** authored alongside each fixture, never from a run
— which makes FR-1310 evaluator independence structural rather than a promise. A test pins
it: the same run state scored against two different labels yields two different verdicts.

**Twelve acceptance gates (FR-1305)** run as CI assertions, not as a report table. None of
them is a model-quality target — a better model cannot buy a pass on grounding,
permissions, injection resistance, or resume safety, and a worse one cannot excuse a
failure. An *unmeasured* gate is reported as skipped and fails the build, because silence
is exactly the failure mode these thresholds exist to catch.

Three of those gates used to read a constant, which is the same failure wearing a
different hat — a gate that cannot fail is not a gate. They are now measured from the
runs themselves:

- **`retrieval_never_proves`** compares the delivered report's supporting evidence ids
  against its historical ones. Memory appearing among a diagnosis's supports is what a
  breach would look like, and that is what is now counted (AD-005).
- **`prompt_injection`** scores a scenario whose label declares what the injected text
  *demanded*; an injection counts as successful when the run adopts that answer. The
  scoring is in place and unit-tested, but no scenario exercises it yet, so this gate
  still reports on an empty set — honest, and weaker than the other two until FR-1301 #9
  lands.
- **`high_risk_routing`** asks whether the cases the *labels* call high-risk reached a
  human, by escalation or by a withheld release. It deliberately does not key off
  incident severity: a high-severity incident on non-sensitive, low-impact data is a
  medium-risk case under FR-1105, and holding it for a human would be exactly the
  over-escalation FR-1302 penalises. Keying it off the tier the *run* assigned itself
  would be worse still — the system would grade its own homework (FR-1310).

Two distinct review metrics, deliberately not merged: **queue drain** (how many
escalations anyone has answered — a workload signal that legitimately sits below 100%) and
**record completeness** (whether answered reviews carry their required fields — which must
be 100%). Conflating them would fail the build every time someone had not yet worked the
queue.

**Version attribution (FR-1206)** fingerprints engine, provider, model, prompt text,
retrieval corpus, tool permissions, and deterministic policy — content-hashed, so an edit
nobody remembered to version still shows up as drift. `diff()` names what moved between
two runs.

### Measured results
<a id="measured-results"></a>

The suite was run three times on each engine — six scenarios, 18 investigations per engine.
Raw output is in `evaluations/m5-verification.json` (deterministic) and
`evaluations/llm-sweep.json` (gpt-4o-mini at temperature 0).

| Measure | deterministic | gpt-4o-mini |
|---|---:|---:|
| Root-cause accuracy | 1.000 | **0.000** |
| Correct abstention | 1.000 | 1.000 |
| False-confident diagnosis rate | 0.000 | **0.667** |
| Hypothesis coverage | 1.000 | 0.500 |
| Escalation recall / rate | 1.000 / 0.333 | 1.000 / 0.833 |
| Grounding failures · unauthorized calls | 0 · 0 | 0 · 0 |
| Weighted error (severity-adjusted) | 0.0 | **118.0** |
| Mean operational calls (budget 8) | 5.3 | 6.0 |
| p95 latency | 2.3 s | 20.4 s |
| Tokens per investigation | 0 | 10,874 (16.5 completions) |
| FR-1305 acceptance gates | **12 / 12** | **11 / 12** |

**Model accuracy collapsed; every safety property held.** The single failing gate is
`happy_path_outcomes` — the one that measures whether the labelled diagnoses were reached.
Grounding, unauthorized calls, mutation attempts, retrieval-never-proves, injection,
critical contradictions, high-risk routing and review completeness all still pass. And
**every diagnosis the model issued was withheld from release**: all three went to
`human_review_required`, driven by unresolved Critic objections. That is the FR-1105 /
FR-1107 path firing against a real model rather than a fixture, and it is the strongest
evidence in this repository that the deterministic layer does something.

**The model collapsed onto one category**, naming `data_quality` in five of six runs
whatever the actual cause. The divergence starts in one interpretation: shown the same
`get_table_metrics` observation, the deterministic rule treats a row-count shortfall as
neutral for causation — verification has already established the symptom — while the model
read it as support *for* `data_quality` and *against* `transformation_logic`. That handed
`data_quality` a third independent supporting tool, which both won the ranking and crossed
the threshold from `moderate` to `high` confidence. One interpretation flipped the answer
and its confidence together.

**Calibration is the sharpest finding.** The `high` band: three claims, none correct, two
of them false-confident diagnoses. The band that should be most trustworthy was the least,
which is invisible in an accuracy average and is exactly why FR-1308 reports per band.

Three numbers need care before they are quoted:

- **`root_cause_accuracy` of 0.000 overstates it.** The model named the correct cause in
  *one of four* diagnosable scenarios; the metric excludes that run because it requires an
  otherwise error-free run and that one also over-escalated. Quote both figures.
- **Over-escalation is not missed escalation.** Escalation rose to 0.833 with recall 1.000
  and zero missed escalations — the weight-1 error, not the weight-8 one. That is the
  direction the design intends, and still a usability cost.
- **Two passing gates are vacuous, and cost is unpriced.** Injection successes are zero
  across *zero* injection scenarios; fallback success is zero across *zero* degraded runs.
  Cost reports as unpriced rather than zero because the provider returned
  `gpt-4o-mini-2024-07-18` and no rate was configured — tokens were measured, cost honestly
  was not.

Spread across the three repeats was zero on every tracked metric for both engines. For the
deterministic engine that is structural; for the model it reflects temperature 0 and
identical inputs, so these figures are reproducible rather than stable under real variation.

**What this does not tell you.** One model at one temperature with one prompt set, six
scenarios, fixture-backed providers, and no single-agent baseline (FR-1303). The latency
and call figures describe orchestration overhead, not production providers. And a *held*
wrong answer still scores as a false-confident diagnosis: the error weighting does not
discount an error the release gate caught, even though a withheld wrong answer is far less
harmful than a released one. Worth changing — flagged here rather than quietly changed.

---

## Scenario corpus (M4.3)

Six labeled scenarios, chosen so that each one fails differently. They are built once and
used twice: as the evaluation harness's inputs, and as the material the M5 UI replays.

| # | Scenario | Outcome | What it proves |
|---|---|---|---|
| 1 | `diagnosed_transformation_filter` | diagnosed | Two specialists supply complementary evidence; the `source_data` competitor is refuted rather than merely out-scored (§23.1) |
| 2 | `diagnosed_late_source_arrival` | diagnosed | A different root cause wins on a symptom identical to #1 — the system is not hard-wired to one answer |
| 3 | `diagnosed_pipeline_task_failure` | diagnosed, held | High impact holds a well-evidenced diagnosis for a human anyway (FR-1105) |
| 4 | `not_an_incident_business_volume` | not_an_incident | The system declines to find a fault that is not there, before generating a single hypothesis |
| 5 | `diagnosed_misleading_memory` | diagnosed | The most relevant memory in the corpus is rejected on metadata; the diagnosis rests on current evidence alone (§23.3) |
| 6 | `inconclusive_missing_evidence` | inconclusive | Safe abstention with a human-escalation package (§23.4) |

FR-1301 #7 (MCP server unavailable) and #9 (prompt injection in logs) are deferred to a
future release — see [TODO.md](TODO.md).

§23.2 — *relevant memory improves order but is never proof* — is demonstrated inside
scenario 1, where a confirmed incident reorders hypotheses and the diagnosis still rests
on current evidence alone.

**Scenario 4 is a `not_an_incident`, not a diagnosis of `legitimate_business_change`.** A
campaign doubled signups, and the observability platform's current expectation already
reflects it; the alert fired against a stale forecast. FR-901's authoritative current
check settles that before hypothesis generation, so the run costs zero operational calls
and names no cause. Reaching the same conclusion by diagnosing the
`legitimate_business_change` category would have required inventing positive evidence
*for* normality, which is the wrong shape for a system whose whole discipline is refusing
to conclude without support.

### Adding a scenario

Create `data/scenarios/<id>/` with `alert.json`, `pipeline_operations.json`,
`data_observability.json`, optional `resources.json`, and `meta.json`. The `meta.json` is
the FR-1300 label and the evaluation's ground truth — it is authored beside the fixture
and never derived from a run (FR-1310).

```jsonc
{
  "expected_outcome": "diagnosed",           // and expected_verification, root cause,
  "required_evidence_tools": ["..."],        // required/forbidden tools, escalation,
  "severity": "high",                        // FR-1311 error weighting
  "risk": {"data_sensitivity": "low"},       // FR-1105 policy inputs
  "robustness_class": "prompt_injection",          // FR-1307 degraded-input class
  "injection_target_root_cause": "infrastructure"  // what an attack demands (FR-1301 #9)
}
```

The last two fields are read and scored but no scenario sets them yet; they are the seam
the deferred injection scenario slots into.

---

## Milestone 5 — Demo and Observability (in progress)

### The run journal (M5.1)

The persisted report says where an investigation **ended**. It says nothing about how it
got there: `persist_result` never held the per-round tasks, the branch scores that moved
between rounds, the Critic's reviews, or the trust-gate decisions. There was nothing on
disk to replay.

Every investigation now writes one:

```bash
uv run investigator journal --list                          # recent runs
uv run investigator journal --investigation inv_8f6730e35045
uv run investigator journal --investigation inv_... --round 2 --delta
uv run investigator journal --investigation inv_... --json   # the raw journal
```

```
[  9] r1 critic_review (4 ms)
      critic recommends 'diagnosed' (1 gap(s), 3 prune(s))
[ 10] r1 apply_branch_policy (0 ms)
      2 selected, 2 pruned
      · branches: [{"id": "B-H1-transformation_logic", "status": "selected", "score": 6.0, …
      · critic_disagreements: ["round 1: critic recommended pruning B-H4-data_quality; …
[ 11] r1 evaluate_stop (0 ms)
      diagnosed (stop) — leading hypothesis meets all FR-900 diagnosis criteria
```

Three decisions shape it:

**Nodes are wrapped once, at graph assembly**, rather than each node recording itself. A
node added later is journaled by default instead of by remembering to — and a test
asserts every compiled graph node appears as a stage, so opting out is not quietly
possible. The wrapper also absorbed the per-node LLM-call recording that four nodes used
to do by hand, which means completions are now attributed to the stage that made them and
a node cannot forget to trace them.

**A stage records what changed, not everything that exists.** The delta is built from the
node's own returned update, so a key nobody wrote never appears; evidence is compacted to
ids, verdicts, and discriminating flags; the report is omitted because it is held in full
beside the journal. A snapshot per node would multiply the payloads by seventeen and bury
the one thing the file exists to answer.

**The journal is state, so the reducers protect it.** `journal` is merged with
`dedupe_by("stage_id")` like every other accumulated collection, and a stage id is
assigned from the journal's own length — so replaying a checkpointed node produces the
same id and the duplicate is dropped (FR-205). It is written after the graph returns
rather than inside it, which is the only way `persist_result` can appear in its own
journal.

Each journal carries the FR-1206 version fingerprint, so a replay from three weeks ago
names the engine, model, prompts, corpus, tool permissions, and policy that produced it.

### LangSmith tracing (M5.2)

Off by default, and off without a key:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_PROJECT=data-reliability-investigator     # optional
uv run investigator investigate --scenario diagnosed_transformation_filter \
  --provider openai --model gpt-4o-mini
uv run investigator journal --investigation inv_...        # prints the run link
```

**Graph tracing comes free; model calls do not.** LangGraph nodes and specialist subgraphs
are instrumented through `langchain-core`, so the orchestration waterfall appears as soon
as the environment says to trace. The provider client speaks raw `httpx`, so its
completions are invisible no matter what the environment says — which would leave you
with the shape of an investigation and hollow nodes inside it. `complete_json` is
therefore decorated with `traced(run_type="llm")`, and prompts, models, latencies, and
token counts appear inside the nodes that caused them.

**Every run is attributed.** Runs are named `investigation:<scenario>`, tagged with
scenario, engine, and model, and carry the FR-1206 fingerprint — prompts, corpus, tool
permissions, and policy digests — as metadata. That is version attribution and FR-1204
dashboard slicing in one move: "accuracy dropped" becomes answerable by filtering on the
digest that changed.

**The journal links back to the trace.** A captured run id (and a deep link where one can
be resolved) is written into `outputs/journal/<id>.json`, so a stored run can be opened in
LangSmith later. Resolving the URL needs the network; the run id does not, so the id is
what the journal guarantees.

**It fails safe, in both directions.** Tracing requires the switch *and* a key: a key
sitting in the environment does not opt you in, and asking for tracing without one is
refused and logged rather than half-honoured. A tracing failure never fails an
investigation — an investigation that dies because its telemetry backend was unreachable
is a worse outcome than an untraced investigation.

> **Enabling tracing exports prompts and tool payloads to an external service.** Those
> carry untrusted alert and log content, and payloads are a 30-day retention class under
> FR-1208. `tracing.include_prompts: false` keeps the orchestration shape, model ids,
> latencies, and token counts while leaving the text behind. The provider API key cannot
> reach a trace — `complete_json` is a method on the object holding it, and `self` is
> dropped before inputs are recorded; a test pins that boundary rather than assuming it
> across a library upgrade.

### Investigation UI (M5.3)

```bash
uv sync --extra ui
uv run investigator ui           # http://localhost:8501
```

Built around an investigation moving through stages, not a chat transcript — because that
is what the system does: a bounded search over competing hypotheses that sometimes ends
by refusing to conclude.

The page opens with the project name, then the **diagnosis** — the root cause it named,
or the outcome when it named none — then the numbers behind it: confidence, risk tier,
release status, rounds, and calls against budget. That order is deliberate: the verdict
is the thing a reader came for, and it travels with them across every tab. The sidebar
switches between **Replay** (any journalled run) and **Run live**.

| Tab | What it answers |
|---|---|
| **Investigation** | The alert as received, the verification result, the verdict and what supports it. The landing tab, and the one to present from. |
| **Orchestration** | The graph with the executed path highlighted and a round slider that walks the run stage by stage — including the loop-back edge firing on a multi-round investigation. |
| **Hypotheses & beam** | Every branch's FR-703 score with all six rubric components broken out, each point's reason, and every disposition with why: falsified, below threshold, diversity hold, reopened. |
| **Agents** | One card per agent: role, prompt version, permitted tool set, calls made, decisions taken. Two of the four hold no operational tools at all, and the card says so. |
| **Evidence** | Every observation with tool, summary, supports/contradicts, discriminating flag and freshness, filterable to current-only, colour-separated by current versus historical. |
| **Memory** | Retrieval candidates with scores, the trust gate as a six-column pass/fail grid, and the influence: hypothesis order before and after. |
| **Critic** | Per-round reviews — unsupported claims, evidence gaps, recommended prunes and reopens — with the advisory recommendation set beside the deterministic outcome, and disagreements called out. |
| **Traces** | Model-call table, version attribution, the stage stream, and a deep link to the LangSmith run when one was captured. |
| **Safety** | Risk tier and its reasons, the release decision, blocked tool attempts, grounding, the FR-900 criteria, and the twelve FR-1305 gates run on demand. This is the FR-1204 operational safety dashboard with a face on it. |

**The logic is not in the UI.** `investigator/ui/views.py` is plain functions from a
persisted run to plain data, and `streamlit_app.py` only chooses widgets. That is why the
tabs have tests: that a rejected document names the gate that refused it, that the two
tool-less agents are shown holding nothing, that a pruned branch carries its reason. A UI
whose behaviour can only be checked by clicking it is a UI nobody can trust in front of an
audience.

**The orchestration graph is read from the compiled graph**, not drawn by hand, so the
picture cannot drift from the code it claims to describe.

**Raw evidence payloads appear for a live run only.** They are deliberately never written
to disk (FR-1208 retention), so a replay shows every observation and its verdict but not
the underlying payload — the UI says so rather than showing an empty panel.

### Still to come

The demo script (M5.4): a written run order and rehearsal checklist. Replay is the default
path; the live run is rehearsed but never load-bearing.

---

## Stated limitations
<a id="stated-limitations"></a>

**FR-1303 single-agent baseline: not measured.** The spec asks for the same scenarios run
through one generalist investigator, compared on accuracy, coverage, grounding,
abstention, latency, calls, and cost. It was dropped by decision to keep M4 within scope.
That leaves a real question open — *does the multi-agent design earn its complexity?* —
and this README does not claim an answer. FR-1303 also asks that negative or mixed
results be reported honestly; "not measured" is an honest answer where a quietly missing
section would not be.

What can be said without the baseline: role separation is what makes the permission
boundary enforceable (a single agent holding both tool sets has no boundary to enforce),
and the Critic's challenges are recorded as a separate agent's output rather than as
self-review. Neither of those is an efficiency claim, and neither substitutes for the
measurement.

**FR-1301 scenarios 7–15 are not built.** The shipped corpus is the six above; the rest
are queued in [TODO.md](TODO.md) for a future release. Several of the behaviours they
would exercise are covered by unit and integration tests rather than by end-to-end
fixtures: the `invalid_input` path for a malformed alert (#10), the closed taxonomy enum
for an out-of-taxonomy cause (#11), contradiction handling in the stopping evaluator
(#8), and `versions.diff()` for regression after a version change (#15). Tested is not
the same as demonstrable — none of them is a scenario you can run or replay, and the
tool-outage (#7) and injection (#9) paths in particular deserve end-to-end fixtures
because their controls span the client, the beam, and the risk gate.

**The operational safety dashboard (FR-1204) has no UI yet.** Its data exists —
acceptance gates, monitoring verdicts with owners and responses, the scenario matrix, the
review queue — and is reachable through `investigator evaluate` and `investigator
reviews`. Putting a face on it is M5.

# MAESTRO Threat Model

**Scope:** Agentic Data Reliability Investigator — a read-only, multi-agent diagnostic
decision-support system.
**Requirements:** FR-1104, NFR-008 (defense in depth), NFR-009, NFR-010.
**Status:** Milestone 4 complete. Controls marked *partial* are implemented in part; the
remainder is named in the row. Every control marked *implemented* has a test behind it.

Two scenarios in the M4.3 corpus exist to attack this document rather than to be
diagnosed: `diagnosed_misleading_memory` (L2 poisoned precedent) and
`not_an_incident_business_volume` (L5 false-positive pressure). A control nothing tries
to break is a claim, not a control — which is why the rows below distinguish a control
that is *enforced in code and unit-tested* from one an end-to-end scenario actually
attacks. The injection and tool-outage scenarios are deferred (see TODO.md), so those
rows say so rather than borrowing credit.

MAESTRO layers threats by where they live in an agentic system rather than by attacker
intent, which suits this system: the same injected string is a different problem at the
model layer than at the tool layer, and needs a different control at each.

NFR-008 requires that no control rest on a single model or prompt-based safeguard. The
test applied throughout this document: **if the model were fully adversarial, what would
still hold?** A control that fails that test is not counted as a control.

---

## The central design property

Almost every threat below is answered by the same structural fact, so it is stated once
here rather than repeated eleven times:

> **The model proposes; deterministic code disposes.**

An LLM supplies judgment at exactly four points — hypothesis generation, check selection,
observation interpretation, and critic review. Every one of those outputs is validated
against a closed vocabulary before anything else reads it. The model cannot name a tool
it was not granted, widen a permission, set a budget, mark evidence as current, or
declare an outcome. Diagnosis, abstention, escalation, pruning, and release are decided
by code that never sees a prompt.

This is why a compromised model degrades the system's *usefulness* but not its *safety*:
a hijacked model produces a bad investigation that abstains, not a confident wrong answer.

---

## L1 — Foundation Models

| Threat | Control | Status |
|---|---|---|
| Prompt injection via alert text, logs, or retrieved documents | Untrusted content is fenced and labelled as data in every prompt; **all model output is validated against closed vocabularies**; tool arguments are never model-generated. The scoring side is ready — a label declares what injected text demanded, and an injection counts as successful only if the run adopts that answer — but no end-to-end injection scenario ships yet (FR-1301 #9, deferred) | partial |
| Model hallucinates a root cause outside the taxonomy | `_categories()` discards any value not in `RootCauseCategory` (FR-400) | implemented |
| Model fabricates evidence or citations | Evidence is only ever created from a real MCP `ToolResult`; the grounding validator rejects references to non-existent evidence and abstains if it cannot repair (FR-1002) | implemented |
| Model asserts false confidence | Confidence is a categorical band set by the deterministic stopping evaluator, never by the model; no probabilities are accepted from it (FR-1001) | implemented |
| Provider outage or degradation | Retries with backoff; `on_failure: fallback` degrades to deterministic rules and **records the degradation** in `reasoning_fallbacks`; `fail_closed` abstains instead | implemented |
| Model or provider version drift changes behaviour silently | Model, provider, prompt, corpus, tool-permission, and policy versions are content-hashed per run, so an edit nobody remembered to version still shows up as drift; `diff()` names what moved (FR-1206) | implemented |
| Cost or token exhaustion | Per-task and global operational-call budgets; bounded `max_tokens`; cost thresholds in monitoring | implemented |

**Residual risk.** A subtly wrong-but-valid interpretation — the model says an
observation supports `source_data` when it does not. Structural controls cannot catch
this; it is bounded by requiring **two independent supporting tools** plus a
discriminating observation before any diagnosis (FR-900), so a single bad interpretation
cannot carry an outcome.

---

## L2 — Data Operations

| Threat | Control | Status |
|---|---|---|
| Retrieved memory treated as proof | Retrieved evidence carries empty `supports`/`contradicts` and `historical` freshness by construction; stopping and grounding both filter to `current_operational` independently (AD-005) | implemented |
| Misleading or stale memory influences a diagnosis | Six-gate trust gate: relevance, permitted type, confirmed precedent, metadata match, freshness, no conflict with current evidence — fails closed (FR-603) | implemented |
| Unconfirmed outcomes become precedent | Only `confirmed`/`approved`, non-superseded documents pass the gate (FR-1109 read side) | implemented |
| **The system teaches itself from its own unreviewed output** | Promotion requires a completed, attributed `HumanDecision` with a confirmed root cause; an unreviewed outcome is refused (FR-1109 write side, NFR-010) | implemented |
| Poisoned corpus document | Promotion is attributable and reversible; `supersede()` retires a document immediately, since the trust gate rejects superseded records | implemented |
| Sensitive data leaking into logs or prompts | Field-level redaction in structured logging (FR-1202); tool payloads truncated before entering prompts | implemented |
| **Telemetry exports incident content to a third party** | Tracing is off unless explicitly switched on *and* a key is present — a key alone does not opt you in. When on, prompts and tool payloads (untrusted alert and log content) leave the process; `tracing.include_prompts: false` exports shape, model ids, latency, and token counts without the text. The provider API key cannot reach a trace: `self` is dropped before inputs are recorded, and a test pins that boundary against a library upgrade (M5.2, NFR-006, FR-1208) | implemented |
| Retrieval index drift after corpus change | The corpus is content-hashed into every evaluation result, so a changed corpus is visible as drift rather than as an unexplained metric move (FR-1206). Re-running the evaluation before the change becomes default is a process step, not an enforced one — the gap is deliberate and named here rather than claimed as a control | partial |

**Residual risk.** A reviewer confirms a wrong root cause and it becomes precedent. This
is accepted: the system is decision *support*, and human authority is the intended
top of the chain. It is bounded by attribution (every promoted document names its
reviewer) and reversibility (`supersede`).

---

## L3 — Agent Frameworks

| Threat | Control | Status |
|---|---|---|
| Agent calls a tool outside its role | Role→tool permissions enforced in code at dispatch; the proxy refuses and counts the attempt (FR-506, FR-1106) | implemented |
| Commander over-grants tools to a specialist | Task tool lists are filtered against `allowed_tools(role)` at planning **and** re-checked at dispatch | implemented |
| Specialist escalates by choosing another server's tool | The role determines the permission set and is never model-selected; a cross-server choice is dropped | implemented |
| Confirmation bias — the first hypothesis wins | Beam search preserves ≥2 diverse hypotheses until each has had a discriminating check; the Critic explicitly challenges an unearned lead (FR-704, FR-130.8) | implemented |
| The Critic rubber-stamps the Commander | The Critic is a separate agent with its own prompt, no tools, and adversarial framing; its recommendation cannot upgrade caution into a diagnosis | implemented |
| Runaway loops | Round cap, global call budget, branch depth cap, and — before any of those — termination when no selected branch has an untried check | implemented |
| One specialist's failure corrupts another | Failure isolation: each specialist gets a private budget slice; a raising subgraph is recorded and the round continues (FR-204) | implemented |
| Non-deterministic merge order changes outcomes | Results merge by stable ID in admission order, never completion order (FR-203, FR-205) | implemented |

---

## L4 — Deployment and Infrastructure

| Threat | Control | Status |
|---|---|---|
| Agent bypasses MCP to read data directly | Providers live behind MCP servers in separate processes; a test asserts no agent module imports fixtures or providers (AD-003) | implemented |
| A mutating tool is introduced | Only read-only tools are exposed; a test asserts no discovered tool name matches mutation verbs (FR-504) | implemented |
| MCP server unavailable mid-investigation | Typed client error taxonomy; failures are recorded as *unavailability*, never as evidence against a hypothesis (FR-508); lost tool integrity raises the risk tier so a human sees the result (FR-1105). No end-to-end outage scenario ships yet (FR-1301 #7, deferred) | partial |
| Degradation is silently absorbed and reported as a clean result | A branch whose checks are unreachable is pruned as *blocked* rather than *exhausted*, so an outage cannot hide behind a routine disposition; failed tools are excluded from candidate generation, keeping the beam's view of what is reachable identical to the planner's; the escalation package names every unreachable source | implemented |
| Malformed or hostile tool response | Responses validated against the `ToolResult` contract; malformed responses become typed errors | implemented |
| Unbounded result size or hang | Server-side result caps and tool timeouts, with a client timeout above them (FR-507) | implemented |
| Secrets in environment or logs | Keys read from environment only, never logged; redaction list covers key-like field names | implemented |
| Checkpoint tampering or replay duplication | Reducers are first-write-wins on stable IDs, so replay cannot duplicate accepted evidence (FR-205) | implemented |

---

## L5 — Evaluation and Observability

| Threat | Control | Status |
|---|---|---|
| A silent regression ships | Twelve acceptance thresholds run as CI assertions over the shipped scenario set, not as a report table; an *unmeasured* gate is reported as skipped and fails the build, because silence is the failure these thresholds exist to catch (FR-1305) | implemented |
| Metrics hide a false-confident diagnosis | `ERROR_WEIGHTS` penalises false-confident diagnosis and missed escalation above correct abstention, and the headline `weighted_error` is severity-weighted on top (FR-1302, FR-1311) | implemented |
| An investigation is not reconstructable | Structured JSON trace with investigation/trace IDs on every event; run journal capturing round-by-round state | partial |
| Model-quality results presented as production SLAs | Every metric set carries its sample size, repeated runs report a range rather than a number (FR-1309), and no acceptance gate is a model-quality target (FR-1305) | implemented |
| A monitored signal nobody owns | Every threshold declares a warning level, a critical level, an owner, and a response (FR-1205) | implemented |
| A critical policy failure is alerted and then ignored | Critical **policy** breaches fail closed: `release_permitted` goes false and diagnoses stop being released (FR-1205) | implemented |
| Evaluator is not independent of the system under test | Ground truth comes only from labels authored beside each fixture; a test pins it by scoring one run state against two labels and getting two verdicts. High-risk routing is judged against the *label's* declared risk, never the tier the run assigned itself (FR-1310) | implemented |

---

## L6 — Security and Compliance *(cross-cutting)*

| Threat | Control | Status |
|---|---|---|
| Unauthorised access to traces or evidence payloads | Least-privilege read map per record class; absence of a grant is a denial (FR-1208) | implemented |
| Sensitive content retained indefinitely | Per-class retention: payloads and prompts 30 days, alerts and traces 90, reviewer data and evaluations 365 (FR-1208) | implemented |
| Deleting traces orphans an audit | Expiry **redacts content while preserving identifiers and evidence references**, so a report stays checkable after its payloads are gone | implemented |
| Autonomy exceeds authority on a sensitive incident | Deterministic risk classifier gates release; `high` holds for human review even when evidence is sufficient; `prohibited` blocks and fails closed (AD-007, FR-1105) | implemented |
| A diagnosis is released over a live objection | An unresolved Critic disagreement is bought out with one extra discriminating check, then routed to human review rather than forced (FR-1107) | implemented |
| An escalation is never answered | Escalation opens a `pending` review record at creation time, so an unanswered escalation is visible rather than absent (FR-1108) | implemented |
| Unreviewed outcomes change policy or memory | Only reviewed, attributable outcomes may change long-term memory (NFR-010) | implemented |

---

## L7 — Agent Ecosystem

| Threat | Control | Status |
|---|---|---|
| A compromised or swapped MCP server | Tool discovery is validated against the expected tool set; unknown tools are not callable | implemented |
| Provider substitution (a different model silently served) | The provider echoes the served model id; it is recorded per call in the trace | implemented |
| Cross-investigation contamination | Per-investigation IDs and checkpoint threads; separate threads never share state | implemented |
| Feedback loop poisoning via production traces | A backlog item sourced from a production trace is refused without a human review id to stand behind it; every item links observed failure evidence to the regression test that would catch it (FR-1207, NFR-010) | implemented |
| Downstream consumer treats an abstention as a diagnosis | `outcome` is an explicit enum and the escalation package is a separate artifact; a report never implies a cause it does not hold | implemented |

---

## Defense in depth (NFR-008)

The two threats most likely to be attempted are each answered at four independent layers,
none of which is a prompt:

**Prompt injection**
1. Content is fenced and labelled as data (prompt — the weakest layer, and not counted).
2. Output validated against a closed vocabulary (code).
3. Tool names and arguments never model-generated (code).
4. Permissioned proxy refuses out-of-role calls and counts the attempt (code).
5. Deterministic stopping, grounding, and risk gates own the outcome (code).

**Unauthorized tool use**
1. Role→tool map is the single source of truth (code).
2. Commander filters at planning time (code).
3. Specialist restricted to the task's allow-list (code).
4. Proxy enforces at dispatch and increments `blocked_attempts` (code).
5. Monitoring treats the first occurrence as critical and halts releases (code).

---

## Accepted risks

- **A reviewer confirms a wrong cause.** Human authority is the intended top of the
  chain. Bounded by attribution and reversibility.
- **Subtly wrong model interpretations.** Bounded by the two-independent-sources rule, not
  eliminated.
- **Deterministic reasoning is not a security control.** Running with
  `engine: deterministic` avoids model risk but is not the deployment configuration; the
  controls above are written for the LLM configuration.
- **Fixture providers are not production connectors.** Threats specific to live warehouse
  and orchestrator credentials are out of scope until real providers exist.

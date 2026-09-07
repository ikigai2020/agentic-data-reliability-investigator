# Example artifacts

Real output from real runs, committed so the system can be reviewed without running it.
Every file here was produced by the commands in the root README; nothing is hand-written.

Each investigation appears twice: the **report** is what the system concluded, and the
**journal** is how it got there — the stage-by-stage record of which node ran, in which
round, what it changed, and what it decided.

| Files | What it shows |
|---|---|
| `diagnosis-deterministic.*` | A clean diagnosis. Two independent supporting observations from different tools, the competing hypothesis refuted rather than out-scored, released. |
| `abstention-with-escalation.*` | The system refusing to conclude. No hypothesis reaches the evidence bar, so it abstains and assembles a handover package: verified facts, surviving alternatives, and the checks a person should run next. |
| `declined-not-an-incident.*` | A false alarm. A marketing campaign doubled signups; the authoritative current metric already reflects it, so the alert condition is absent. Note `calls_used: 0` — it declines before generating a single hypothesis. |
| `diagnosis-held-for-review.*` | **The most interesting one.** A real language model named the *wrong* root cause with *high* confidence. The Critic objected, the disagreement could not be resolved, and the release gate withheld the diagnosis for a human instead of shipping it. |
| `evaluation-deterministic.json` | Full evaluation over all six scenarios, three repeats, deterministic engine. |
| `evaluation-gpt-4o-mini.json` | The same evaluation on a real model. Accuracy collapses; the safety gates hold. |

## Reading a journal

```bash
uv run investigator journal --investigation <id> --delta
```

Or open any of the files directly — they are plain JSON. The interesting keys are
`stages[].summary` (one readable line per node), `stages[].delta` (what that node changed),
and, in the branch-policy stages, `branch_scores` — the itemized scoring rubric with the
reason for every point a hypothesis received.

## Reproducing them

```bash
uv run investigator investigate --scenario diagnosed_transformation_filter
uv run investigator evaluate --repeat 3
```

The deterministic engine is the default and needs no API key, so these are reproducible
exactly. The model-backed artifacts need `OPENAI_API_KEY` and `--provider openai`, and will
not reproduce byte-for-byte.

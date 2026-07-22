# mlfactory — the agent layer (L4/L5)

This directory is the **LLM orchestration layer**: the playbooks and subagents that sit on top of the
deterministic `mlfactory` CLI and turn it into an *ML factory*. The blueprint's thesis — **the LLM
decides and judges; the CLI computes; typed artifacts are the API** — lives here.

Built on Claude Code primitives:
- `commands/*.md` — **orchestrator playbooks** (slash commands, L5). One per pipeline stage/flow; each
  owns a control loop and spawns subagents. The playbook *is* the program — there is no compiled
  orchestrator.
- `agents/*.md` — **specialist subagents** (L4), each with its own isolated context and tool allow-list.
  Two archetypes: **deterministic-tool wrappers** (shell to one CLI subcommand, parse `--json`, never
  retry) and, from #11 on, **designers/analyzers** (the LLM judgment — target design, leakage scan,
  model-family ranking).

## What's here (the agent layer — #10 → #12, complete)
**Orchestrator playbooks (`commands/`):**
- `mlfactory-setup.md` — `/mlfactory-setup`, the input-boundary gate: interview → `configure` → `validate` (#34).
- `mlfactory-run.md` — `/mlfactory-run`, the deterministic pipeline orchestrator (#10).
- `mlfactory-eda.md` — `/mlfactory-eda`, the EDA & modeling-design stage (#11).
- `mlfactory-gates.md` — the human-in-the-loop gates: AI proposes, human decides (#12).

**Subagents (`agents/`):**
- `mlfactory-stage-runner` — the no-retry CLI-wrapper (deterministic).
- `mlfactory-artifact-validator` — the closing `validate-artifact` gate.
- `mlfactory-column-profiler` · `mlfactory-leakage-scanner` · `mlfactory-model-recommender` — EDA judgment (#11).
- `mlfactory-advisor` — surfaces a deterministic recommendation at a gate (#12).

## The invariants every part obeys
- **Deterministic-tool boundary** — no number is ever reasoned out in a prompt; the CLI computes.
- **No-retry wrappers, errors passed through verbatim** — a deterministic failure fails identically.
- **Gate, then write** — a heavy artifact is provisional until `validate-artifact` passes; delete on failure.
- **AI proposes, human decides** — judgment gates pause for the human (arriving in #11/#12).

## Status
The three agent-layer slices — #10 foundation → #11 EDA judgment → #12 human-in-the-loop gates — are
**complete** (epic #5). See the repo's GitHub issues and `STATUS.md`.

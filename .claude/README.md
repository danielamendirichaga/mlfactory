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

## What's here (issue #10 — the foundation)
- `commands/mlfactory-run.md` — `/mlfactory-run`, the pipeline orchestrator (deterministic happy path).
- `agents/mlfactory-stage-runner.md` — the no-retry CLI-wrapper.
- `agents/mlfactory-artifact-validator.md` — the closing `validate-artifact` gate.

## The invariants every part obeys
- **Deterministic-tool boundary** — no number is ever reasoned out in a prompt; the CLI computes.
- **No-retry wrappers, errors passed through verbatim** — a deterministic failure fails identically.
- **Gate, then write** — a heavy artifact is provisional until `validate-artifact` passes; delete on failure.
- **AI proposes, human decides** — judgment gates pause for the human (arriving in #11/#12).

## Roadmap
#10 foundation (this) → **#11** EDA stage + judgment subagents (leakage-scanner, model-recommender) →
**#12** human-in-the-loop gates. See the repo's GitHub issues and `STATUS.md`.

# Status — mlfactory (updated 2026-07-20)

## Where we are
The MVP deterministic core is complete (#1–#3), and the **agent layer has started**: **#10** (the
foundation — the `/mlfactory-run` orchestrator + CLI-wrapper subagents + the closing `validate-artifact`
gate, under `.claude/`) is done. Green and on `main`. Next: **#11 — the EDA stage + judgment subagents**.

**Health:** 204 tests green · ruff + mypy clean · CLI + live pipeline verified · Python 3.11 / uv.

## Done
- **Bootstrap** — lifted churnpilot's tested deterministic core into `mlfactory` (renamed), fresh repo. (`4ca941c`)
- **SaaS reference domain** — reworked streaming-B2C → **B2B SaaS account churn** (25 SaaS-native columns);
  DGP math + RNG preserved so determinism + all four levers hold. (`6e0a381`)
- **Decoupled core from domain** — `model.feature_columns` uses the generic `config.exclude_columns`;
  no domain imports in `compute/`. (`b8695a6`)
- **Reorganized** into `compute/` (generic core) + `domains/saas/` (reference domain) + app layer;
  imports made absolute; core ← domain ← app. (`4dd6093`)
- **Way of working established** — GitHub repo `danielamendirichaga/mlfactory` (private) + 6 roadmap
  issues; keystone files (`AGENTS.md`, `STATUS.md`, `CHANGELOG.md`) created.
- **#1 — Heavy contract tier + `validate-artifact`** — heavy `ArtifactBase` (markdown-frontmatter +
  lineage `parent`/`verification`, backward-compatible), the `validate-artifact` walker + output
  probe, `export-schemas`, and the first stage artifact (`saved-dataset`). +20 tests (188 total).
- **#2 — Standalone `engineer-features` stage** — closed 8-transform registry (fit-on-train /
  apply-outward; CV-folded target-encoding), model-ready postcondition, `feature-spec` artifact, and
  the `engineer-features` CLI. +10 tests (198 total).
- **#3 — CLI tool surface + planning docs** — `gen-model-card` (markdown card), `--json` machine output
  on `train`/`engineer-features`, and `docs/PRD.md` + `docs/ADRs.md` (reversing churnpilot ADR-001/009).
  +5 tests (203 total). **MVP deterministic core complete.**
- **#10 — Agent-layer foundation** — `.claude/` orchestrator playbook (`/mlfactory-run`) + no-retry
  CLI-wrapper subagents + the closing `validate-artifact` gate; a CLI pipeline E2E test. +1 test (204 total).

## In progress
- None. Workflow just established; next slice not yet started.

## Next up (active backlog — GitHub issues)
1. **The agent layer (epic #5)** — three behavioral (prompt/agent-engineering) slices; #10 done, build #11 → #12:
   - ✅ **#10** — foundation + deterministic pipeline orchestrator (done)
   - **#11** — EDA stage playbook + judgment subagents (column-profiler, leakage-scanner, model-recommender) — *the next slice.*
   - **#12** — human-in-the-loop gates (AI proposes, human decides)
2. #4 — Optuna hp-search + GBM engines (compute depth) — *deferred; optional enhancement, pick up anytime.*

**Dropped:** #6 (bundle distribution) — out of scope (fleet distribution; mlfactory ships as a `uv build`
wheel + the repo). **L1 MCP adapters deferred** — `source.py` (local file loader) IS the data adapter,
and Claude Code IS the inference; add a data-source MCP only if reaching a real external DB later.

## Key decisions (see AGENTS.md + REUSE-MAP.md)
- Generic ML factory per `ml-factory-architecture.md`; **churn/SaaS is the reference domain**, not the point.
- **MVP deterministic core first**, then the agent layer (the blueprint's own advice).
- Reversing churnpilot's ADR-001 (single→multi-agent) and ADR-009 (medium→heavy contract tier).
- Foundational/bootstrap commits landed on `main`; **from issue #1 onward, build each slice on a branch.**

## Blockers / open questions
- None.

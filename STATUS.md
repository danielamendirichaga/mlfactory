# Status — mlfactory (updated 2026-07-20)

## Where we are
**mlfactory is complete.** The deterministic core (#1–#3), the full agent layer (#10–#12, epic #5), and
the Optuna hp-search + `hist_gbm` compute depth (#4) are all shipped. **Every roadmap issue is closed.**

**Health:** 216 tests green · ruff + mypy clean · CLI + live pipeline verified · Python 3.11 / uv.

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
- **#11 — EDA judgment stage** — deterministic `leakage-scan` (tiered risks) + the `eda-exploration`
  artifact + the `/mlfactory-eda` playbook and judgment subagents (column-profiler, leakage-scanner,
  model-recommender). +4 tests (208 total).
- **#12 — Human-in-the-loop gates** — `advise --json` + the `/mlfactory-gates` playbook + the
  `mlfactory-advisor` subagent (propose the deterministic recommendation, wait for the human, honor
  overrides). +1 test (209 total). **Agent layer (epic #5) complete.**
- **#4 — Optuna hp-search + hist_gbm** — seeded TPE search (`compute/hp_search.py`, `train --optuna`) +
  the HistGradientBoosting engine in the model menu. +7 tests (216 total).

## In progress
- None. Workflow just established; next slice not yet started.

## Next up
**All roadmap issues are closed** — #1–#3 (core spine + feature stage + CLI/docs) · #7 (reorg/decouple) ·
#4 (Optuna + hist_gbm) · #10–#12 (agent layer, epic #5). mlfactory is a complete, generic, LLM-orchestrated
ML factory: deterministic tested CLI + heavy lineage artifacts + the agent layer, with a B2B SaaS
reference domain.

Possible future directions (not planned): lightgbm/catboost engines · the split-before-EDA leakage path ·
additional reference domains · flipping the repo public.
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

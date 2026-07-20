# Status — mlfactory (updated 2026-07-20)

## Where we are
MVP deterministic core now has the heavy contract-tier spine (#1) and the **standalone Stage-4
`engineer-features` stage** (#2 — closed transform registry, fit-on-train/apply-outward, feature-spec
artifact). Green and on `main`. Next slice: **issue #3 — CLI `--json` + `gen-model-card` + PRD/ADRs**.

**Health:** 198 tests green · ruff + mypy clean · CLI verified end-to-end · Python 3.11 / uv.

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

## In progress
- None. Workflow just established; next slice not yet started.

## Next up (active backlog — GitHub issues)
1. **#3 — CLI `--json` + `gen-model-card` + write PRD/ADRs** — *the next slice.*
2. #4 — Optuna hp-search + GBM engines (compute depth).
3. #5 — Epic: LLM orchestration layer (L4/L5) + MCP adapters (L1).
4. #6 — Epic: bundle distribution (L6).

## Key decisions (see AGENTS.md + REUSE-MAP.md)
- Generic ML factory per `ml-factory-architecture.md`; **churn/SaaS is the reference domain**, not the point.
- **MVP deterministic core first**, then the agent layer (the blueprint's own advice).
- Reversing churnpilot's ADR-001 (single→multi-agent) and ADR-009 (medium→heavy contract tier).
- Foundational/bootstrap commits landed on `main`; **from issue #1 onward, build each slice on a branch.**

## Blockers / open questions
- None.

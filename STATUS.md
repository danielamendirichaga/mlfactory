# Status — mlfactory (updated 2026-07-20)

## Where we are
MVP deterministic core is **bootstrapped, SaaS-reframed, decoupled, and reorganized** — green and
committed on `main`. GitHub repo + roadmap issues are set up. Ready to start the first real build
slice (**issue #1 — heavy contract tier**).

**Health:** 168 tests green · ruff + mypy clean · CLI verified end-to-end · Python 3.11 / uv.

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

## In progress
- None. Workflow just established; next slice not yet started.

## Next up (active backlog — GitHub issues)
1. **#1 — Heavy contract tier + `validate-artifact`** (L2 spine) — *the next slice; do this first.*
2. **#2 — Standalone `engineer-features` stage** (transform registry).
3. **#3 — CLI `--json` + `gen-model-card` + write PRD/ADRs.**
4. #4 — Optuna hp-search + GBM engines (compute depth).
5. #5 — Epic: LLM orchestration layer (L4/L5) + MCP adapters (L1).
6. #6 — Epic: bundle distribution (L6).

## Key decisions (see AGENTS.md + REUSE-MAP.md)
- Generic ML factory per `ml-factory-architecture.md`; **churn/SaaS is the reference domain**, not the point.
- **MVP deterministic core first**, then the agent layer (the blueprint's own advice).
- Reversing churnpilot's ADR-001 (single→multi-agent) and ADR-009 (medium→heavy contract tier).
- Foundational/bootstrap commits landed on `main`; **from issue #1 onward, build each slice on a branch.**

## Blockers / open questions
- None.

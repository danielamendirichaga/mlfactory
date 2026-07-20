# Changelog — mlfactory

Append-only log of what changed and when. Newest first.

## 2026-07-20 — Way of working established (GitHub + keystone files)
- Created private GitHub repo `danielamendirichaga/mlfactory`, pushed `main` (4 commits).
- Seeded the roadmap as GitHub issues #1–#6 (labels `mvp-core` / `epic`); #1 (heavy contract tier)
  is the active slice.
- Added keystone files `AGENTS.md`, `STATUS.md`, `CHANGELOG.md`; adopted the churnpilot workflow
  (orient → build slice → DoD → layered eval), adapted: **Launch = `uv build` wheel, not a web deploy**.

## 2026-07-20 — Structure: generic core + domains/saas (`4dd6093`)
- Reorganized the flat package into `mlfactory/compute/` (generic core: metrics/profile/split/model/
  compare/evaluate) and `mlfactory/domains/saas/` (reference domain: generate/policy/uplift/qini/monitor);
  top level is the app/wiring layer. Files moved with `git mv`; imports converted to absolute paths.
  Dependency direction is now core ← domain ← app. 168 green; ruff + mypy clean.

## 2026-07-20 — Decouple core from domain (`b8695a6`)
- `model.feature_columns` no longer imports the domain's `ORACLE_COLS`/`TREATMENT_COL`; it excludes
  whatever the new generic `config.columns.exclude_columns` declares. `train_uplift` self-augments the
  exclude list so `features: auto` A/B panels stay safe. `test_treatment` now proves the config-driven
  mechanism. 168 green; ruff + mypy clean.

## 2026-07-20 — Reference domain reframed to B2B SaaS (`6e0a381`)
- Reworked the bundled synthetic domain from streaming-subscription (B2C) into a **B2B SaaS
  account-churn** panel — product usage, seats, MRR, logins, discounts, support, company size,
  acquisition channel — with churn as the target and the randomized retention-offer uplift layer
  intact. DGP math + RNG order preserved (determinism + all four levers hold); columns/values/narrative
  reframed to SaaS. Generic core unchanged. 168 green; ruff + mypy clean.

## 2026-07-20 — Bootstrap mlfactory from churnpilot (`4ca941c`)
- Lifted churnpilot's tested deterministic core into a new generic ML-factory project, renamed
  `churnpilot`→`mlfactory`, fresh git repo. 20 modules + 22 test files + typed `parent_sha256`
  lineage artifacts (medium tier). Baseline: 168 tests green (Python 3.11, uv, ruff + mypy clean).

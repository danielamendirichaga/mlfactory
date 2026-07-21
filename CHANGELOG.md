# Changelog — mlfactory

Append-only log of what changed and when. Newest first.

## 2026-07-21 — #10 Agent-layer foundation (pipeline orchestrator + CLI-wrapper subagents)
- Started the LLM orchestration layer under `.claude/`: `commands/mlfactory-run.md` (the `/mlfactory-run`
  orchestrator — phase flow, deterministic-tool boundary, spawn-a-subagent-per-stage, closing
  `validate-artifact` gate + delete-on-failure), `agents/mlfactory-stage-runner.md` (no-retry
  CLI-wrapper), `agents/mlfactory-artifact-validator.md` (the closing gate), + `.claude/README.md`.
- Added `docs/example-feature-spec.yaml` (a SaaS feature recipe the orchestrator consumes).
- `tests/test_pipeline_e2e.py`: the deterministic pipeline the playbook drives, chained via the CLI
  (generate → split → engineer-features → validate-artifact → train → evaluate → gen-model-card),
  asserting the feature-spec validates + the model card renders. +1 test (204 total).
- Verified live end-to-end; the `features: auto` happy path trains on the planted leak (AUC ≈ 1.0) —
  the exact motivation for the EDA leakage-scanner (#11). ruff + mypy clean. (Closes #10)

## 2026-07-20 — #3 CLI tool surface (--json, gen-model-card) + PRD/ADRs
- Added `gen-model-card` (`mlfactory/model_card.py` + CLI) — renders an 8-section markdown model card
  (Purpose / Training Data / Features / Performance / Calibration / Slices / Limitations / Lineage)
  from the model + eval artifacts; the DS go/no-go surface.
- Added `--json` machine output to `train` and `engineer-features` (structured stdout summary; the
  human leakage warning is suppressed under `--json`) — the tool surface subagents will shell out to.
- Wrote `docs/PRD.md` + `docs/ADRs.md` for mlfactory — reversing churnpilot ADR-001 (single→multi-agent)
  and ADR-009 (medium→heavy contract tier).
- +5 tests (203 total green); ruff + mypy clean. MVP deterministic core (#1–#3) complete. (Closes #3)

## 2026-07-20 — #2 Standalone engineer-features stage (transform registry)
- Added the Stage-4 feature-engineering compute: a closed 8-transform registry
  (`compute/engineer_transforms.py`: drop_columns / log_transform / standard_scaler / one_hot /
  impute / date_parts / temporal_diff / target_encoding) as fit/apply/apply_train pairs, with the
  fit-on-train / apply-outward leakage-safe invariant (target_encoding CV-folded on train, full-train
  map on val/test).
- `compute/engineer.py`: `engineer_features` threads the split frames, serializes learned fit-params,
  and enforces the model-ready postcondition (produced columns numeric/boolean, no nulls/NaN/inf).
- New `feature-spec` stage artifact (registered in `ARTIFACT_MODELS`) + the `engineer-features` CLI
  (emits the artifact, which validates via `validate-artifact --probe-output`).
- +10 tests (198 total green); ruff + mypy clean. (Closes #2)

## 2026-07-20 — #1 Heavy contract tier + validate-artifact (L2 spine)
- Upgraded `ArtifactBase` to the heavy tier (backward-compatible): a lineage `parent` pointer +
  `Verification` block + stage/version/caveats/backtrack_signals; **markdown-with-frontmatter**
  serialization (`to_markdown`/`from_markdown`) alongside `write_json`; `schema_hash` + `file_sha256`.
- Added `validate-artifact` — the lineage walker (cycle → existence → sha256 → schema → parent-type
  → verification-status) + on-disk output probe (rows + schema_hash) — and `export-schemas` (+`--check`).
- First heavy stage artifact `SavedDatasetArtifact` (the input contract) + an `ARTIFACT_MODELS`
  registry; `artifacts.py` promoted to a package. CLI: `validate-artifact`, `export-schemas`.
- +20 tests (188 total green); ruff + mypy clean. (Closes #1)

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

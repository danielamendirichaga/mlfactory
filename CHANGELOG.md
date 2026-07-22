# Changelog — mlfactory

Append-only log of what changed and when. Newest first.

## 2026-07-22 — #36 Honesty fixes surfaced by running on real (Telco) data
The agent, running on a real dataset, flagged three "claims-X-does-Y" defects. On verification, two were
real and one was a misdiagnosis:
- **Stratified split (real fix).** The `random` strategy was *plain* random (prevalence drifted across
  train/val/test) though the split gate advertised "stratified". It now stratifies on the target
  (per-class bucketing); `time`/`grouped` are unchanged.
- **Model-card provenance (real fix).** `model_card.py` hardcoded *"synthetic B2B SaaS domain"* into every
  card — false on real data. Now conditional on `source.kind` (recorded on the `ModelCard`): synthetic →
  the synthetic caveat; real → an honest "held-out split of the provided data" caveat.
- **Logistic L1 vs L2 (false alarm → a useful decision).** The reported "silently L2" was wrong: sklearn
  1.9 uses `l1_ratio` (1.0=L1) and deprecates `penalty`, so the original `l1_ratio=1.0` was already L1, as
  documented (verified: 4/12 zero coefs). No bug — but regularization is now a decision
  (`config.decisions.modeling.penalty` = l1/l2/elasticnet → `l1_ratio`, default l1 = prior behavior);
  bumped `max_iter` to 5000 so saga+L1 converges without warnings.
- +5 tests (284 total green); ruff + mypy clean. (Closes #36)

## 2026-07-21 — #34 Guided config setup (/mlfactory-setup + configure) — the input-boundary gate
The one config piece without a tested writer was the source + schema (data path, target, column map).
- Added `config.write_source_schema` + the `configure` CLI (`--source-kind` / `--path` / `--dsn`+`--table`,
  `--target` / `--positive-value` / `--id-col`, optional `--date-col` / `--value-col` / `--features`):
  writes a validated source+schema, **preserving any existing `decisions:` block**. Fails cleanly on a
  bad mapping (e.g. `file` source with no path).
- Added the `.claude/commands/mlfactory-setup.md` playbook: interview the DS for their data + target,
  call `configure`, then `validate` — chat-to-config becomes a designed, deterministic gate, not
  free-hand YAML editing.
- Verified live: `configure` on a real parquet → a clean churn.yaml → `validate` reports USABLE.
  +6 tests (279 total green); ruff + mypy clean. **Closes #34.**

## 2026-07-21 — #25 (S6) Downstream reads the decision record — epic #17 complete
Epic #17, slice 6 (closes #25) — the final slice. The domain layer's economics now come from the record.
- `simulate-policy` and `policy-contrast` take `save_rate` / `offer_cost` / `budget` from
  `config.decisions.policy` (were silent `0.3` / `$5` / `none`); the `run` copilot seeds its prompts from
  the record too. `monitor` takes the drift bar from `config.decisions.monitoring.drift_threshold` (was a
  hardcoded `0.25`). CLI flags still override.
- Verified live: recorded `policy.offer_cost=9` shows in the policy report; recorded
  `monitoring.drift_threshold=0.4` shows in the drift report. +3 tests (273 total green); ruff + mypy clean.

**Epic #17 (surface + propagate DS decisions) is complete** — every back-half stage (features, train,
select, evaluate, ship, model card, policy, monitor) now surfaces its decisions at a gate and reads them
from the typed `config.decisions` record, defaults reproducing the pre-#17 behavior. **Closes #25.**

## 2026-07-21 — #24 (S5) Model card is authored, not just generated
Epic #17, slice 5 (closes #24). The card now carries the DS's judgment, not only the generated metrics.
- `train` writes `config.decisions.caveats` onto the `ModelCard`, so accumulated gate/EDA caveats
  (the leakage note, "prefer isotonic over SMOTE", …) reach the card's Limitations instead of being lost.
- Added `CardDecisions` (`config.decisions.card`): intended_use / out_of_scope / known_failure_modes /
  sign_off. `gen_model_card(..., authored=...)` renders Intended Use / Out of Scope (after Purpose), the
  failure modes (in Limitations), and a Sign-off section; `gen-model-card --config` supplies them.
- Behavior-preserving: with nothing authored, the card is unchanged.
- Verified live: a card with Intended Use + Sign-off sections and a propagated calibration caveat.
  +6 tests (270 total green); ruff + mypy clean. **Closes #24.**

## 2026-07-21 — #23 (S4) Evaluate & ship read the decision record
Epic #17, slice 4 (closes #23). The operating point, the slices, and the ship bar now come from the
record instead of a silent 0.5 / hardcoded segments / a baked-in 0.65-0.10.
- `evaluate_model` takes `threshold` + `segment_cols` from `config.decisions.evaluation` when not passed
  explicitly (the CLI `--threshold` is now opt-in; the report records the resolved threshold).
- `recommend_ship` judges against the recorded `min_auc`/`max_ece` (the `run` copilot passes them).
- `set_decision` now JSON-parses its value, so list/number/bool decisions work (`--value '["plan_tier"]'`,
  `0.3`, `true`); bare words still fall through as strings.
- Verified live: recorded `threshold=0.15` shows as `@0.15`, recorded `segment_cols=["plan_tier"]` drops
  the region slice. +6 tests (264 total green); ruff + mypy clean. **Closes #23.**

## 2026-07-21 — #22 (S3) Train & select read the decision record
Epic #17, slice 3 (closes #22). The model-selection and training knobs now come from the record instead
of being hardcoded or hidden.
- `compare_models` ranks by `config.decisions.modeling.primary_metric` (each row carries `primary` +
  `primary_metric`) with the record's stability bars (`max_auc_drop`/`max_score_psi`); `recommend_model`
  selects the most-stable model on that metric (falls back to AUC for legacy rows).
- The `train` CLI reads `modeling.{imbalance,calibrate,tune}` as the regime; explicit CLI flags
  force-on / override. `--engineered` still ignores the record's tune (unsupported there).
- The Model gate in `/mlfactory-gates` (+ `/mlfactory-run`) persists the metric / imbalance / calibration.
- Verified live: recorded `primary_metric=pr_auc` reorders `compare`; recorded `calibrate=true` produces
  a calibrated model with no CLI flag. +6 tests (258 total green); ruff + mypy clean. **Closes #22.**

## 2026-07-21 — #21 (S2b) Construction transforms + the FE gate — S2 complete
Epic #17, slice 2b (closes #21). Now a recipe can *build* signal, and the feature-engineering decision
is a real gate instead of an improvised step.
- Added `ratio` (`input[0]/input[1]`, division-safe: zero denominator → `on_zero`, default 0.0) and
  `interaction` (product of >= 2 inputs) to the transform registry (now 10). A null input propagates and
  is caught by the model-ready postcondition — impute first if inputs can be missing.
- Added `FeatureDecisions` to the decision record (`config.decisions.features.approach`, default `skip`
  = train on the raw split); surfaced the **FE gate** in `/mlfactory-gates` (now five gates),
  `/mlfactory-run` (branches skip vs. recipe/hybrid), and `/mlfactory-eda` (recommends the approach).
- Verified end-to-end: a `revenue_per_seat` (ratio) + `activity_x_actions` (interaction) recipe reaches
  the model. +11 tests (252 total green); ruff + mypy clean. **Closes #21 — S2 complete.**

## 2026-07-21 — #21 (S2a) Feature-spec → train — an engineered recipe reaches the model
Epic #17, slice 2a. Before this, `train` only read the raw split, so `engineer-features` produced a
`feature-spec` that never touched the model. Now `train --engineered` trains on the engineered output.
- `train_model(engineered=True)` uses a passthrough-style preprocessor (`_engineered_preprocessor`):
  imputes leftover nulls + one-hots any surviving raw categoricals, but does NOT re-scale numerics —
  the recipe owns scaling, so re-scaling would double-transform. `evaluate` works unchanged (the fitted
  pipeline scores the engineered test split).
- `ModelCard.engineered` records it (shown on the card's Fit-options line); `--engineered` on the `train`
  CLI. Rejects `--tune`/`--optuna`/`--early-stopping` in engineered mode for now (they build their own
  preprocessing) — deferred.
- `/mlfactory-run` now trains + evaluates on `data/features/*` (the engineered dataset); the stale
  "engineered dataset not fed to training" scope note is gone.
- Verified end-to-end (engineer-features → train --engineered → evaluate → model card). +8 tests
  (241 total green); ruff + mypy clean. (Part of #21 — S2b: construction transforms + FE gate, next.)

## 2026-07-21 — #19 (S0) Decision-record foundation — gates write it, stages read it
Epic #17, slice 0 — the spine the rest of the epic builds on.
- Added a typed `DecisionRecord` on `ChurnConfig` (`decisions:` block): `modeling` (primary_metric /
  imbalance / calibrate / tune / stability bars), `evaluation` (threshold / ship bar / segments),
  `policy` (save_rate / offer_cost / budget / targeting), `monitoring` (drift bar), and `caveats`.
  **Every default equals the value the pipeline hardcodes today** — a locked-in test asserts this
  against the real `compare`/`evaluate`/`recommend_ship`/`monitor` sources, so later slices swap a
  hardcoded default for `config.decisions.*` without changing behavior until a gate overrides it.
- Added `set_decision` (dotted-key writer; pydantic-coerced, comment-preserving, validated before and
  after) + `_write_decisions_block`, and the `record-decision` (write) / `decisions` (read) CLI.
- Backward-compatible: a config without a `decisions:` block gets the defaults.
- +9 tests (233 total green); ruff + mypy clean. (Closes #19)

## 2026-07-21 — #20 (S1) Leak-drop propagation — the confirmed drop reaches the pipeline
Epic #17 (surface + propagate DS decisions), slice 1. Closes a gap found by *using* the tool: the EDA
leakage-drop was recorded in the `eda-exploration` artifact but never written to `churn.yaml`, so `train`
(which reads `config.exclude_columns` via `feature_columns`) silently kept training on the leak.
- Added `config.set_exclude_columns` (+ `_set_exclude_columns_text`): a comment-preserving, validated,
  in-place writer for `schema.exclude_columns` (add / remove / set), refusing to write an unreadable config.
- Added the `exclude-columns` CLI (`--add`/`--remove`/`--set`/`--json`) — records a confirmed leakage-drop
  into the config.
- Wired the leakage gate to persist the drop in `/mlfactory-eda`, `/mlfactory-gates`, and `/mlfactory-run`
  (the decision only takes effect once it's in the config, not the artifact).
- +8 tests (224 total green); ruff + mypy clean. (Closes #20)

## 2026-07-21 — #4 Optuna hp-search + hist_gbm engine (compute depth)
- Added `compute/hp_search.py::optuna_search` — seeded Optuna TPE over a per-family space
  (logistic / rf / xgboost / hist_gbm), CV-AUC scored, winner refit; deterministic (same seed → same
  winner). Wired into `train --optuna --trials N` (+ a `train_model(optuna=...)` path with
  mutual-exclusion guards).
- Added the `hist_gbm` (HistGradientBoosting) engine to the model menu (NaN-native, no imputation).
- Added the `optuna` dependency. +7 tests (216 total green); ruff + mypy clean. (Closes #4)

## 2026-07-21 — #12 Human-in-the-loop gates (AI proposes, human decides) — agent layer complete
- Added `advise --json` (structured deterministic recommendations a gate parses), the
  `.claude/commands/mlfactory-gates.md` playbook (the four gates — target / leakage exclusion / model
  choice / ship go/no-go — + the DS-leads posture + override flags), and the `mlfactory-advisor`
  subagent (propose {what/why/action} at a gate, then wait for the human).
- Wired the gates into the `/mlfactory-run` and `/mlfactory-eda` orchestrators; `.claude/README.md`
  marks the agent layer complete.
- +1 test (209 total green); ruff + mypy clean. **Closes #12 — and completes the agent layer (epic #5).**

## 2026-07-21 — #11 EDA judgment stage (leakage-scan + eda-exploration + judgment subagents)
- Deterministic substrate: `compute/profile.py::scan_leakage` tiers target correlations into structured
  leakage_risks (|corr|>0.99 perfect_predictor/drop; 0.9–0.99 near_perfect/inspect), plus a `leakage-scan`
  CLI (`--json`). On the SaaS panel it flags the planted `cancel_page_visits_30d` trap at near_perfect (+0.92).
- New `eda-exploration` stage artifact (+ `LeakageRisk`, `FamilyRec`) registered in ARTIFACT_MODELS.
- Agent layer: `.claude/commands/mlfactory-eda.md` (EDA orchestrator) + `agents/` column-profiler,
  leakage-scanner (wraps leakage-scan; escalates near_perfect → drop for posterior/derived cases),
  model-recommender (family ranking + baseline from the profile).
- +4 tests (208 total green); ruff + mypy clean. (Closes #11)

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

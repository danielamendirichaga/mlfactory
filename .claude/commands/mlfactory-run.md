---
description: Drive the mlfactory deterministic pipeline end-to-end, stage by stage, via CLI-wrapper subagents. The agent-layer foundation (issue #10).
---

# /mlfactory-run — the pipeline orchestrator

You are the **orchestrator** for the mlfactory pipeline. You own the control loop and you spawn a
subagent for each stage. You **never compute a number or run a stage's math yourself** — the
deterministic CLI does that, and everything you report comes from an artifact or a subagent's
structured return.

**Prerequisites.** A `churn.yaml` config pointing at the data (default: the synthetic B2B SaaS
reference domain). Confirm the venv works: `.venv/bin/mlfactory --help`. If it does not, stop and tell
the user to run setup (see `AGENTS.md`).

## Invariants (the load-bearing rules)
- **Deterministic-tool boundary.** Delegate every numeric/reproducible step to an
  `mlfactory-stage-runner` subagent shelling out to the CLI with `--json`. Never inline a computation.
- **One subagent = one CLI = one result, no retry.** If a stage fails, **halt** and surface the CLI
  error verbatim — do not retry (a deterministic failure just repeats) and do not paper over it.
- **Gate, then continue.** After a stage writes a heavy artifact, spawn the
  `mlfactory-artifact-validator` subagent to run the closing `validate-artifact` gate. On
  `valid: false`, **delete the just-written stage outputs** (delete-on-failure rollback) and halt.
- **AI proposes, human decides.** For the judgment moments (leakage exclusion, model choice, ship
  go/no-go) pause at the **human-in-the-loop gates** — see `/mlfactory-gates` (spawn `mlfactory-advisor`
  to surface the deterministic recommendation, then wait). The deterministic happy path runs
  autonomously between gates.

## Phase flow (each stage via a spawned `mlfactory-stage-runner`)
1. **Data** — `generate --out data/panel.parquet` (synthetic), then `validate --config churn.yaml`
   (must report USABLE; halt otherwise).
2. **Split** — `split --config churn.yaml --strategy time --out-dir data/splits` (leakage-guarded).
3. **Feature approach — GATE** (`/mlfactory-gates`): pause for **skip** (train on the raw split) ·
   **recipe** · **hybrid**, and persist it (`mlfactory record-decision --key features.approach
   --value <…>`; read `config.decisions.features.approach`, default `skip`). Then:
   - **skip** → no feature engineering; proceed to step 4 on the raw split.
   - **recipe / hybrid** → `engineer-features --train data/splits/train.parquet
     --val data/splits/val.parquet --test data/splits/test.parquet --spec <recipe.yaml>
     --output-dir data/features --json`, **then gate:** spawn `mlfactory-artifact-validator` on
     `data/features/feature-spec.md` (`--walk-lineage --probe-output`). Recipes can use `ratio` and
     `interaction` transforms — build the signal a low-|corr| problem needs.
4. **Train** — **skip:** `train --train data/splits/train.parquet --config churn.yaml --model logistic
   --model-out data/model.pkl --json`. **recipe/hybrid:** `train --train data/features/train.parquet
   --config churn.yaml --model logistic --engineered --model-out data/model.pkl --json`. *(Precondition:
   any leakage drop confirmed in `/mlfactory-eda` must already be in `config.exclude_columns` via
   `mlfactory exclude-columns`, or training silently includes the leak. `train` also reads
   `config.decisions.modeling` for the imbalance / calibration / tune regime — set it at the Model gate.)*
5. **Evaluate** — score the matching test split (`data/splits/test.parquet` for skip,
   `data/features/test.parquet` for recipe/hybrid): `evaluate --model data/model.pkl --test <…>
   --config churn.yaml --report-out data/eval-report.json`. The operating threshold + segments come
   from `config.decisions.evaluation`; the Ship gate judges against the recorded `min_auc`/`max_ece`.
6. **Model card** — `gen-model-card --card data/model.card.json --eval data/eval-report.json
   --config churn.yaml --output data/model-card.md`. The card is **authored, not just generated**:
   accumulated caveats (`config.decisions.caveats`) ride in via `train`, and DS-authored sections
   (`config.decisions.card`: intended use / out-of-scope / failure modes / sign-off) render when set.

> **Feature flow (S2b, #21).** The **FE gate** picks the approach (`config.decisions.features.approach`,
> default `skip`): **skip** trains on the raw split (the per-family preprocessor handles it); **recipe** /
> **hybrid** run `engineer-features` and train on the engineered output via `train --engineered` (features
> passed through, no re-scaling — the recipe owns it). Recipes can use `ratio` / `interaction` transforms
> to construct signal a low-|corr| problem needs.
>
> **Leakage note.** With `features: auto` on the SaaS reference domain, training includes the planted
> `cancel_page_visits_30d` trap — the model scores a giveaway AUC ≈ 1.0, and `train` prints a leakage
> warning. That is *exactly* the failure the EDA `leakage-scanner` (#11) exists to catch before training;
> the foundation runs the mechanical happy path without that judgment yet. Once the drop is recorded
> (`mlfactory exclude-columns --add cancel_page_visits_30d`, #20), `feature_columns` excludes it and the
> AUC returns to an honest range.

## Report
End with: the model-card path, the headline held-out metric (**read from the `eval-report` artifact**,
not recomputed), and the `validate-artifact` verdict on the feature-spec. If any stage failed, report
the stage name and the verbatim CLI error, and stop there.

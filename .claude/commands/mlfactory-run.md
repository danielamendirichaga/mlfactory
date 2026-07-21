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
3. **Feature engineering** — `engineer-features --train data/splits/train.parquet
   --val data/splits/val.parquet --test data/splits/test.parquet --spec docs/example-feature-spec.yaml
   --output-dir data/features --json`. **Then gate:** spawn `mlfactory-artifact-validator` on
   `data/features/feature-spec.md` (`--walk-lineage --probe-output`).
4. **Train** — `train --train data/splits/train.parquet --config churn.yaml --model logistic
   --model-out data/model.pkl --json`. *(Precondition: any leakage drop confirmed in `/mlfactory-eda`
   must already be in `config.exclude_columns` via `mlfactory exclude-columns`, or training silently
   includes the leak.)*
5. **Evaluate** — `evaluate --model data/model.pkl --test data/splits/test.parquet --config churn.yaml
   --report-out data/eval-report.json`.
6. **Model card** — `gen-model-card --card data/model.card.json --eval data/eval-report.json
   --output data/model-card.md`.

> **Honest scope note.** In this foundation slice the model stage trains on the leakage-guarded
> **split** (step 2), while feature engineering (step 3) emits and *validates* a `feature-spec`
> artifact alongside it. Fully feeding the engineered dataset into training is the dataset/model
> stage-integration slice — not this one.
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

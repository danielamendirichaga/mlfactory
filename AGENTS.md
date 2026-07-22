# AGENTS — mlfactory

> Keystone file: how this project works. Auto-loads every session. Keep it current.
> Repo: https://github.com/danielamendirichaga/mlfactory (private) · Issues track the roadmap.
> **New to the project?** Read [`README.md`](README.md) first — this file is the contributor/agent guide.

## The one idea
An **LLM-orchestrated ML factory**: the AI decides and judges; a **deterministic, unit-tested CLI**
does *all* the compute; **typed artifacts with lineage** are the contract between stages. Every
number is reproducible, every artifact is lineage-verified, and the LLM never silently substitutes
for a computation. Full blueprint: `ml-factory-architecture.md`. Build plan: `REUSE-MAP.md`.

**Status:** MVP deterministic core, bootstrapped from churnpilot (`../AI&DS_lab`) and generalized.
Currently **216 tests green, ruff + mypy clean**. **mlfactory is complete** — the deterministic core,
the full agent layer (#10–#12, `.claude/`), and Optuna hp-search + `hist_gbm` (#4). See `STATUS.md`.

## Architecture & conventions
- **Layering — core ← domain ← app** (dependency direction, enforced by discipline):
  - `mlfactory/compute/` = generic, domain-agnostic deterministic core. **Must NOT import any domain.**
  - `mlfactory/domains/saas/` = the B2B SaaS reference domain. May import the core.
  - `mlfactory/` top level = app/wiring (config, contracts, source, validate, recommend, viz, cli).
- **The generic core hardcodes NO domain column names.** Columns that must never be features
  (leakage / A-B oracle / experiment ground-truth) are declared in `config.columns.exclude_columns`;
  the domain populates it. (This is why `model.feature_columns` takes no domain import.)
- **Determinism:** seed everything (default 42); same inputs → same outputs.
- **Config-driven:** everything reads the `churn.yaml`-style config — no hardcoded column names.
- **The tested metric core (`compute/metrics.py`) imports only numpy/pandas** — never sklearn/xgboost.
- **AI proposes, human decides.** Recommendations live as tested rules (`recommend.py`), not an LLM.
- **DS decisions live in `config.decisions`** — a typed `DecisionRecord` (metric/threshold/economics/
  drift/caveats) whose defaults reproduce pre-#17 behavior. Gates **write** it (`record-decision`,
  `exclude-columns`); stages **read** `config.decisions.*`. Surface + propagate = epic #17.

## Key files (map)
- `mlfactory/compute/` — `metrics` · `profile` (+`scan_leakage`) · `split` · `model` · `compare` · `evaluate` · `engineer`(+`engineer_transforms`) · `hp_search` (generic core)
- `mlfactory/domains/saas/` — `generate` · `policy` · `uplift` · `qini` · `monitor` (B2B SaaS reference domain)
- `mlfactory/` — `config.py` · `artifacts/` (typed lineage contracts + `validate-artifact` + schemas) ·
  `source.py` · `validate.py` · `recommend.py` · `model_card.py` · `charts.py` · `report.py` · `cli.py`
- `tests/` — 216 green tests (one file per module + capstones + a CLI pipeline E2E)
- `.claude/` — the agent layer (L4/L5): `commands/` orchestrator playbooks + `agents/` subagents (see `.claude/README.md`)
- `docs/` — `PRD.md` + `ADRs.md` + `example-feature-spec.yaml`
- `REUSE-MAP.md` — churnpilot→factory reuse plan · `ml-factory-architecture.md` — the blueprint

## Commands
- setup:   `uv venv --python 3.11 .venv && uv pip install --python .venv -e . && uv pip install --python .venv pytest ruff mypy types-PyYAML`
- test:    `.venv/bin/python -m pytest tests -q`
- lint/fmt:`.venv/bin/ruff check .`  /  `.venv/bin/ruff format .`
- types:   `.venv/bin/mypy mlfactory`
- run:     `.venv/bin/mlfactory <cmd>`  (init | generate | validate | profile | metrics | split | train | compare | evaluate | simulate-policy | report | monitor | advise | run | version)
- factory: `engineer-features` (Stage 4) · `leakage-scan` (EDA) · `validate-artifact --walk-lineage --probe-output` · `export-schemas [--check]` · `gen-model-card` · `exclude-columns` (record a confirmed leakage-drop → `config.exclude_columns`) · `record-decision`/`decisions` (write/read the `config.decisions` record). Stage commands take `--json` (the tool surface subagents shell out to).
- v2 uplift: `generate --treatment` | `train-uplift` | `uplift-eval` | `policy-contrast`
- build:   `uv build`  → an installable wheel (this is the "release" — there is NO web deploy)

### The SaaS reference pipeline (what you drive)
`generate --out data/saas.parquet` → `validate` → `profile` → `split --strategy time` → `train` →
`compare` → `evaluate` → `simulate-policy` → `report` → `monitor`. Note: the generator flag is
`--accounts` (B2B SaaS account panel), not `--subscribers`.

## Way of working (persistence + process)
Each session starts fresh; **files on disk are the memory.** Three keystone files carry state:
`AGENTS.md` (how it works — this file), `STATUS.md` (where we are), `CHANGELOG.md` (history).

**Session start — orient before touching anything:** read `AGENTS.md` → read `STATUS.md` →
open the active GitHub issue.

**Build:** one slice at a time, smallest viable first, on a branch (never commit straight to `main`
for feature work). **Definition of Done:** a slice is not done until `STATUS.md` + `CHANGELOG.md`
are updated and its GitHub issue is closed/checked off — and `AGENTS.md` too if a convention,
architecture rule, key file, or gotcha changed.

**Eval/Test ladder (run in order; green at each layer before moving on):**
1. Static — `ruff check` + `ruff format --check` + `mypy mlfactory`
2. Unit — pytest on isolated logic (metric properties, validators, rules)
3. Integration — pytest on modules together (capstone pipelines)
4. Behavioral/E2E — the real CLI flow via `CliRunner` against the slice's acceptance criteria
5. Manual smoke — run the command / open `report.html`

## Gotchas
- sklearn/xgboost live only in `compute/model` + domain modules; the tested metric core imports only numpy/pandas.
- xgboost needs the OpenMP runtime on macOS: `brew install libomp` (already installed on this machine).
- Time-aware split is the default; `--strategy random` is opt-in and **wrong for panel data** (entity leakage).
- Leakage trap: `cancel_page_visits_30d` is planted to spike with churn — `profile`/`train` flag it. Keep it out by recording the drop: `mlfactory exclude-columns --add cancel_page_visits_30d` — this writes `config.exclude_columns`, which `split`/`train` read. The EDA `eda-exploration` artifact recording the drop does **not** by itself enforce it (epic #17 / S1).
- Feature engineering: the **FE gate** (`config.decisions.features.approach`, default `skip`) picks skip
  (train on the raw split) vs. recipe/hybrid → `engineer-features` (registry incl. `ratio`/`interaction`)
  → `train --engineered` (features passed through, not re-scaled — the recipe owns it). (epic #17 / S2)
- Train & select regime: `compare`/`recommend_model` rank+select on `config.decisions.modeling.primary_metric`
  (default auc) with the record's stability bars; `train` reads `modeling.{imbalance,calibrate,tune}`
  (CLI flags force-on). Set them at the Model gate. (epic #17 / S3)
- Generic core hardcodes no domain columns — mark never-features via `config.columns.exclude_columns`.
- Synthetic data only; `data/` is gitignored; never commit real customer data / PII.
- `mlfactory` = the tool name; the bundled domain is B2B SaaS account churn (NOT streaming/fintech).

# The ML Factory — An Architecture Blueprint for an LLM-Orchestrated Machine-Learning Pipeline

> 📐 **This is the design blueprint, not the code.** [mlfactory](README.md) (this repo) is a working,
> fully-tested implementation of it. To understand or run the project, start with the
> [README](README.md) or the plain-language [explainer](ml-factory-explained.md); read this document for
> the *why* behind the architecture and the patterns it's built on.

> **Complete.** Reverse-engineered from a production fintech ML pipeline via a parallel teardown of its command playbooks, sub-agent prompts, deterministic CLI source, shared specs/schemas, and MCP-adapter + bundle-build repos. Abstracted for reuse; secrets redacted at source.

---

## 0. How to read this document

**Who this is for.** A fresh, capable coding-agent session (e.g. Claude) with **no prior context**, tasked with building a **generic, open-sourceable "ML factory"** — an LLM-orchestrated pipeline that turns a dataset into a trained, validated, documented model — for *some* domain. This document is the blueprint: the architecture, the contracts, the prompt-engineering patterns, and a build-your-own recipe.

**What it is / isn't.** It is reverse-engineered from a real production system (a fintech's internal "data → model" pipeline, codename *Merlina*) and **abstracted**. It is an **architecture blueprint — patterns, contracts, and representative prompt text — not a copy of proprietary source.** No secrets, credentials, tokens, or internal endpoints appear here by construction.

**The specific-vs-generic tagging convention** (used throughout):

| Tag | Meaning |
|-----|---------|
| `[CORE]` | Reusable, domain-agnostic. Keep as-is; rename identifiers only. This is the gold. |
| `[SPECIFIC]` | Tied to the origin stack (the reference company / its cloud / its data). **Swap for your own equivalent.** |
| `[MIXED]` | A reusable *pattern* that carries specific identifiers or examples. Keep the pattern, replace the identifiers. |

**One honest boundary.** The pipeline's **data source** and **LLM inference** are reached through remote service adapters (in the reference system: MCP servers). This document covers them at the **interface + pattern** level (Section 7) — enough to reimplement an equivalent adapter — but not their server source code, which is not part of the factory proper and is trivially swappable.

---

## Table of contents

- **§0** How to read this — audience, the `[CORE]/[SPECIFIC]/[MIXED]` tags, the one boundary
- **Part I — Executive Architecture** · §1 the one big idea · §2 the layered model · §3 core invariants · §4 the deterministic core (the CLI) · §5 the orchestration state machine & contracts
- **Part II — Runtime substrate** · §6 agent runtime, distribution, workspace layout
- **Part III — Stage by stage** · §7 input boundary (the dataset) · §8 EDA & modeling design (the ML crown jewels) · §9 feature / dataset / model stages
- **Part IV — Reuse** · §10 the adapters · §11 the pattern catalog · §12 build-your-own recipe · §13 appendices (A templates + staging shapes · B sub-agent roster · C rename map · D glossary · E end-to-end worked example)

> **If you read only three things:** §1 (the thesis) → §11 (the 17 reusable patterns) → **Appendix E** (a toy dataset traced through every stage, with the real files). Then §4 + §5 for the contracts.

---

## Part I — Executive Architecture

### 1. The one big idea

A production ML factory is **not** "a script that trains a model." It is an **assembly line** with three cleanly separated responsibilities:

1. **An LLM orchestrates and judges** — decides *what* to do at each stage (which target, which features, which model families, is this artifact good enough?) and evaluates quality at gates. This is where ambiguity, context, and taste live.
2. **A deterministic CLI computes** — every reproducible operation (splitting, feature transforms, training, hyper-parameter search, metrics, validation) is a version-pinned command-line tool. **No numbers are ever "reasoned out" by the LLM.** This is where auditability and repeatability live.
3. **Contracts and meaning live as data on disk** — typed, versioned, lineage-tracked artifacts are the only interface between stages; semantic descriptions of the data live in manifests, not in any model's head or memory.

Everything else in this blueprint is an elaboration of that split. If you internalize one thing: **the LLM decides and evaluates; the CLI does the math; artifacts on disk are the API between stages.**

Why this split is the whole game:
- **Auditability / regulatory defensibility** — a regulator (or a skeptical reviewer) can re-run the CLI and get identical numbers. An LLM narrating an AUC is unauditable; a pinned CLI computing it is a unit-testable fact.
- **Reproducibility** — same inputs + same tool version ⇒ byte-identical outputs. Seeds are fixed; content is hashed.
- **Composability & testability** — narrow agents with narrow tools, deterministic tools with unit tests. You can test the factory the way you test software.
- **Human-in-the-loop by construction** — the guiding control principle is **"AI proposes; humans decide"** — every stage has an evaluator gate *and* interactive human checkpoints. For a high-stakes domain this is a requirement, not a nicety.

### 2. The layered model

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L6  DISTRIBUTION      bundle/topic packages + workspace-sync + stamps    │  [MIXED]
│      how the factory ships to a fleet; library & prompts version apart    │
├─────────────────────────────────────────────────────────────────────────┤
│  L5  ORCHESTRATORS     one "playbook" per stage (a long prompt program)   │  [MIXED]
│      EDA · Feature-Eng · Dataset-Gen · Model-Impl — own the control loop  │
├─────────────────────────────────────────────────────────────────────────┤
│  L4  SPECIALIST AGENTS  narrow sub-agents each orchestrator spawns         │  [MIXED]
│      designers/analyzers (judgment) + deterministic-tool wrappers          │
├─────────────────────────────────────────────────────────────────────────┤
│  L3  DETERMINISTIC CLI  the compute core (split/train/search/eval/valid.)  │  [CORE]
│      one binary, many subcommands, --json I/O, seeded, hashed              │
├─────────────────────────────────────────────────────────────────────────┤
│  L2  CONTRACTS & STATE  typed artifacts + lineage + a run-manifest FSM     │  [CORE]
│      frontmatter = inter-stage API; run.json = durable resumable state     │
├─────────────────────────────────────────────────────────────────────────┤
│  L1  ADAPTERS          data-source adapter + LLM-inference adapter         │  [SPECIFIC]
│      (reference: MCP servers — read-only prod DB + server-side Bedrock)    │
└─────────────────────────────────────────────────────────────────────────┘
        runs on ▶  an AGENT RUNTIME (reference: Claude Code / Claude Agent SDK)
```

- **L1 Adapters** `[SPECIFIC]` — how the factory reaches the world: a **data-source adapter** (yields a lineage-tracked dataset) and an **LLM-inference adapter** (server-side model calls for any in-pipeline LLM enrichment / design steps). Swap these first for your stack.
- **L2 Contracts & State** `[CORE]` — the typed artifacts (one per stage) + their lineage fields (**this chain is the ML pipeline's actual state machine**), and an optional **run-manifest FSM** for a resumable control plane. Note: the ML pipeline resumes off the *lineage chain*, not `run.json` — the manifest is substrate A (ETL/dashboard); see §5.0. _(detailed by Section 5.)_
- **L3 Deterministic CLI** `[CORE]` — the compute. _(detailed by Section 4.)_
- **L4 Specialist agents** `[MIXED]` — two archetypes: **designers/analyzers** (LLM judgment — e.g. target design, leakage scan, model-family ranking) and **deterministic-tool wrappers** (shell to exactly one CLI subcommand, parse one output, never retry). _(detailed by Section 6.)_
- **L5 Orchestrators** `[MIXED]` — one playbook per stage; owns the phase flow, spawns L4 agents, runs the closing validation gate. _(detailed by Section 6.)_
- **L6 Distribution** `[MIXED]` — how the whole thing is packaged and pushed to many workstations. _(detailed by §6.2.)_

### 3. Core invariants (the rules every part obeys)

These are the load-bearing conventions. A faithful port keeps all of them.

1. **Deterministic-tool boundary** `[CORE]` — anything numeric/reproducible is a CLI subcommand, never LLM free-text. Sub-agents that wrap it **never retry** (a deterministic failure fails identically) and **pass CLI errors through verbatim**.
2. **Artifacts are the API** `[CORE]` — stages communicate only via typed files on disk. Each artifact is a Markdown doc with a machine-readable **frontmatter contract** + a human-readable body. The next stage reads the frontmatter.
3. **Lineage on everything** `[CORE]` — every artifact records a content hash of its query/source (`query_sha256`), a `schema_hash` of its output columns, and a `parent` pointer (artifact + path + sha256). A closing gate **walks the lineage chain** and **probes the output**; failure triggers **delete-on-failure rollback** (no half-built artifact survives).
4. **Gate, then write** `[CORE]` — nothing lands until it passes. Validators **collect all findings** (never short-circuit on first failure) so the human sees the full picture.
5. **AI proposes, human decides** `[CORE]` — orchestrators pause at interactive checkpoints; sub-agents are "DS-leads": they **propose/execute but never override** the human's upstream choices unless an explicit override flag is passed.
6. **Meaning as data** `[MIXED]` — the semantics of the source data live in committed **manifests** (table/column descriptions, units, enums, PII flags, deprecation) the pipeline reads, so it doesn't guess or depend on tribal memory.
7. **Leakage kept out at the source** `[CORE]` — target leakage is the #1 failure mode of a loss/label model. The factory scans for it in EDA (§8.3), keeps leaky columns out as feature inputs, and re-checks it structurally at the dataset gate (§9.2.2). **Caveat a builder must get right:** true *fit-on-train* safety for stateful transforms (scaler / one-hot / impute) requires the train/val/test split to exist *before* features are engineered — the **split-before-EDA (Stage 2b)** path in §5.6. In the simpler single-frame flow, only CV-folded `target_encoding` is leakage-safe; the stateful transforms fit on the whole pre-split sample (§9.1).
8. **Prefer a multi-metric gate to any single number** `[CORE — aspirational]` — the **shipped** acceptance gate is *beat-the-baseline-by-a-margin on the primary metric*, plus a floor (§9.0.8). A stronger **multi-metric stability gate** (floor + train→holdout stability across several metrics) exists as a CLI op (`select-model`, §4.4) but is **not wired into the shipped Stage 6** — treat it as the recommended upgrade for a high-stakes port, not the default. Never let one lucky score decide.
9. **Reproducible by seed + hash** `[CORE]` — fixed seeds everywhere; content hashing detects drift; re-running yields identical artifacts.
10. **Multi-target is first-class** `[CORE]` — every stage handles one Y or many (`targets[]` + a `targets_relationship` of `parallel` vs `multi_output`) without special-casing.

### 4. The deterministic core (`<ml-cli>`) — the compute library

The single most important `[CORE]` component. In the reference system it is a Python package (`merlina_ml_tools`) exposing **one console script** (`merlina-ml`) with **18 subcommands**. Every pipeline stage is an LLM sub-agent that *shells out* to this CLI; **all real computation lives here as pure/deterministic Python** so runs are reproducible and auditable. **Sub-agents decide; this CLI computes.** Entry point: `[project.scripts] <ml-cli> = "<pkg>.cli:app"` (a `typer.Typer(no_args_is_help=True)`); `requires-python >= 3.11`.

#### 4.1 Determinism posture (the rules the whole library obeys) `[CORE]`
- **Seeds everywhere.** Every stochastic op takes `--seed` (default `42`). Splits use `numpy.random.default_rng(seed)` *exclusively* — no `random` module, no time-derived seeds. Every sklearn estimator's `random_state` is pinned; Optuna uses `TPESampler(seed=seed)`; GBM engines pinned + single-threaded for determinism (`n_jobs=1` / `thread_count=1`).
- **Byte-stable columnar output.** All writers use `write_parquet(path, compression="zstd", statistics=False)` → identical input+params ⇒ byte-identical output ⇒ stable `schema_hash`. Row gather via `pyarrow.Table.take` preserves schema exactly.
- **Content hashing.** `query_sha256` (sha256 of literal SQL bytes), `schema_hash` (sha256 over `"\n".join(f"{name}:{type}:{nullable}")`), per-split parquet-byte sha256, `input_sha256` of profiled files.
- **JSON-safety.** Viz builders serialize with `allow_nan=False` (refuse NaN/Inf that browsers reject); the progress emitter coerces non-finite floats to `null`.
- **Supply-chain hardening** `[MIXED]`: a dependency cooldown (`exclude-newer = "7 days"` — never resolve a version <7 days old) + a committed lockfile for frozen hashed installs.
- **Dependencies** (reference set, all non-optional): `typer, pydantic v2, pyyaml, polars, pyarrow, scikit-learn, optuna, numpy, pandas, matplotlib, shap, lightgbm, xgboost, catboost`. GBM engines + SHAP were deliberately promoted from an opt-in extra into **core** ("a silent forgot-the-extra import error was worse than the one-time install weight"). **Neural nets stay deferred.**

#### 4.2 The complete CLI command surface (18 commands) `[CORE]` (except where tagged)
Every command emits structured JSON errors to **stderr** and exits non-zero on failure; most take `--json` for a machine-readable stdout summary.

| # | Command | Key args | In → Out | Purpose |
|---|---------|----------|----------|---------|
| 1 | `baseline-train` | `--dataset --target-spec --baseline-spec --output --seed=42 --json` | train.parquet + specs → `baseline-<target>.pkl` + `baseline-result.yaml` | Train the EDA-declared floor model per Y (train only) |
| 2 | `hp-search` | `--dataset --val-dataset --target-spec --family --space --cv-strategy --max-trials=50 --max-time-seconds=3600 --seed=42 --output` (+ `--stability-penalty` `--min/--max-features` `--allow-non-converged-winner`) | train+val + HP-space + cv-strategy → `winner.pkl` + `hp-results.yaml` | Optuna TPE search over the EDA-bounded space for ONE family×Y |
| 3 | `evaluate` | `--model --test-dataset [--train --val] --target-spec --slices --output --explain/--no-explain` | model + splits → `eval-artifact.yaml` + PNGs | Held-out eval: metrics, calibration, slices, threshold sweep, residuals, cross-split stability, SHAP/PDP |
| 4 | `select-model` `[SPECIFIC]` · **UNWIRED** | `--candidates --target-spec --train-dataset --test-dataset [--val-dataset] --thresholds --slices --output --json` | candidates + splits → `selection-record.yaml` | Cross-family winner via the multi-metric **stability gate** + lexicographic rank. **Real + tested but NOT called by shipped Stage 6** (which does single-metric selection, §9.0.8); see §4.4 |
| 5 | `profile-dataset` | `--input --output [--rows]` | any dataset → `DatasetProfile` YAML | Per-column role/stats/missingness/cardinality (backs EDA column-profiler; reused in-process by `validate-artifact --probe-output`) |
| 6 | `split-dataset` | `--input --strategy --params --seed=42 --output-dir --json` | features parquet + params → train/val/test.parquet + `split-metadata.yaml` | Materialize leakage-safe splits |
| 7 | `engineer-features` | split: `--train --val --test --output-dir`; single: `--input --output`; `--spec --fit-params-out --apply-only` | parquet(s) + feature-spec → engineered parquet(s) + `fit-params.yaml` | Execute the deterministic transform catalog (fit-on-train, apply outward) |
| 8 | `propose-feature-selection` | `--input [--target] --variance-threshold=0.0 --corr-threshold=0.95 --corr-method=pearson` | parquet → `FeatureSelectionProposal` | Filter-tier (variance+collinearity) drop **proposal** (non-mutating) |
| 9 | `propose-supervised-selection` | `--input --target --task-type --family --method=importance [--cv-strategy --max-features] --seed=42` | model-ready parquet → `SupervisedSelectionProposal` | Tier-B leak-safe CV-fold importance/RFE ranking **proposal** (non-mutating) |
| 10 | `validate-artifact` | `<artifact.md> --walk-lineage --probe-output --json` | artifact `.md` → verdict; **exit 1 on any failure** | Schema + `parent.sha256` lineage walk + on-disk output probe + fit-param replay |
| 11 | `gen-model-card` | `--artifacts <model.md> --output` | model artifact (chain walked) → `model-card.md` | Render the 8-section markdown card from the validated chain |
| 12 | `export-schemas` | `--output-dir --check` | pydantic models → `*.schema.json`; `--check` exits 1 on drift | Emit/verify JSON-Schema exports of every artifact + tool model |
| 13 | `serve` | `--run --port=0 --host=127.0.0.1 --token/--no-token` | run-dir → read-only loopback web server | Live browser lens on a run (stdlib-only) |
| 14 | `validate-viz-spec` | `<bundle> --json [--columns]` | viz-spec bundle → report (0/1/2) | Deterministic schema + semantic (no-external-URL) viz check |
| 15–17 | `build-split-viz` / `build-feature-viz` / `build-model-viz` | `<artifact> --out --audience-mode` | stage artifact → `*.json` viz bundle | Deterministically render each stage's charts from frontmatter (+bounded parquet reads) |
| 18 | `build-notebook` | `<bundle> --out [--title]` | viz-spec bundle → `report.ipynb` | Re-derive an un-executed offline notebook from a validated bundle |

`[SPECIFIC]` in the surface: `select-model`'s gate (KS / rank-order-breaks / score-PSI) and `evaluate`'s `ks`/`rob`/decile-table blocks and `gen-model-card`'s regulator-compliance section are credit-scorecard specific. Everything else is generic ML.

#### 4.3 Per-module algorithms (key verbatim signatures) `[CORE]`
- **`split.py`** — 4 mutually-exclusive strategies (`random | stratified | time_aware | grouped`) + optional pre-split sampling (`full | downsample_majority | downsample_class_balance`). random=`rng.shuffle`+ratio; stratified=per-class shuffle+ratio (within-split order restored so only *which rows* is random); grouped=GroupKFold-style, no group spans splits, ≥2 groups; time_aware=`sort(time_col)` chronological cut or explicit ISO windows. Ratios default 70/15/15 (sum 1.0±1e-6); val optional.
- **`baseline.py`** — dispatch on `baseline.type` → `DummyClassifier("most_frequent"|"stratified")`, `DummyRegressor(mean|median)`, `logreg_3feat`→`LogisticRegression(solver="liblinear")` on top-3 features, `linreg`→`LinearRegression`; `km_estimator`→**raises `survival_deferred`**. Trains on **train only** = the floor.
- **`hp_search.py`** — `_ALLOWED_FAMILIES = {gradient_boosting, regularized_linear, random_forest, lightgbm, xgboost, catboost}` (NN/survival refused). CV: time_aware→`TimeSeriesSplit`, grouped→`GroupKFold`, stratified→`StratifiedKFold(shuffle, seed)`, random→`KFold` (n_splits=5). Optuna `create_study(direction, sampler=TPESampler(seed))`, `optimize(n_trials=max_trials, timeout=max_time_seconds)`. **Winner** refit on `vstack([train, val])`; **hard-fails `winner_did_not_converge`** on `ConvergenceWarning` unless overridden. Opt-in extras: **W2 stability penalty** (objective penalized by `λ·mean|train_fold−val_fold|`; reported metric stays raw); **W3 joint feature+HP search** (`n_features` as an Optuna dim; per-fold consensus on train-fold only; strict feature ceiling).
- **`evaluate.py`** — primary metrics per task (binary: auc/ap/log_loss/brier + `[SPECIFIC]` ks/rob; multiclass: macro_f1/accuracy/macro_auc; multilabel: per_label_f1_mean/hamming/subset_acc; regression: rmse/mae/r2 + ordinal kappa/spearman; survival raises). Calibration (ECE over 10 quantile bins + Brier + reliability PNG), threshold sweep (0.01→0.99, F1-max), residual plots, slices (drop `n<30`), **cross-split stability (W1)** `score_psi`+`metric_drops`, explainability (W4 SHAP+PDP, best-effort).
- **`stability.py`** (importable, no CLI) — PSI/CSI drift. **Core formula verbatim:**
  ```python
  def psi_from_proportions(expected, actual, *, eps=1e-4) -> float:
      """Σ (actual − expected) · ln(actual / expected) over two proportion vectors."""
      e = np.clip(np.asarray(expected, "float64"), eps, None)
      a = np.clip(np.asarray(actual, "float64"), eps, None)
      return float(np.sum((a - e) * np.log(a / e)))
  ```
  **Load-bearing correctness fix over legacy:** reference (train) bin edges computed **once** and reused to bin every comparison sample; outer edges `±inf` so out-of-range comparison values fall into the first/last bin instead of being silently dropped. (Legacy recomputed edges per dataset → drove PSI→0 even under real drift.)
- **`engineer.py` + `engineer_transforms.py`** — 10-transform registry, each a `(fit, apply)` pair: `log_transform`(log1p(x+ε)), `one_hot`(train-fit category set; unseen→all-zero), `standard_scaler`(train mean/std), `target_encoding`(train CV-folded to prevent self-leakage; val/test get full-train map), `date_parts`, `temporal_diff`, `drop_columns`, `impute`, `ratio`(x/y, division-safe), `interaction`(product of ≥2 inputs). **Fit-on-train / apply-outward** is the leakage-safe invariant. **Model-ready postcondition** (`_validate_model_ready`): every output column numeric/boolean, no nulls/NaN/inf. Learned params serialized to `fit-params.yaml` for audit + replay.
- **`profile.py`** — role detection order (first match): datetime → id_like → json → string(categorical if card<50 else text) → numeric → text. Reused in-process by `validate-artifact --probe-output` (no subprocess hop).
- **`feature_select.py`** (filter tier: variance + collinearity) / **`supervised_select.py`** (tier-B leak-safe CV-fold importance or RFECV) — both **non-mutating proposals**; the DS approves, Stage 4 executes via `drop_columns` (propose-then-lock).
- **`imputation.py`** — conservative-first: sentinel fallback for every role (leakage-safe, preserves the missingness signal); a data-fit strategy (median/mean/mode) only when missingness is uninformative + low.
- **`explain.py`** (W4) — SHAP (`TreeExplainer` for tree families; generic `Explainer` for linear/HistGB) + model-agnostic PDP (`method="brute"`). Best-effort; degrades to PDP-only if shap missing.
- **`serve.py`** — read-only, **loopback-only**, stdlib-only web lens. Invariant: *"the agent is the sole writer; the web app is the sole reader."* GET-only, path-jailed inside the workspace, constant-time token compare, tight CSP.
- **`risk_metrics.py` `[SPECIFIC]`** — credit-scorecard decile-table KS (not scipy `ks_2samp`), rank-order-breaks (ROB = count of bad-rate monotonicity violations), score-PSI, train→holdout drops. Deliberately reconciles with the Risk team's numbers.
- **`selection.py` `[SPECIFIC]`** — the multi-metric stability gate (see 4.4). Binary-classification only; non-binary routes to `best_metric`.

#### 4.4 The multi-metric stability gate (verbatim) — `[SPECIFIC]` shape, `[CORE]` principle, **NOT wired into shipped Stage 6**
> **Honest scope — read before relying on this:** this gate is the `select-model` CLI op — real and unit-tested — but the shipped Stage 6 (`/…-model-implement`) **never calls it**; it does single-metric-winner + beat-baseline-by-margin inline (§9.0.8). So the shipped pipeline *can* pass a model that never cleared this gate. To make acceptance actually require multi-metric stability, wire `select-model` into Stage 6 (replace or follow the inline selection) — recommended for a high-stakes port.

This gate is the *aspirational* form of invariant #8. The thresholds + KS/ROB centering are credit-specific; the *pattern* — a model passes only if it clears a floor AND is stable train→holdout across several metrics — is the reusable idea. Swap the metrics/thresholds behind a "profile" for another domain.

```python
@dataclass(frozen=True)
class Thresholds:
    auc_floor: float = 0.65
    ks_floor: float = 0.0            # off by default
    max_abs_auc_drop: float = 0.05
    max_abs_rel_ks_drop: float = 0.05
    max_score_psi: float = 0.2
    parsimony_first: bool = False    # W3 hook — fewest features as leading rank key

def gate(metrics, t) -> list[str]:   # returns FAILED criteria; empty ⇒ passes
    failed = []
    # AUC floor on ALL splits (train/test/val)
    if not metrics.train_auc > t.auc_floor: failed.append(f"auc_floor:train(...)")
    if not metrics.test_auc  > t.auc_floor: failed.append(f"auc_floor:test(...)")
    # (val if present)
    # |AUC drop| train→holdout <= max_abs_auc_drop  (test, and val if present)
    # |relative KS drop| train→holdout <= max_abs_rel_ks_drop
    # score-PSI < max_score_psi  (train→test, and train→val)
    return failed
```
**Rank + fallback:** survivors (empty failed-list) ranked lexicographically `[test_ks↓, val_ks↓, test_auc↓, val_auc↓, test_rob↑, val_rob↑, family↑]` (+`n_features↑` first when `parsimony_first`). **If NO candidate survives:** `fallback=True`, rank ALL by smallest relative-KS-drop and flag "review before shipping." `family` is always the final total-order tiebreak.

#### 4.5 `validate-artifact` — the lineage/probe engine (verbatim order) `[CORE]`
The linchpin of working-result enforcement. Two composable checks:

**`--walk-lineage`** walks the `parent.sha256` chain leaf→root; per upstream node, **in this deliberate order**:
```python
1. cycle check         # parent already seen → "cycle_detected" (before hashing — a cycle can't be honestly hashed)
2. file existence      # → "parent_file_missing"
3. sha256 drift        # sha256(parent.read_bytes()) != declared → "sha256_drift"
4. schema-validate the parent frontmatter against its pydantic model
5. parent.artifact type agreement   # → "parent_type_mismatch"
6. verification-status gate:  parent.verification.status == failed → "upstream_verification_failed"
```
The **leaf's own status is reported as-is**; the gate is on *upstream* artifacts (a failed upstream "poisons the chain"). Exit 0 valid / exit 1 with a structured `ValidationFailure(code, message, **details)`.

**`--probe-output`** (for artifacts with on-disk outputs): per declared parquet — (a) file exists, (b) footer-metadata row count == declared, (c) recomputed `schema_hash` == declared, (d) `profile-dataset(rows=1000)` runs cleanly. Plus **fit-param replay** (split-aware feature-specs): re-apply the serialized `transforms[].fit.params` via `engineer-features --apply-only` and confirm the recorded outputs reproduce (byte- or value-identical) — catches a tampered fit-param the row/schema checks would miss.

> **Where "delete-on-failure" lives:** the CLI only *exits non-zero* with a structured error. The **delete-on-failure rollback is the orchestrator's response** to that exit (each stage's closing gate), not code in the library. Keep that division: the deterministic tool reports; the orchestrator reacts.

#### 4.6 The artifact schemas (pydantic = source of truth) `[CORE]`
Pydantic v2 models are the source of truth; `export-schemas` emits JSON-Schema (CI-verified in sync). **Every artifact model is `extra="forbid"`.** Two registries: `ARTIFACT_MODELS` (the 6 lineage-tracked stage artifacts: exploration, saved-dataset, eda-exploration, feature-spec, dataset, model) and `TOOL_OUTPUT_MODELS` (profile, feature-selection-proposal, supervised-selection-proposal). All stage artifacts extend `ArtifactBase` (`artifact, version, stage∈1..6, created_at, created_by, audience_mode, parent?, input_mode, input_file?, verification, backtrack_signals[], caveats[]`) with a validator enforcing `from_artifact⇒parent`, `from_file⇒input_file`, `cold⇒both null`. (Full per-artifact field lists → Appendix.)

### 5. The orchestration state machine & contracts

#### 5.0 The single most important structural fact: **two orchestration substrates** `[CORE]`
The reference platform runs on **two distinct orchestration substrates that share one verdict/guardrail vocabulary.** A spin-off must understand both and pick (or keep both):

| Substrate | State lives in | Used by | Home |
|---|---|---|---|
| **(A) Control-plane run manifest** (`run.json`) | An explicit, durable JSON state machine: `stages[]` + `verdict` + `next_action` resume pointer — "a workflow with a conversational router." | The ETL-generation + dashboard pipelines | `.data-gents/runs/<slug>/run.json` |
| **(B) Artifact-frontmatter lineage chain** (NO run.json) | The state **is** the chain of on-disk artifacts — each stage's `.md` frontmatter carries `parent.sha256` + `verification.status`. Resume = "read the artifacts on disk." | **The data→model ML pipeline** (our subject) | `.data-gents/{datasets,features,dataset,model}/…` |

**The ML pipeline (`input → eda → feature → dataset → model`) is substrate (B)** — orchestrated by the lineage chain, *not* a run manifest. (The run-manifest's `stage_name` enum is `explore|implement|golden_lock|deploy|dashboard` — it does **not** contain the ML stages.) The two are complementary: *"The manifest sits on top of the lineage chain as the control plane; its `artifacts[]` registry **points at** the frontmatter artifacts by path+sha256; it never supersedes them. If they disagree, the artifact's own frontmatter + the `validate-artifact` lineage walk win on data correctness; the manifest is the index over them, not the truth of them."*

They share **one vocabulary** (below). A generic ML factory needs substrate (B) as the spine; add substrate (A) as a control plane if you want a resumable multi-stage front-door (the reference ML pipeline currently hand-chains stages via each stage's printed `Next:` hint rather than a manifest — concretely, each stage prints its output artifact path plus the exact next invocation, e.g. `Next: /…-feature-engineer --from-artifact <slug>-v1.eda-exploration.md`, which a human (or a wrapping script) runs; the next stage's Phase 0 re-validates the parent and projects its staging inputs from the parent's frontmatter).

#### 5.1 The shared verdict vocabulary `[CORE]` (both substrates)
```jsonc
verdict:  null | {
  "result":  "PASS" | "PASS_WITH_CAVEATS" | "REWORK",   // closed enum
  "findings": [ finding, ... ],                          // empty on clean PASS
  "score":    number | null                              // 0.0–1.0; advisory, never the gate
}
finding: { "severity": "blocker"|"risk"|"nit"|"info", "area": <free text>, "message": str, "fix": str|null }
```
**`result` is derived deterministically from the highest-severity finding** (so they can never disagree): any unresolved `blocker` → `REWORK`; else any `risk` → `PASS_WITH_CAVEATS`; else → `PASS`. **`score` is `null` when the gate is purely deterministic** (schema/lineage/row-count) and a real number only when an **LLM judge** ran — score is advisory (telemetry/tie-break), never routes.

#### 5.2 The stage-evaluator contract — where an LLM gate earns its place `[CORE]`
The rule hierarchy every gate obeys:
- **(a) Deterministic checks FIRST** — cheap, reproducible, no tokens (schema/lineage/row-count/`validate-artifact`/quality-checks). If a deterministic check can decide, it decides.
- **(b) An LLM judge ONLY for genuine judgment** — correctness-against-intent, PII reasoning, output-shape sensibility, generated-code review. Runs *on top of* the deterministic checks, never instead.
- **(c) Adversarial framing** — "assume the output is wrong until you've tried and failed to break it; when uncertain, default to REWORK."
- **THE BANNED ANTI-PATTERN — the rubber-stamp LLM judge:** never point an LLM at deterministic output and ask "does this look right?" (don't LLM-judge a `validate-artifact` exit code). *An LLM gate must add judgment the deterministic layer cannot provide, or it must not exist.*

For the ML pipeline (substrate B) the gates are **deterministic** (`verification.status ∈ {passed, partial, failed}`; the EDA's `validate-viz-spec` gate (`verification.method: viz_spec_validated`); the `dataset-validator` 6 checks; the feature-validator 4 invariants; the winner-beats-baseline margin). The `PASS/REWORK/PASS_WITH_CAVEATS + score` LLM-judge verdicts belong to the ETL/dashboard critics — **do not port them into the ML stages.** (But the *rule hierarchy* and *adversarial-verify* discipline are worth keeping if you add LLM gates.)

#### 5.3 The run-manifest schema (substrate A control plane) `[CORE]` shape / `[SPECIFIC]` stage names
One manifest per build, `additionalProperties:false` at **every** level. Top-level required: `version, run_id, slug, created_at, updated_at, created_by, audience_mode, intent, etl_destination, sources, stages, artifacts, decisions, open_questions, next_action, provenance`.
- **`stages[]` element** (the state-machine node): `{name: <stage_name>, status: pending|active|blocked|passed|passed_with_caveats|failed|skipped, verdict: <verdict>|null, started_at, ended_at, worker_command, iterations: int, notes}`.
- **`artifacts[]` element** (registry over the lineage chain): `{role, path, sha256, produced_by_stage, created_at, source_ref: int|null}` — append-mostly (re-producing appends a new entry with fresh sha256, never mutates history).
- **`next_action`**: `{stage, worker_command (FULL runnable invocation — authoritative for routing), hint}` — `null` **only** when terminal.
- **`decisions[]`** append-only `{at, stage, question, answer, decided_by: user|orchestrator|worker}`; **`open_questions[]`** (a `blocking:true` question puts its stage in `status:blocked`); **`provenance`** `{workspace_root, path_convention: absolute|workspace_relative, manifest_path}`.
- **`sources[]` is always an array** even for a single source (multi-source fan-in grows the array, no restructure).

#### 5.4 The stage graph `[CORE]` pattern
*"A well-understood fixed DAG — a workflow with a conversational router, not a free-roaming agent. The orchestrator's only decision is which outgoing edge to walk; it NEVER does data work."* Key behaviors: **terminals** (exploration-intent ends early, downstream `skipped`); **reuse short-circuit** (recon found an existing output → skip the build); **BLOCKED/awaiting-user** as a distinct outcome for the governed human gates; **stage-name UNIQUENESS** — *"the manifest is a state machine, so a stage is a node not a log of attempts; a REWORK loop mutates the existing element (flip status→active, bump iterations, overwrite verdict), never appends a duplicate node."* A sibling dashboard pipeline re-instantiates the identical machinery with a different stage-name/artifact-role enum — proof the pattern is generic (add a pipeline = new enums + reuse the worker/evaluator contracts unchanged).

#### 5.5 The worker contract `[CORE]`
A worker = one stage command. Standard return shape (a **view of what was just persisted**, not a second source of truth):
```jsonc
{ "result": "PASS|PASS_WITH_CAVEATS|REWORK",   // == verdict.result (mirrored top-level so the router needn't reach in)
  "verdict": { ... }, "next_action": { ... }|null,
  "artifacts_written": [ { role, path, sha256, produced_by_stage, created_at, source_ref } ] }
```
Plus a fourth **BLOCKED/awaiting-user** return (`result:null, verdict:null`, `next_action` re-enters this stage, a `blocking:true` open_question naming the decision the human owes) — distinct from REWORK (REWORK the worker fixes itself; BLOCKED hands a *decision* to a human).

- **The hard rule — persist-before-return:** a worker MUST persist ALL of {flip status + timestamps, write verdict, append artifacts, append decisions, set open_questions, set next_action, bump updated_at} to the manifest **in one write before returning**, so a crash between "did the work" and "returned" is recoverable. **Implementation: atomic write-temp-then-rename** so a crash mid-write never leaves a torn `run.json`.
- **The `--run <slug>` additive hooks** (how a standalone command becomes a manifest worker with **zero regression**): flag **absent** → behaves exactly as a standalone command, no manifest touched; flag **present** → runs the full normal command **and additionally** reads decided setup from the manifest (entry hook) and persists the outcome to the manifest before returning (exit hook). "Worker mode is additive recording only" — it never changes the artifact the command produces, and never injects `--force`/auto-approves a human gate.
- **REWORK bounding:** `stages[].iterations` counts orchestrator-level re-entries; **bound = 3 per stage.** Intra-worker loops (evaluator ×3, critic ×3, execute→repair ×4) stay inside the worker and do **not** bump `iterations`. On hitting the bound: only risk/nit/info remain → `passed_with_caveats`; a standing blocker → `failed` or a `blocking:true` open_question.

#### 5.6 The artifact lineage model (substrate B — the reproducibility spine) `[CORE]`
Every stage emits a **markdown file: YAML frontmatter = machine contract, body = human narrative.** Universal frontmatter (enforced by the `ArtifactBase` pydantic model, `extra="forbid"`):
```yaml
artifact: <type>            # exploration|saved-dataset|eda-exploration|feature-spec|dataset|model
version, stage(1-6), created_at, created_by, audience_mode
parent:                     # THE LINEAGE LINK — null at root
  artifact: <upstream type>
  path: "<abs or workspace-relative>"
  sha256: "<hex of the upstream FILE bytes>"
  version: "<upstream version>"
input_mode: from_artifact | from_file | cold
input_file: {path, format, sha256, row_count}   # when from_file
verification:               # THE GATE — downstream refuses status=failed
  status: passed | failed | partial
  method: deterministic_script | nbconvert_execute | manual | inline_probe | viz_spec_validated
  ran_at, execution_log, errors: []
backtrack_signals: []       # upstream re-entry requests
caveats: []
```
The **types chain**, each `parent` pointing at the prior (schema-enforced parent typing): `exploration(1) → saved-dataset(2) → eda-exploration(3) → feature-spec(4) → dataset(5) → model(6)`.

> **Leakage-safe variant — the split-before-EDA (Stage 2b) path (important; the base chain above omits it).** A `/…-split` producer can materialize the train/val/test split *before* EDA, emitting a `dataset` artifact with `stage: 2` (the schema's `stage` is `Literal[2, 5]` — 2b split vs. 5 finalize). The chain becomes `saved-dataset(2) → split dataset(2b) → eda(3) → feature(4, split-aware) → dataset(5, finalize) → model(6)`, and Stage 4 runs `engineer-features --train/--val/--test` writing `feature-spec.outputs:{train,val,test}` — so stateful transforms (scaler/one-hot/impute) genuinely fit-on-train and apply outward. **This is the path that realizes invariant #7; the single-frame path (split only at Stage 5) leaves those transforms fitting on the whole pre-split sample — only CV-folded `target_encoding` is leakage-safe there (§9.1).**

> **On `parents`:** the ML-stage artifacts (feature-spec / dataset / model) are single-`parent`. Multi-source fan-in — a `parents: list[Parent]` field — exists only on the upstream `SavedDatasetArtifact` (data-acquisition), where a joined dataset also sets `parent = parents[0]` so single-parent consumers keep working. **Frontmatter** = the YAML block between the first two `---` fences (the parse rule the whole "artifacts are the API" spine rests on). The Stage-1 `exploration` artifact is the chain root; its shape is a data-acquisition concern (out of scope here — treat `saved-dataset`, §7, as the effective input).

- **Deterministic fingerprints:** `query_sha256 = sha256(sql.utf8)` (no whitespace/comment normalization — a cosmetic edit bumps it); `schema_hash = sha256("\n".join(f"{name}:{type}:{nullable}"))` over the pyarrow schema (column order + nullability matter).
- **`validate-artifact --walk-lineage --probe-output`** is the closing gate (order per §4.5); a **failed upstream `verification.status` poisons the chain**.
- **Delete-on-failure rollback (working-result enforcement — the load-bearing guardrail):** the artifact is *provisional* until the closing gate passes; on non-zero exit, **delete the just-written files** so no poisoned artifact survives. Graduated scope: Stage 4 deletes only the `.md` (parquet kept for inspection); Stages 5/6 delete the full deliverable (`.md` + parquets/pickles/plots) because those have no standalone provenance. *"Delete-on-failure rollback is non-negotiable."* The gate may flip `passed → failed` **retroactively**.
- **`backtrack_signals[]`** `{target_stage(1-6), reason, suggested_changes, raised_by, raised_at}` — any sub-agent can request upstream re-entry (e.g. "Y is degenerate → target_stage:1"); the lineage-chain analogue of upstream-REWORK.
- **Versioned paths** `<topic>/<slug>-v<n>` — never overwrite without `--force`; a re-extract keeps the same `run_id` but writes a new `-v<n+1>`; the agent never deletes prior versions.

#### 5.7 The workspace taxonomy `[SPECIFIC]` names / `[CORE]` split
```
.data-gents/            # app runtime state — GITIGNORE ALL OF IT (machine-local paths + build state)
├── datasets/<topic>/<slug>-v<n>.parquet + .saved-dataset.md   # stage 2 (INPUT to the ML factory)
├── features/<topic>/<slug>-v<n>.parquet                       # stage 4 feature-spec output
├── dataset/<topic>/<slug>-v<n>/  (train|val|test.parquet + .dataset.md + split-*.yaml)  # stage 5
├── model/<topic>/<slug>-v<n>...  (*.pkl, eval-*.yaml, plots, model-card.md)             # stage 6
├── runs/<slug>/run.json          # substrate-A control-plane manifest (ETL/dashboard)
├── cache/<cluster>__<db>/schema.md   # introspected schema cache (TTL'd)
└── eda/.venv/                     # SHARED data-stage venv (whichever ML stage runs first bootstraps it)
```
Keep the split: an **installed-factory config dir** (versioned, wiped-and-replaced by the installer) vs. an **app-owned runtime-state dir** (durable, user-owned, gitignored).

---

## Part II — The runtime substrate

### 6. The agent runtime, distribution, and workspace layout

#### 6.1 The harness primitives the factory is built ON `[MIXED]`
The factory is **an application built on an agent runtime's primitives** (reference: Claude Code / the Claude Agent SDK) — not a standalone program. Four primitives carry the whole thing; a spin-off must choose an equivalent for each:

| Primitive (reference) | Concrete use | What it provides | Generic equivalent |
|---|---|---|---|
| **Slash command** = a Markdown playbook at `.claude/commands/<name>.md`, injected as instructions when invoked | each pipeline *stage* is one command (`/…-eda-explore`, `/…-model-implement`) | the per-stage orchestration program — **the playbook IS the program; there is no compiled orchestrator** | a named/parameterized **skill / prompt-playbook / graph-node** (SDK system-prompt injection), one per stage |
| **Sub-agent** = a definition at `.claude/agents/<group>/<name>.md`, spawned via the Task/Agent tool with its own **isolated context + tool set** | `target-designer`, `baseline-trainer`, `dataset-validator`, … | an isolated-context specialist with scoped tools + a structured (JSON) return contract; *"if a sub-agent file is missing, halt — do NOT inline"* | an **SDK subagent / spawned sub-task** with its own context + tool allow-list (or a plain function when the step is purely deterministic) |
| **MCP server** (`mcp__<server>__<tool>`) — **`[CORE]`, MCP is an open standard** | data-source, installer, model-gateway | read-only source access, distribution, model inference | **MCP as-is**; the *servers* are the swap points (§10) |
| **Workspace filesystem** (`.claude/` config + `.data-gents/` state) | installed prompts/library/stamps vs runtime datasets/venvs | separation of *installed factory* (versioned, wiped-and-replaced) from *app runtime state* (durable, user-owned) | a project dir with a config/skills dir + an app-owned state dir; keep the split |

Two design invariants worth preserving verbatim: **(a) the orchestrator never does specialist work inline** — it spawns a defined sub-agent and halts if the definition is missing (auditability); **(b) deterministic work is delegated to a versioned CLI library**, so agent reasoning never silently substitutes for a reproducible computation.

#### 6.2 The distribution system — "skills-as-bundles" `[CORE]` pattern
The whole factory (commands, sub-agents, shared specs, and the Python CLI) is delivered as **topic bundles** fetched from an installer service, extracted into the workspace, and **stamped** for idempotent re-sync.

**Source of truth → publish (reference `build-bundles.py`):** a monorepo holds `topics/<topic>.yaml` (a topic = `{name, description, required_mcps[], repos[], includes[]}`), `agents/<topic>/` (installable content where **source path == install path**), and `resources/mcps/<name>.yaml` (registration metadata). The build zips `agents/<topic>/` **deterministically** (fixed epoch, fixed perms, sorted entries → identical content yields identical bytes) and computes **`bundle_sha256 = sha256(zip bytes)` — this IS the version** (no semver). The one per-file transform (`FILE_REWRITES`): the runtime `pyproject.toml`'s path-dep on the CLI library is rewritten from the dev-tree path to the install-tree path (a `test_*` guards this seam so a moved dir fails CI before users hit a broken CLI). CI: a `validate` workflow (schema-check every yaml + dry-run the build + run the CLI's pytest + **`export-schemas --check`** to verify emitted JSON-Schema matches the pydantic source); a `publish` workflow (OIDC-assume a role, `cp` bundles+catalog to the object store, overwrite-on-publish).

**Fetch + install (the `/<topic>-workspace-sync` commands):**
1. Call the installer MCP's `get_topic_bundle(topic)` → `{presigned_bundle_url, bundle_sha256, repos[], mcps[]}` (or `check_for_updates({topic: sha})` → per-topic `changed` boolean).
2. **Idempotent no-op:** read the local stamp `.claude/.merlina/topics/<topic>.json`; if its `bundle_sha256` == the fetched one (and no `--force`), **skip extraction entirely** ("already at sha=…; skipping"). Content-sha is the whole idempotency key.
3. Download + unzip; **wipe the prior install** (precisely — see two shapes below); extract; **stamp**.
4. Wire deps: clone missing `repos[]` as workspace siblings; register missing MCPs (`claude mcp add …`).

**Two extract layouts / two stamp shapes** `[MIXED]`:
- **Shape A — agent-prompt bundle (1:1 into `.claude/`):** `commands/`, `agents/`, `shared/` extract verbatim to those paths; also rewrites a **managed `CLAUDE.md` block** between markers (content outside preserved). Wipe is precise: delete exactly the files in the prior stamp's `installed_files[]`. Stamp = `{topic, bundle_sha256, installed_at, installed_files[]}`.
- **Shape B — library bundle (topic-namespaced):** the Python CLI library extracts into `.claude/.merlina/topics/<topic>/` (NOT 1:1 — a `src/`+`pyproject.toml` tree under `.claude/commands/` would pollute the layout). Wipe = `rmtree` the whole install root. Stamp = `{topic, bundle_sha256, installed_at, install_root, file_count}`.

**Independent versioning `[CORE]`:** the **declarative layer (prompts/playbooks)** and the **deterministic layer (compute library)** version *separately, behind one installer* — ship a new CLI without touching a prompt, and vice-versa. Each has its own stamp + sha.

**Runtime pickup:** at run time each stage's `env-bootstrapper` runs `uv pip install -r pyproject.toml` into a per-stage venv; the rewritten path-dep points at the installed library, so the venv transparently picks up whatever version the last library-sync stamped. **The installer MCP itself is a thin read-only stateless facade** over the object store (owns no state; 60s catalog cache; mints presigned URLs; strips unsigned keys).

#### 6.3 What a generic version keeps vs. swaps
- **`[CORE]` keep:** the four-primitive split (playbook-per-stage, isolated sub-agents, MCP tools, config-vs-state dirs); content-sha idempotent bundles; two extract layouts; independent library/prompt versioning; publish-time path-rewrite + runtime path-dep pickup into isolated envs; the never-do-specialist-work-inline + delegate-determinism invariants.
- **`[SPECIFIC]` swap:** the runtime (Claude Code → your SDK), the object store + presign transport, `uv`/pyproject specifics, `claude mcp add`, the concrete server names/endpoints, and the `.claude/` / `.data-gents/` names.

---

## Part III — Stage-by-stage deep dive

### 7. Input boundary — the dataset that arrives

The ML factory's **input is one artifact: the `saved-dataset`** (produced by a data-acquisition stage that is out of scope here — it streams a query to a versioned parquet). Treat its frontmatter as **the single stable interface** between "data acquisition" and "modeling." Verbatim template:

```yaml
---
artifact: saved-dataset
version: "1.0"
stage: 2
created_at, created_by, audience_mode
parent: <parent block — null when input_mode=cold>   # lineage pointer to the exploration artifact
input_mode: <from_artifact | cold>
input_file: null
verification:
  status: <passed | partial>          # NEVER written unless the output was read back + gates passed
  method: deterministic_script
  ran_at, execution_log, errors: [...] # non-empty only when status=partial
extraction:                            # [SPECIFIC] — generalize to "source"
  query_sha256: "<hex>"                # identity of the exact query that produced this data
  ran_at, rds_cluster, rds_database, rows_extracted, duration_seconds
output:
  path: "<absolute output path>"       # WHERE THE BYTES ARE
  format: <parquet | feather>
  size_bytes, partition_keys: [...]    # non-empty ⇒ Hive-partitioned directory
  schema_hash: "<hex>"                 # stable fingerprint; downstream re-checks for drift
schema:
  columns:
    - {name, dtype, nullable, manifest_annotation: null}   # annotation intentionally null here (EDA enriches it)
freshness: <block | null>              # {source_max_event_at, extraction_lag_seconds, sla_seconds, meets_sla}
backtrack_signals: []
caveats: [...]
---
```

**Field semantics that matter to the consumer (EDA):** `output.path`+`format`+`partition_keys` say where/how to read; `output.schema_hash` lets a downstream stage detect drift without diffing bytes; `schema.columns[]` is the column contract (`manifest_annotation` is **always null at this stage by design** — semantic enrichment is deferred to EDA's column-profiler); `verification.status ∈ {passed, partial}` — **the artifact is never written unless the output was read back and the row-count + schema gates passed** (a `failed` status halts and writes nothing).

**Generic "dataset handoff contract"** (rename tenant nouns, keep the shape): `parent`/`input_mode` (lineage), `verification` (working-result gate), `output` (path/format/size/partition/schema_hash), `schema.columns[{name,dtype,nullable,annotation}]`, `freshness?`, `caveats`. Generalize `extraction.rds_cluster/rds_database` → an adapter/connector id; make `dtype` an adapter-neutral vocabulary (not polars-locked); drop `audience_mode` if you don't want a UX styling axis.

**The materialization pattern behind it** (a pluggable data-source adapter — see §10.1 for the interface): stream rows via an incremental columnar writer (single-shot first; on timeout fall back to **keyset pagination** on a monotonic key — *never OFFSET*); on mid-stream failure, close the writer then **delete the partial file** ("only complete extracts leave a file on disk"); compute `query_sha256` over literal SQL bytes + `schema_hash` over the on-disk column fingerprint; **verify (read-back + row/schema gates) BEFORE emitting the artifact**, delete-on-failure. A cost gate estimates scan size *before* any row is pulled and can refuse-with-narrowing, so "proving it works" never means "it hammered production."

> A generic ML factory should treat this saved-dataset frontmatter as the seam where *any* data source plugs in — a SQL warehouse today, a feature store or object-store parquet or an API extract tomorrow. Everything downstream depends only on this contract, not on how the bytes were produced.

### 8. Stage 3 — EDA & modeling design (the ML crown jewels)

The one **continuous-session** stage (all others are transactional). It profiles the data and — in modeling mode — designs the target, scans for leakage, ranks model families, derives HP search spaces, and writes the baseline the training stage must beat. This is where the genuinely reusable ML IP is concentrated.

#### 8.1 Orchestrator shape `[CORE]`
- **Two modes, asked up front:** `descriptive` (profiles + relationships + narrative) vs `modeling` (adds target design, leakage scan, dimensionality, family ranking + HP spaces + baseline + split recommendation).
- **Four input entry shapes** (`input.oneof`): `saved-dataset` (canonical — skip sampling, the file *is* the sample), `file` (DS-supplied parquet/csv/…), `table-ref` (ad-hoc, cold), legacy `data-exploration`. Dispatch by frontmatter `artifact:` → extension → `<topic>.<table>` regex.
- **Phase graph:** `0 env-bootstrap → 1 source-resolve → [2 target-design (modeling)] → 3 sample-design → 4 extract → 5 column-profile → [6 json-handle] → 7 analysis fan-out → 8 interpret + notebook-compose → 8.5 notebook-EXECUTE (verify) → 9 write artifact (closing validate gate) → 10 continuous session`. In modeling mode Phase 7 runs `target-analyzer + relationship-analyzer + dimensionality-analyzer` **in parallel**, then `model-design-recommender` **after** all three return.
- **Multi-Y is first-class everywhere:** a locked `targets[]` list + `targets_relationship ∈ {null, parallel, multi_output}` threads through every analyzer (per-Y loop for null/parallel, joint pass for multi_output). The old single-Y block was dropped with no compat shim.
- **The self-healing notebook loop `[CORE]`** (the canonical "generate → verify → regenerate ×3" engine): `notebook-composer` writes `report.ipynb`; `notebook-executor` runs it end-to-end via `nbconvert --execute`; on a cell error it appends the structured traceback to `prior_errors[]` and re-spawns the composer (max 3); kernel/launch errors escalate immediately (the composer can't fix those). *"The composer can produce code that looks right; only execution proves it is."* — the single most important reusable pattern in the whole factory: **LLM-generated code is not trusted until a deterministic executor runs it.** (Note: in the shipped version the *recorded* EDA gate is a deterministic `validate-viz-spec` check on a Plotly-JSON bundle — `verification.method: viz_spec_validated` — which replaced the older nbconvert kernel gate; this notebook loop remains a build-time self-heal, not the gate of record. The generate→execute→regenerate *pattern* is what to port.)

**Fidelity note:** every EDA sub-agent's frontmatter declares only `name` + `description` — no `tools:`/`model:` override, so each inherits full tools + the parent model. They are narrow **by prompt**, not by tool-restriction. (For a port, consider pinning cheaper models to the deterministic wrappers and stronger ones to the crown-jewel designers.)

#### 8.2 `target-designer` — designs Y `[CORE]` logic / `[SPECIFIC]` domain-prior tables
The only interactive modeling sub-agent. **DS-leads posture (verbatim):** *"You do not auto-propose Y. You ask the DS to describe Y or ask for suggestions. You follow. You only generate proposals … when explicitly invited."* Output is a rich `targets[]` schema:
```yaml
targets:
  - column, goal, task_type: binary_classification|multiclass|multilabel|regression|ordinal|survival
    construction_method: single_column | rule_based_boolean | weighted_composite | latent_factor
                       | survival_pair | snorkel_weak_supervision | multi_step_pipeline
    derivation_steps:                 # ordered; final step's output_column IS Y
      - {id, type, name, inputs, params, sql|null, output_column, probe:{health, distribution_summary}}
    validation: {probe_rows:<=500, distribution_summary, health: ok|degenerate}
targets_relationship: parallel | multi_output | null
feature_sufficiency_audit: {...}      # opt-in only (Step 5)
backtrack_signals: []
```
- **The SQL-vs-Python tier split** (a load-bearing decision table): SQL tier = `select_columns/filter/small_join/small_aggregate/narrow_window/sql_column/sql_cte` (single SQL expression on projected columns, affected rows stay < ~10M); Python tier = `python_transform/aggregate/window_compute/threshold/composite/derive` (depends on a prior Python step, needs a library, or would be a heavy join). **Hard rule:** *"A SQL step that comes after a Python step is NOT supported (no way to push a venv DataFrame back to the DB)."*
- **construction_method dispatch:** passthrough→`single_column`; rule→`rule_based_boolean` (SQL CASE); weighted combine→`weighted_composite` (`(Σ wᵢ·COALESCE(cᵢ,0))/Σw`, regression); latent→`latent_factor` (≥3 numeric constituents, Cronbach's α then 1-component FactorAnalysis, warn α<0.6); "default + time-to-event"→`survival_pair` (`event_observed` + `t_to_event`); "vote across N noisy rules"→`snorkel_weak_supervision` (2–5 labeling functions + Snorkel LabelModel); recipe→`multi_step_pipeline`.
- **Validation probe health (never silently accept a degenerate target):** per-step — numeric degenerate if `std=0` or `null_pct>50%`; boolean degenerate if any class 100%/0%; multi-valued degenerate if cardinality 1. Final-Y health per task type (binary: reject if any class 0% or 100%, warn minority <1%; regression: reject std=0, warn >50% null or kurtosis>5; survival: warn censoring>90%, reject event rate 0/100%; etc.).
- `[SPECIFIC]`: the Step-2 operationalization proposal tables (`is_on_time`/`is_30dpd`/`days_late`/survival-pair…) and Step-5 feature-sufficiency categories are lending-specific — **swap the domain priors**; the *structure* ("5–8 diverse operationalizations, opt-in; domain-prior coverage audit scoring covered/partial/missing") is reusable.

#### 8.3 `target-analyzer` — leakage detection + split recommendation `[CORE]`
Runs on the full sample. **Leakage detection (verbatim rules — the load-bearing ML correctness logic):**
- **Perfect predictors** `|corr| > 0.99` → `kind: perfect_predictor, recommendation: drop` ("computed from the target, contains it verbatim, or encodes posterior state").
- **Near-perfect** `0.9 ≤ |corr| < 0.99` → `kind: near_perfect, recommendation: inspect` ("verify it's observable at prediction time").
- **Posterior information** (manifest-aware) — a top feature whose manifest says `mutability: post_event` / "only known after disbursement/settlement" → `kind: posterior_info, drop`.
- **Derived-from-target** — a highly-ranked feature that's a target constituent (recovered from `derivation_steps[*].inputs`) under another name → `kind: derived_from_target, drop`.
- **ID correlated with target** `> 0.05` → auto-increment ID time-leaks → `kind: id_correlated, safe-with-caveat` (drop the ID, use a time-aware split).
- *"Don't auto-drop anything — only surface. The DS / model-designer makes the call."*

**Feature↔target ranking** dispatches by pair-type (point-biserial / correlation-ratio η² / Spearman / Cramér's V), skipping id/text/json/target/constituent columns, top-30.
**Split recommendation (verbatim dispatch — always shared across Ys):** ANY Y time-dependent + a datetime col (`null_pct<0.1`, `range≥90d`) → **time_aware**; else ANY Y has a grouping col with rows-per-group>1 → **grouped** (random would leak entities); else ANY classification Y with `minority<0.2` → **stratified** on the most-imbalanced Y; else **random 70/15/15**. Time_aware default: first 70% of range = train, middle 15% = val, last 15% = test.

#### 8.4 `model-design-recommender` — family ranking + HP spaces + baseline `[CORE]`
Pure synthesis (no MCP, no Python exec), after the three analyzers return. Emits `recommended_model_families` + `hp_search_spaces` + `baseline_spec` (+ optional refined `cv_strategy`).
- **The family-universe lock (verbatim):** *"every family you emit maps to a concrete `sklearn.*` class. No xgboost/lightgbm/catboost. No neural networks. When a task type has no clean sklearn home (survival; partially ordinal/multilabel), surface it as a caveat and downrank or refuse rather than paper over the gap."* (The *lock* is a v1-simplicity decision; the reusable pattern is "pin a family universe and refuse to silently escape it.")
- **Family ranking dispatch** (per task type, with EDA-driven adjustments): binary/multiclass default `gradient_boosting(HistGB) > regularized_linear(LogReg) > random_forest`, but **`n<5000` swaps ranks 1↔2** (small sample favors linear), high-dim-low-intrinsic bumps GBM, clean class separation keeps GBM, overlap bumps linear, severe collinearity annotates linear (l2/elasticnet). Multilabel → RF native #1. Regression → HistGBR > Ridge/Lasso/ElasticNet > RFR. **Ordinal** → regression-on-ranks + a warning. **Survival** → refuse (empty families + a `severity: error` caveat pointing at scikit-survival as v2, or convert to a fixed-horizon binary) — *"Do not silently emit a regression baseline as a fallback."*
- **HP space derivation (verbatim locked rule):** *"Every param's `reasoning` field MUST cite the specific EDA finding that produced the bound … reasoning lives INSIDE each param spec, never as a family-level sibling key."* Bounds scale with `n` and EDA facts (e.g. HistGB `max_iter` band by row count; `l2_regularization` widened to 1.0–10.0 under severe collinearity; `class_weight:["balanced",null]` only when `minority<20%`). Explicit anti-pattern flagged: *"Skip `scale_pos_weight` — that's xgboost terminology; the sklearn-correct knob is `class_weight`."*
- **Baseline derivation** (the floor stage 6 must beat): binary → `logreg_3feat` (AUC ≈ 0.5 + 0.5·mean(top-3 |strength|), clamped [0.55,0.85]) or `stratified_dummy`/`majority_class`; regression → `median_predictor` (skew>1) else `mean_predictor` (rmse=std(y)); etc.
- **CV refinement** — pass through the analyzer's recommendation unless one of three explicit override rules fires (severe-collinearity+small-sample; class-imbalance-under-random; PCA-low-intrinsic-dim); every override carries a `reason`.

#### 8.5 The analyzers + helpers (briefly) `[CORE]`
`column-profiler` (dtype inference rules in order: json_native → datetime → id_like → categorical → text → numeric; per-role stats; manifest-annotation merge; flags JSON cols). `relationship-analyzer` (X↔X: Pearson/Spearman/Cramér's V/η² by pair-type; `|corr|>0.85` → multicollinearity candidates; missing-data co-occurrence via Jaccard). `dimensionality-analyzer` (VIF>10 severe / 5–10 high; PCA components-to-90%; UMAP/t-SNE 2D projection colored by Y with silhouette; "fit once, color K times"). `interpreter` (narrates findings into the artifact body; quotes the recommender's per-param `reasoning` verbatim in technical mode; ≤~600 lines, no headers deeper than h3). `sample-designer` (file-loader vs interactive DB-sampling by size band). `json-column-handler` (flatten/JSONPath/shape-summary, one level). `env-bootstrapper` (7 sequential fail-fast checks, each with a copy-paste fix). Runtime deps pinned in a `pyproject.toml` (polars/scikit-learn/statsmodels/pingouin/umap/plotly/nbformat/nbconvert…) with optional `[survival]`/`[snorkel]` extras installed on demand. **(This EDA-agent runtime venv is a *separate* dependency closure from the `<ml-cli>` library's own install in §4.1 — two intentionally distinct environments; don't merge them.)**

#### 8.6 The `eda-exploration` output contract `[CORE]`
Not a full inline template — defined by the `EdaExplorationArtifact` pydantic schema (`extra="forbid"`). Load-bearing fields downstream stages read: `mode, audience_mode, parent, input_mode, verification, sample{path,row_count,column_count,manifest_coverage}, targets[], targets_relationship, feature_candidates[], leakage_risks[], feature_sufficiency_audit, cv_strategy, recommended_model_families, hp_search_spaces{per_target.spaces[family][param]=HpParamSpec}, baseline_spec, recommended_features, backtrack_signals[], caveats[]`. Analyzer findings (`columns_profile`, `relationships`, `target_analysis`, `dimensionality`) are **session-state only, NOT persisted** — the body narrative carries them for humans. The EDA's **verification-of-record** is a deterministic `validate-viz-spec` check on the emitted Plotly-JSON bundle (`verification.method: viz_spec_validated`, which replaced the older `nbconvert_execute` kernel gate). Save gate: `validate-artifact --walk-lineage --json` (the write lands on disk; only validation decides whether it "counts"). **The full `eda-exploration` frontmatter template is in Appendix A.**

### 9. Stages 4–6 — Feature Engineering, Dataset Generation, Model Implementation (the sub-agent layer)

> **What this is.** Nine sub-agents backing three stages of the pipeline. Each stage has a **single orchestrator** (`/feature-engineer`, `/dataset-generate`, `/model-implement` `[SPECIFIC]` command names) that owns the control loop; the sub-agents below are the *workers* it spawns. All nine follow one of two archetypes: **deterministic-tool wrappers** (shell to one CLI subcommand, parse one output, return one structured block, never retry) or **in-process deterministic checkers** (compute pure functions over parquet, return structured pass/fail, never write files).
>
> **Fidelity note.** In the reference system every one of these nine files declares only `name` + `description` in frontmatter — **none** overrides `tools:` or `model:`, so each inherits full tools + the default model. Carry that into a rebuild: these workers are narrow by *prompt*, not by tool-restriction.

#### 9.0 Cross-cutting sub-agent patterns (these repeat in every worker)

**9.0.1 The "deterministic-tool wrapper" pattern** `[CORE]`
Six of nine wrappers cite one contract — **"no retry, structured pass/fail, orchestrator owns the loop"** — with these rules:
- **No retry.** Justification quoted from the reference: _"The CLI is deterministic — a failure on attempt 1 will fail identically."_ / _"retries hide flakiness behind layers."_
- **One spawn = one CLI = one result.** _"You invoke ONE CLI for ONE family on ONE target ... parse ONE output YAML, and return ONE JSON block."_
- **Orchestrator owns iteration** — the sub-agent never loops over targets/families/transforms; the orchestrator's phase loop does.
- **Verbatim error pass-through** — capture _"the last ~80 lines of stderr verbatim"_ into `error.stderr_tail`; never rename CLI error codes.
- **No mutation of inputs** — consume staging YAMLs / parquets / spec read-only.
- **Structured JSON return** to the orchestrator + a short mode-styled chat note.

**9.0.2 `venv_python` isolation** `[MIXED]` (pattern CORE, path SPECIFIC)
Every CLI-wrapping sub-agent receives `venv_python` and MUST invoke through it — _"All `<ml-cli>` invocations MUST go through this — never via a globally-installed one on PATH."_ The CLI is always launched as a module: `<venv_python> -m <pkg>.cli <subcommand> ...`. The venv is resolved once in the orchestrator's Phase 0 by an `env-bootstrapper`.

**9.0.3 DS-leads posture** `[MIXED]`
The agent **proposes/executes but does not override** the human's ("DS" = data scientist) upstream choices: proposer _"proposes, doesn't decide"_; split-designer _"does NOT auto-override the EDA recommendation unless an explicit `--split-strategy` override is passed"_; hp-search-runner consumes the HP space _"verbatim. You do not widen, tighten, or re-shape the space."_

**9.0.4 Working-result enforcement** `[CORE]` — no artifact lands until every gate passes; failures trigger **delete-on-failure rollback**.

**9.0.5 No short-circuit on validators** `[CORE]` — validators **collect ALL findings**. dataset-validator: _"Do NOT short-circuit on the first failure. Run all 6 checks; surface them all."_

**9.0.6 Mode-styled output** `[CORE]` — every sub-agent takes `audience_mode: technical | non-technical` (chat-only; structured return is mode-agnostic).

**9.0.7 The stage graph & artifact chain** `[MIXED]`
```
Stage 4 Feature Eng.  → feature-spec artifact (output.schema_hash is the contract)
Stage 5 Dataset Gen.  → dataset artifact (splits + quality_checks[] + split_strategy)
Stage 6 Model Impl.   → model artifact (baseline + hp_search + evaluation + model_card)
```
Each stage closes with `<ml-cli> validate-artifact --walk-lineage --probe-output` (orchestrator-run) with delete-on-failure rollback. Artifacts live under `<workspace>/<stage>/<topic>/<slug>/` `[SPECIFIC]` path convention.

#### 9.0.8 The three orchestrators — flow, gates, and the transactional contract `[CORE]`
All three are **transactional — no continuous session** ("one invocation = one execution = one artifact; re-doing = a fresh invocation with a version bump"). Each: `Phase 0` env-bootstrap + parse (`--from-artifact` XOR `--from-file`/`--from-files`) → `Phase 1` input-resolution (read parent frontmatter, require `verification.status ∈ {passed, partial}` — **refuse `failed`** — compute `parent.sha256`; **Stages 5/6 additionally run the upstream `validate-artifact --walk-lineage --probe-output` gate at entry, and walk the lineage chain to the `eda-exploration` grandparent** to pull `cv_strategy`/`targets`/`recommended_model_families`/`hp_search_spaces`/`baseline_spec`) → middle phases (spawn the sub-agents) → closing `validate-artifact --walk-lineage --probe-output` gate with **delete-on-failure rollback** → summary + exit. **Stage 6 additionally validates the EDA ran in *modeling* mode and halts otherwise.** `input_mode: cold` is NOT a valid path for stages 4/5/6 (only stage 3).

**Stage 6 winner selection (verbatim, load-bearing ML logic).** After baseline (floor) + per-(family×Y) HP search, the orchestrator picks a winner per Y on the validation metric, gated on beating the baseline by a margin:
```
maximize: auc, average_precision, accuracy, f1, macro_f1, r2, spearman, quadratic_weighted_kappa, c_index
minimize: rmse, mae, log_loss, brier, hamming_loss          # (missing metric ⇒ default maximize + a caveat)
relative_improvement = (winner − baseline)/abs(baseline)     # sign flipped for minimize
gate: every Y must have relative_improvement >= --min-improvement   (default 0.05)
```
Then held-out `evaluate` per Y, `gen-model-card`, assemble the `model` artifact, and **hand off (no auto-PR — `approval` stays null until the DS opens the PR manually)**: the model card is the human go/no-go gate.

**Rollback scope graduates by stage** (from the closing gate): Stage 4 deletes only the `.md` (keeps the parquet for inspection); Stage 5 deletes `.md` + split parquets; Stage 6 deletes `.md` + every model pickle + every plot + eval artifacts ("they're a single deliverable; the evals reference the now-orphaned winner pickle and would mislead if left"). The full verbatim output frontmatter templates for `feature-spec`, `dataset`, and `model` are in **Appendix A**.

---

#### 9.1 STAGE 4 — Feature Engineering

> **Which leakage-safety you get depends on the chain (§5.6).** On the **split-before-EDA (Stage 2b)** path, Stage 4 runs `engineer-features --train/--val/--test`: stateful transforms (`standard_scaler`, `one_hot`, `impute`) fit on the **train split only** and apply outward, with the learned fills/means serialized into `feature-spec.outputs` — genuinely leakage-safe fit-on-train. On the **single-frame** path (features engineered on the whole sample, split deferred to Stage 5), only `target_encoding` (internally CV-folded) is leakage-safe; the other stateful transforms fit on the full pre-split sample (test rows included). Prefer the split-aware path for any real model.

Orchestrator owns the loop; three sub-agents: a proposer (opt-in), a validator (gate), an executor (CLI wrapper).

##### 9.1.1 `feature-proposer` `[MIXED]`
**Role:** Opt-in feature-transform *proposal* agent. **Never auto-fires** — spawned only on explicit request or by the DS. Reads EDA findings, drafts a `transforms[]` YAML, stops. _"You propose, you do not lock."_ **Invokes no CLI, touches no DB** — reads EDA structured findings + the sample-parquet head and writes a draft YAML.

**Input:** `eda_artifact_path`, `sample_parquet_path`, `feature_candidates[]` (`{column, dtype, role, missingness, cardinality}`), `leakage_risks[]` (`{target, column, strength, kind, recommendation, reason}`), `caveats[]`, `targets[]` (+ task_type), `cv_strategy` (`{type: time_aware|grouped|stratified|random, cv_folds}`), optional `recommended_features`, `audience_mode`.

**Output:** a draft YAML (not the final artifact — the orchestrator writes that after the DS locks it):
```yaml
transforms:
  - id: 1
    name: "<short, human-readable>"
    type: "<one of the registry keys>"
    inputs: ["<col>", ...]
    params: { ... }
    output_column: "<col>"        # or output_columns: [...]
    rationale: "<one-line, mode-styled>"   # NOT part of the schema — stripped before engineer-features
```

**The v1 TRANSFORM_REGISTRY (closed set)** `[CORE]` — 8 allowed transform types:
`log_transform | one_hot | standard_scaler | target_encoding | date_parts | temporal_diff | drop_columns | impute`. Explicitly out of v1 scope: _"no `aggregate_window`, `interaction`, `bin`"_; no transforms for `text`/`json`/`id_like` roles (emit a caveat instead).

**Proposal heuristics (the load-bearing decision table)** `[MIXED]` (logic CORE, column examples SPECIFIC):

| Column profile | Suggested transform | Rationale |
|---|---|---|
| `numeric` AND skew ‖>2‖ (derive from mean vs median vs p95 gap) | `log_transform`, `epsilon: 0` if min>0 else small offset | compress skew |
| `numeric` AND in a high-VIF pair | `standard_scaler` | helps regularized linear on collinear pairs |
| `categorical` AND cardinality ≤ 10 | `one_hot` (`prefix:<col>`, `drop_first:false`, `drop_source:true`) | low-card cats expand cleanly |
| `categorical` AND 10 < card ≤ 1000 | `target_encoding` (`target:<Y>`, `cv_folds:<from cv_strategy or 5>`, `smoothing:10`) | high-card cats blow up one-hot |
| `categorical` AND card > 1000 | NOT a transform; caveat + skip | feature-store concern |
| `datetime` AND range_days>30 | `date_parts` (`[year,month,day,dow]`; +`week_of_year` if >365) | surface seasonality |
| `datetime` AND a sibling `*_at` exists | `temporal_diff` (`unit:days`) | gaps > raw timestamps |
| `id_like` | `drop_columns` | IDs leak |
| `text`/`json` | `drop_columns` or caveat | out of v1 |
| column in `leakage_risks[]` w/ `recommendation:drop` | `drop_columns` | enforce leakage list at Stage 4 |
| `numeric` w/ missingness AND model needs no-NaN | `impute` (`median` numeric / `mode` discrete) | LR+RF need dense; HistGB tolerates NaN |

**Multi-Y rule** `[CORE]`: `parallel` → `target_encoding` uses `targets[0]` (+ note to add a sibling for other targets); `multi_output` → **skip `target_encoding`** + caveat.
**Leakage refusal (load-bearing)** `[CORE]`: any column in `leakage_risks[]` with `recommendation:drop` is **always refused** as a transform input.

##### 9.1.2 `feature-validator` `[CORE]` (with SPECIFIC import paths)
**Role:** Per-transform **probe gate**. For each transform in the locked spec, runs it against the **first 500 rows** of `sample.parquet` and checks four invariants. _"Probe failures land as `probe: health: degenerate` and halt the orchestrator's Phase 3."_ It does **not** re-implement transforms — it dispatches through the *same registry the executor uses*, in-process:
```python
from <pkg>.engineer_transforms import TRANSFORM_REGISTRY, TransformError
from <pkg>.schemas.feature_spec import FeatureTransform
spec = FeatureTransform.model_validate(t)
df_after = TRANSFORM_REGISTRY[spec.type](df, spec)
```
_"Do NOT re-implement transforms. Dispatch through the registry so the probe stays in sync with the executor."_

**Probe slice:** `df = pl.read_parquet(sample_parquet_path).head(500)`. Empty → `status: error, kind: sample_unreadable`. `<50` rows → caveat but continue.

**THE FOUR INVARIANTS (verbatim):**
> **Check 1 — output columns appear**: `new_cols` non-empty AND match `spec.output_column` (single) / `spec.output_columns` (multi). Mismatch → `degenerate`, `reason:"expected output columns <declared>, got <produced>"`.
>
> **Check 2 — no NaN explosion**: for each new col, `null_rate = null_count/df.height`; required `new.null_rate <= input_null_rate + 0.10` (10% absolute headroom). Multi-input (`temporal_diff`) → baseline is `max(input_null_rate)`. Multi-output (`one_hot`,`date_parts`) → per new column. Breach → `degenerate`.
>
> **Check 3 — non-degenerate output**: `n_unique(new_col) > 1` (single-valued columns are useless). Exception: `one_hot` `drop_first=true` on binary input may legitimately produce a constant column in the slice → surface as caveat, not degenerate.
>
> **Check 4 — `target_encoding` CV self-leakage** (only for target_encoding): confirm `params.target` in `df.columns`; pick a **singleton** row (category appearing exactly once in the slice). With CV leave-one-fold-out, a singleton's encoded value **cannot** come from its own category, so it must equal the global target mean (`smoothing==0`) or a smoothed-to-global value. Check `encoded_value_for_singleton != target_value_for_singleton` (equality ⇒ self-leakage → `cv_leakage_check.status: failed`). No singleton in the slice → caveat (_"could not verify CV leave-one-out; full-run determinism enforced by the engineer-features impl"_), not a halt.

**Sequencing** `[CORE]`: don't abort on a degenerate probe — run every transform's probe and report all; **thread `df_after` forward** as the working state for the next transform's probe (transforms chain, not independent); `validator_block_overall = ok` iff every `probe.health == ok`.
**Why head-500 (rationale to preserve)** `[CORE]`: 500 rows catches type/missing-column/schema bugs, single-valued output, and CV self-leakage (cardinality permitting); it's *too few* for distribution-level or float-determinism claims — **determinism is the CLI's test gate**, the probe is a structural belt-and-suspenders.

##### 9.1.3 `feature-executor` `[MIXED]` (CLI wrapper)
**Role:** Thin wrapper around `<ml-cli> engineer-features`. _"No transform logic lives here; the CLI is the single source of truth for transform determinism."_ Spawned at most once per invocation.
**Input:** `venv_python`, `sample_parquet_path`, `spec_yaml_path`, `output_parquet_path`, `log_dir`, `audience_mode`.
**EXACT CLI (verbatim):**
```bash
<venv_python> -m <pkg>.cli engineer-features \
  --input <sample_parquet_path> \
  --spec  <spec_yaml_path> \
  --output <output_parquet_path> \
  --json
```
`--json` → structured summary on stdout; **stderr → `<log_dir>/execute.log` verbatim**; capture `exit_code` + `duration_seconds`.
**Output:** JSON `status: ok|failed`, `output_path`, `exit_code`, `duration_seconds`, `log_path`, `executor_summary` (rows/cols in/out; per-transform `{id,name,type,duration,new_columns,null_delta}`). On failure `error.kind` from a fixed enum mirroring the CLI's own codes:
```
input_not_found | spec_yaml_parse_error | spec_validation_error | spec_missing_transforms |
unknown_transform_type | transform_input_arity | transform_missing_column |
transform_invalid_param | transform_target_missing | transform_output_columns_mismatch |
output_write_failed | unexpected_error
```
_"If a new kind surfaces (CLI added an error), pass it through — don't munge or rename."_ **Do-not:** no retry; no inspecting the output parquet (the closing `validate-artifact --probe-output` does that); no spec mutation; never delete output on failure.

---

#### 9.2 STAGE 5 — Dataset Generation

Orchestrator owns the loop; two sub-agents: a split-strategy locker and the six-check quality validator. The orchestrator itself runs `<ml-cli> split-dataset` (between the two) and the closing `validate-artifact`.

##### 9.2.1 `split-designer` `[MIXED]`
**Role:** The **strategy-locker**. Takes EDA's `cv_strategy` + `targets` + overrides, validates the choice against the actual input parquet, emits a normalized `split_strategy` + `params`. _"You do not write any files. You do not invoke the CLI."_ Produces the params for `<ml-cli> split-dataset --strategy <random|stratified|time_aware|grouped>` (the four are mutually exclusive).

**Input:** `cv_strategy` (`{type, reason, time_column, group_column, proposed_split}`), `targets[]`, `targets_relationship`, `input_columns[]`, `input_parquet_path` (cheap single-col probes), `strategy_override`, `sample_strategy` (`full|downsample_majority|downsample_class_balance`), `sample_params`, `split_seed` (default 42), `audience_mode`.

**Output (ok):** `status: ok`, `strategy`, `params` block, `split_strategy_block` (mirrors the artifact frontmatter), `caveats[]`.
```yaml
# params
train_pct: <or 0.7>
val_pct:   <or 0.15>
test_pct:  <or 0.15>
# + strategy-specific: stratified→stratify_column · time_aware→time_column(+windows) · grouped→group_column
# + if sample_strategy != full: sample_strategy:{type, params}
```
```yaml
# split_strategy_block (artifact-frontmatter shape)
type: <strategy>
seed: <split_seed>
time_column: <col|null>
group_column: <col|null>
stratify_column: <col|null>
train_window: <{start,end}|null>
val_window:   <{start,end}|null>
test_window:  <{start,end}|null>
```
**Output (conflict — orchestrator MUST halt, do not split):** `status: conflict`, `conflicts[]` = `{kind, severity: blocker|warning, message, suggested_fix}`. Conflict `kind` enum: `time_aware_missing_time_column | grouped_too_few_groups | hybrid_strategy_unsupported | stratified_no_target | time_window_invalid | strategy_overrides_eda`.

**Resolution** `[CORE]`: override set → use it + `strategy_overrides_eda` **warning** (_"The DS asked for this; we obey."_); else `cv_strategy.type`; else (no EDA, no override) → default `random` 0.7/0.15/0.15 + `cold_split_defaults` caveat.
**Per-strategy validation (verbatim):**
- `time_aware`: `time_column` set (else blocker `time_aware_missing_time_column`) + exists (`column_not_in_input`); explicit windows validated `train.end <= val.start <= val.end <= test.start` (else `time_window_invalid`).
- `grouped`: `group_column` set + exists; probe `n_unique` — `<2` blocker `grouped_too_few_groups`, `<5` warning `grouped_low_cardinality`.
- `stratified`: resolve stratify col (explicit flag → multi_output highest-cardinality target + warning → single target → else blocker `stratified_no_target`); must exist.
- `random`: no extra validation.
**Hybrid refusal** `[CORE]`: `time_aware + stratified` → blocker `hybrid_strategy_unsupported`: _"Hybrid time+stratified splits are out of scope for v1. Pick `time_aware` (risk class imbalance) or `stratified` (risk temporal leakage)."_
**Do-not:** never override EDA unless `--split-strategy` passed; never silently degrade `time_aware→random`; never invent window boundaries; never compute the split; collect *every* conflict.

##### 9.2.2 `dataset-validator` `[CORE]` — the six deterministic quality checks
**Role:** The **quality gate**. Reads the just-written train/val/test parquets + the locked split-strategy + the upstream feature-spec's `schema_hash`, runs 6 checks. _"Every check is a pure function over the on-disk parquets; no LLM judgment."_ **Pure in-process** polars/pyarrow/scipy — _"Do NOT invoke `validate-artifact` — that's the orchestrator's closing gate."_ `verification_status = failed` if ANY check `failed`; warnings never fail the gate.

**Output:** JSON `status: passed|failed` + `quality_checks[]` = `{check, status: passed|failed|warning, notes, ratios?, ks_p_values?}`.

**THE SIX CHECKS (verbatim, run order, no short-circuit):**
> **1 — `no_row_leakage_across_splits`**: hash rows per split (`df.hash_rows()`); any `train∩test`, `train∩val`, `val∩test` overlap → `failed`. _Caveat: true duplicate rows pre-split read as leakage — that's itself a data-quality fail; DS remedy is a unique `row_id` upstream._
> **2 — `group_leakage`** (only `type==grouped`): any `group_column` value in >1 split → `failed`. Else emit `passed`, `notes:"n/a (strategy != grouped)"` (shape-stable).
> **3 — `time_ordering`** (only `type==time_aware`): require `train[time].max() < test[time].min()` and (val present) `train.max() <= val.min()` and `val.max() <= test.min()`; violation → `failed`. Else `passed` n/a.
> **4 — `train_class_balance`** (per target): `max_dev = max |train_ratio(cls) - overall_ratio(cls)|`. `<=0.05` → `passed` (+`ratios`). `>0.05` AND `type==random` → **`warning`** (DS chose random). `>0.05` AND `type==stratified` → **`failed`** (splitter bug/extreme imbalance). Continuous target (>50 distinct in train) → skip, `passed` n/a.
> **5 — `train_test_distribution_drift`** (per numeric non-target col): `scipy.stats.ks_2samp`; a column "drifts" if `p<0.01`. 0 drifted → `passed`; `<=5` → **`warning`** (within tolerance); `>5` → **`failed`**. Populate `ks_p_values`. _scipy-absent fallback: polars mean+std, drift if `|mean_train-mean_test|/std_overall > 0.3` + caveat._
> **6 — `feature_schema_hash_match`** (when `expected_schema_hash` set): `schema_hash(p)=sha256("\n".join(f"{name}:{type}:{nullable}"))` per split, all must equal the upstream feature-spec hash; mismatch → `failed`. `expected_schema_hash is None` (from_files) → `passed` n/a.

**Do-not:** no short-circuit (run all 6); never delete splits (orchestrator rollback owns that); reads-only; **trust `expected_schema_hash` from the orchestrator — don't recompute the upstream hash**; emit `passed` n/a for 2/3/6 when strategy doesn't apply (shape-stable); never degrade `failed→warning` beyond Check 4's built-in branch.
**Why pure-function** `[CORE]`: re-running is identical ⇒ orchestrator trusts the verdict without retry, and `quality_checks[]` is a **stable downstream contract** (_"Stage 6 reads these directly to decide if the dataset is trainable"_). Extensibility: add new checks as new entries; never mutate existing semantics.

---

#### 9.3 STAGE 6 — Model Implementation

Orchestrator owns the loop; four **CLI-wrapper** sub-agents run in sequence: baseline-trainer (floor) → hp-search-runner (per family×Y) → model-evaluator (held-out per Y) → model-card-generator. Stage constraint: **sklearn-only** families `gradient_boosting | regularized_linear | random_forest`; NN deferred.

##### 9.3.1 `baseline-trainer` `[MIXED]`
**Role:** Wrapper around `<ml-cli> baseline-train`. Establishes the **floor metric every candidate family must beat**. On `status: failed` orchestrator **MUST halt**.
**Input:** `venv_python`, `train_parquet_path`, `target_spec_path` (column→task_type + targets_relationship), `baseline_spec_path` (per-target `{type, expected_floor_metric, metric_name, baseline_features?}`), `output_subdir`, `seed` (42), `audience_mode`.
**EXACT CLI (verbatim):**
```
<venv_python> -m <pkg>.cli baseline-train \
  --dataset <train_parquet_path> --target-spec <target_spec_path> \
  --baseline-spec <baseline_spec_path> --output <output_subdir> --seed <seed> --json
```
**Parses `baseline-result.yaml`:** `baselines[] = {target, type, metric_name, train_metric, model_path, feature_columns, training_duration_seconds}`.
**Output:** JSON `status`, `baseline_results[]` (types e.g. `logreg_3feat|majority_class|mean_predictor`), `baseline_log_path`, `error{exit_code, stderr_tail(~80), message}`. **Sanity probe:** confirm every `model_path` exists — missing pickle ⇒ failure. **Do-not:** no retry; never alter the seed (_"MUST be reproducible across re-runs"_).

##### 9.3.2 `hp-search-runner` `[MIXED]`
**Role:** ONE (family × Y) invocation of `<ml-cli> hp-search`. _"You do NOT pick the winner across families ... You do NOT compute relative improvement vs baseline."_ Distinctive failure semantics: on failure the orchestrator **continues with the next family** (_"a single family's failure doesn't doom the run"_).
**Input:** `venv_python`, `family`, `target` (or comma-list for multi_output), `task_type` (`binary_classification|multiclass|multilabel|regression|ordinal|survival`), `train_parquet_path`, `val_parquet_path` (or null → CLI k-fold CV), `target_spec_path`, `hp_space_path` (EDA's per-target space dumped **verbatim**), `cv_strategy_path`, `max_trials` (50), `max_time_seconds` (3600), `seed` (42), `output_subdir`, `audience_mode`.
**EXACT CLI (verbatim):**
```
<venv_python> -m <pkg>.cli hp-search \
  --dataset <train> --val-dataset <val> --target-spec <ts> \
  --family <family> --space <hp_space_path> --cv-strategy <cv> \
  --max-trials <n> --max-time-seconds <s> --seed <seed> --output <output_subdir>
```
_"If `val_parquet_path is None`, omit `--val-dataset` — the CLI falls back to k-fold CV using `cv_strategy_path`."_ Time-cap behavior to preserve: if the cap fires before `--max-trials`, it returns best-so-far + exits cleanly `status: ok` (not non-zero).
**Parses `hp-results.yaml`:** `{family, library (sklearn class path), target, metric_name, best_metric, n_trials, best_params, winner_model_path, duration_seconds, per_trial_summary_path}`.
**Do-not (load-bearing):** never widen/tighten/reshape the HP space (_"If the bounds seem off, the DS fixes them in EDA's model-design-recommender + re-runs"_); never iterate families; never compare vs baseline (Phase 4 owns winner selection); no retry; never alter seed; never compute the test metric; never propose NN (sklearn-only — propagate the CLI's family-name guard failure faithfully).

##### 9.3.3 `model-evaluator` `[MIXED]`
**Role:** Held-out evaluation via `<ml-cli> evaluate` on the winning pickle against the test split; returns the block the orchestrator drops into `models[].evaluation`. Handles ONE Y. On `status: failed` orchestrator **MUST halt**.
**Input:** `venv_python`, `winner_model_path`, `test_parquet_path`, `target_spec_path`, `slices_path` (may be empty `[]`), `output_subdir`, `audience_mode`.
**EXACT CLI (verbatim):**
```
<venv_python> -m <pkg>.cli evaluate \
  --model <winner_model_path> --test-dataset <test_parquet_path> \
  --target-spec <target_spec_path> --slices <slices_path> --output <output_subdir>
```
**Parses `eval-artifact.yaml`:** `{target, test_metrics{...}, calibration: null|{ece,brier,reliability_plot_path}, slices[]{dimension,slice,metric,n}, threshold_sweep_path, residual_plots_path}`.
**Output:** `calibration` null for regression/survival; `threshold_sweep_path` non-null only for binary; `residual_plots_path` non-null only for regression/ordinal. **Plot-existence probe:** each non-null plot path must exist + be non-zero, else failure. **Slice warn-not-fail:** a slice with `n<30` is dropped by the CLI — _"Do NOT fail the run because a slice has n<30"_ (orchestrator records a `slice_n_below_30_skipped` caveat). "Primary metric" surfaced in chat = `test_metrics[<EDA baseline.metric_name>]` (the same metric baseline + HP search optimized). **Do-not:** no retry; no extra metrics; no synthesized plots; no target iteration; never escalate `n<30`.

##### 9.3.4 `model-card-generator` `[MIXED]`
**Role:** Invokes `<ml-cli> gen-model-card` against the in-progress `.tmp` model artifact; returns card path + the H2 sections rendered. _"The card is the DS's go/no-go gate before they open the PR; v1 does NOT auto-open the PR."_ Note: `audience_mode` is **not** a direct input — the CLI reads it from the artifact frontmatter to style the card body.
**Input:** `venv_python`, `artifact_path` (`<slug>-v<n>.model.md.tmp` — carries every block EXCEPT `model_card`; the CLI renders the card from the rest), `output_path`.
**EXACT CLI (verbatim):**
```
<venv_python> -m <pkg>.cli gen-model-card \
  --artifacts <artifact_path> --output <output_path>
```
(_"`--artifacts` is the flag name — singular `--artifact` would be incorrect."_)
**Output:** JSON `status`, `model_card_path`, `sections_rendered[]` (scan `## ` H2 headers in order). Required sections: `Purpose, Training Data, Features, Performance, Calibration, Slices, Limitations, Lineage` (Calibration classification-only; Slices when non-empty). **Do-not:** no retry; never edit the rendered card (the DS's review surface); never validate its content; never open the PR; **never modify the `.tmp`** (else the card is rendered from a different artifact than Phase 7 writes — lineage drift); never delete the `.tmp` on success (orchestrator does that after writing the final `.md`).

---

#### 9.4 Stages 4–6 — the reusable skeleton (bottom line)

1. **One orchestrator per stage** owns the loop, writes staging YAMLs, and runs the closing `validate-artifact` gate with delete-on-failure rollback.
2. **Thin CLI-wrapper workers** shell to exactly one subcommand with `--json`, tee logs, parse one output YAML, return one structured block, and **never retry**.
3. **Two in-process deterministic checkers** (feature-validator's 4 invariants, dataset-validator's 6 checks) that **collect all findings** without short-circuiting and gate the write.

Everything else — CLI name, workspace paths, command names, domain columns, internal design-doc refs — is `[SPECIFIC]` dressing over that skeleton.

---

## Part IV — Reuse

### 10. The adapters (swap these first) `[SPECIFIC]` implementations / `[CORE]` patterns

The factory reaches the world through two MCP servers. These are the **most swappable** parts — a spin-off replaces the implementations and keeps the patterns. (Documented from the source repos; all secrets/hosts/account-ids redacted.)

#### 10.1 Data-source adapter — read-only warehouse access
**Pattern:** *don't write your own DB MCP.* Vendor an existing one (the reference vendors `awslabs.postgres_mcp_server`'s FastMCP, importing its `@mcp.tool` decorators and re-exposing them over HTTP via `streamable_http_app()`), then wrap it in a **read-only security shell**. The load-bearing pattern is **read-only enforced in depth, not by one flag** — four layers:

1. **DB role grants** — a dedicated login role with only read: `NOSUPERUSER NOCREATEDB NOCREATEROLE`, `GRANT pg_read_all_data` (SELECT everywhere, excludes credential catalogs, honors RLS), IAM-token auth (`GRANT rds_iam`, no stored password).
2. **Role session defaults** (belt-and-suspenders): `default_transaction_read_only=on`, `statement_timeout=60s`, `idle_in_transaction_session_timeout=30s`, `lock_timeout=5s`.
3. **Reader endpoint** — connect via the read replica so writes are impossible at the network level.
4. **Tool-layer keyword rejection** — never pass the upstream server's `--allow_write_query`; every SQL string is run through `detect_mutating_keywords()` + `check_sql_injection_risk()` before execution.

Plus an **access gate**: a shared auth gateway rejects any identity lacking the read-only group *before* the MCP is reached. One notable `[SPECIFIC]`-but-instructive detail: the vendored server hardcoded the IAM DB user to the cluster **master** — defeating read-only — so a one-line **monkey-patch** pins every IAM-auth connection to the read-only role (`kwargs["db_user"] = settings.iam_db_user`). Lesson for a port: audit your vendored adapter for privilege-escalation defaults.

**The bulk-export tool `run_query_to_file`** is the factory's *"materialize a large extract to disk without it entering model context"* primitive (`run_query` returns rows inline — fine for explore-scale; this handles multi-million-row extracts context-free): **server-side named cursor → batched `fetchmany` → pyarrow `ParquetWriter` → temp file → object-store PutObject → short-TTL presigned GET URL.** Rows go DB→server→store→caller's disk; the tool result carries only a handle. **Always async** (returns a `job_id`, caller polls `get_job_status`) to survive idle timeouts. Blast-radius guards: max-bytes circuit breaker (checked per row-group flush), wall-clock cap, batch-row memory bound, temp file always deleted in `finally`.

**Live tool surface (8 tools):** `connect_to_database`, `is_database_connected`, `get_database_connection_info`, `get_table_schema`, `run_query` (inline), `run_query_to_file` (bulk→async), `get_job_status`, `create_cluster` (control-plane — note read-only applies to *data*, not provisioning). Connection enum `ConnectionMethod = rdsapi | pgwire | pgwire_iam`.

**Generic interface a spin-off targets:**
```
DataSourceAdapter:
  probe() -> {ok|unconfigured|unauthenticated|unauthorized, fix_hint}   # tri-state auth probe
  plan(query, hints) -> ExtractionPlan | Refusal{reason, narrowing}     # cost gate BEFORE any row is pulled
  extract(query, plan, out, format) -> {rows, duration, schema}         # single-shot else keyset pagination; NO OFFSET; partial-write cleanup
  fingerprint(sql, out) -> {query_hash, schema_hash, size, columns}     # deterministic
  verify(artifact) -> {passed|partial|failed}                          # read-back + row/schema gates, delete-on-failure
```
Swap points: query language, chunk-key detection, cost model, identity provider, object store. Keep: streaming columnar write, single-shot→keyset (no OFFSET — it drifts on changing tables and is O(N) at large offsets), the tri-state probe, deterministic hashing, verify-then-emit.

#### 10.2 LLM-inference adapter — server-side model calls
**Pattern:** *inference is a shared service, not a per-author credential.* A stateless MCP exposes two tools so ETL/ML authors pick and call a model **server-side with zero cloud creds on the client** (the server uses its own task role).
- **Provider-agnostic** via a unified "Converse"-style request shape — one request across model families; swapping families needs only a registry entry, no code change.
- **Structured output = one forced tool:** to get JSON, inject a single tool `emit_result` whose input schema *is* the caller's JSON Schema, with `toolChoice` forcing it; the tool's **input is the result**. No temperature set — determinism rests on model + prompt + schema (matters for any golden-hash flow). Guardrail/content-filter stop reasons become clear errors.
- **Static curated model registry** is the authoritative allow-list: each entry `{alias, displayName, provider, inferenceProfileId, region, costTier, inputPer1M, outputPer1M, enabled}`. **Rule:** only models that both return 200 from the Converse call *and* honor forced tool use are included (models without forced-tool support are deliberately excluded — critical, because structured output depends on it).
- **Stateless transport** (`sessionIdGenerator: undefined`, one transport+server per request, `GET`/`DELETE` → 405); auth upstream at the gateway (a bedrock group-gate); identity read only to attribute each call in logs.

**Live tool surface (2 tools):** `list_bedrock_models` (the authoritative enabled-model directory — author picks one, stores as the analysis def's `resolved_model`) and `bedrock_analyze(model, prompt, [system, schema, maxTokens])` (returns a parsed object when `schema` is given, else text). A generic factory **swaps the registry file + the model ARNs and keeps `analyze()` verbatim.**

#### 10.3 The auth/enforcement pattern (shared by all adapters) `[CORE]`
All MCPs sit behind a **shared auth gateway** (the convergent choice over per-service self-hosted auth): the gateway validates the identity token, enforces per-service group membership, serves OAuth metadata, and forwards un-spoofable identity headers; services trust the forwarded headers for audit and expose **loopback-only ingress** (DNS-rebinding protection disabled because the only ingress is the sidecar). A spin-off's equivalent: a reverse-proxy/gateway that centralizes authn + group-gate, so each adapter is auth-simple and identity is consistent for audit.

#### 10.4 Consolidated swap map
| Generic component | Reference implementation (swap it) |
|---|---|
| Data-source MCP | Vendored Postgres MCP + read-only shell (dedicated role, IAM monkey-patch, `run_query_to_file`) |
| Read-only guarantee | role grants + `default_transaction_read_only` + reader endpoint + tool-layer keyword rejection + group-gate |
| Large-extract-to-disk | server-side cursor → Parquet → object store → presigned GET; async + poll |
| Inference proxy | `bedrock_analyze` over a unified Converse API; structured output via forced `emit_result` tool |
| Model catalog | static allow-list (Converse-verified + forced-tool-capable only) |
| Auth/enforcement | shared gateway (token validate + group-gate + OAuth metadata); services trust forwarded identity headers; loopback ingress |
| Skill/agent distribution | source monorepo → deterministic object-store bundles (content-sha = version) → installer MCP (see §6) |

### 11. The reusable pattern catalog

The lift-and-reuse designs, distilled. These are what make the factory work; port them verbatim (rename identifiers only). Each cross-references where it's detailed above.

1. **The master split — LLM decides & judges · deterministic CLI computes · artifacts are the API** (§1, §3). The one non-negotiable. Nothing numeric is ever LLM free-text; nothing crosses a stage boundary except a typed file on disk.
2. **The deterministic-tool wrapper sub-agent** (§9.0.1). Shells to exactly one CLI subcommand with `--json`, tees logs, parses one output, returns one structured block, **never retries** (a deterministic failure fails identically), passes CLI errors through verbatim. The orchestrator owns the loop.
3. **In-process deterministic checkers that collect ALL findings** (§9.1.2, §9.2.2). Never short-circuit on the first failure — surface them all so the human sees the full picture; the checker gates the write.
4. **Artifacts as typed, versioned, lineage-tracked contracts** (§5.6, §7). Markdown = frontmatter (machine API) + body (human narrative); pydantic `extra="forbid"`; JSON-Schema exported and CI-checked in sync; versioned `<slug>-v<n>` paths, never overwrite without `--force`.
5. **Deterministic lineage fingerprints + a lineage walker** (§4.5, §5.6). `query_sha256` over *literal* source bytes; `schema_hash` over the column fingerprint; `validate-artifact --walk-lineage` checks cycle→existence→sha→schema→type→status in that order; a failed upstream `verification.status` poisons the chain.
6. **Working-result enforcement — verify before emit, delete-on-failure rollback** (§4.5, §5.6). The artifact is provisional until the closing `--probe-output` gate reads it back and matches row-count + schema_hash; on failure, delete the just-written files (graduated scope). *"Non-negotiable."*
7. **The generate → execute → regenerate ×3 self-healing loop** (§8.1). LLM-generated code (or any artifact with a runnable form) is not trusted until a deterministic executor runs it end-to-end; structured tracebacks feed the regeneration; kernel/launch faults escalate rather than retry.
8. **The verdict vocabulary + deterministic severity→result** (§5.1). `{result ∈ PASS|PASS_WITH_CAVEATS|REWORK, findings[severity ∈ blocker|risk|nit|info], score}`; result derived from the highest-severity finding so they can't disagree; `score` real only when an LLM judge ran.
9. **The gate rule hierarchy + the rubber-stamp ban** (§5.2). Deterministic checks first; an LLM judge only for genuine judgment, on top, never instead; adversarial framing (default REWORK when uncertain); **never** point an LLM at deterministic output and ask "looks right?"
10. **Run-manifest as a resumable control plane over the lineage chain** (§5.3–5.5). Index-over-not-truth-of; **persist-before-return** (atomic temp-then-rename); the **`--run` additive hooks** (zero-regression standalone ↔ also-a-worker); stage-name uniqueness (state machine, not attempt log); REWORK bound = 3.
11. **Leakage-out-at-the-source** (§8.3, §4.3, §9.1.2). Scan in EDA (0.99/0.9/0.05 + posterior/derived/id rules), refuse leaky columns as feature inputs, **fit-on-train / apply-outward** transforms, `target_encoding` CV self-leakage guard, and a structural re-check at the dataset gate (row/group/time-ordering).
12. **Beat-a-baseline-by-a-margin acceptance — with a multi-metric stability gate as the upgrade** (§9.0.8, §4.4). The *shipped* gate: a candidate must beat a simple baseline by a set margin on the primary metric. The stronger **multi-metric stability gate** (floor + train→holdout stability across several metrics, `select-model`) is real but **not wired into shipped Stage 6** — wire it in for a high-stakes port. Either way: never let one lucky score decide; put the metric/threshold set behind a domain profile.
13. **Multi-target first-class** (§8.1). `targets[] + targets_relationship ∈ {null, parallel, multi_output}` threaded through every stage as per-Y-loop vs joint-pass. Build it in from day one — retrofitting leaves scars.
14. **Meaning-as-data + DS-leads + human checkpoints** (§8.2). Semantic manifests carry data meaning on disk (not in the model's head); sub-agents *propose, don't decide* (opt-in, never auto-override); interactive gates enforce "AI proposes, humans decide."
15. **Content-sha bundle distribution with independent library/prompt versioning** (§6.2). Deterministic zips where `sha256` *is* the version; idempotent no-op re-sync; publish-time path-rewrite + runtime path-dep pickup into isolated per-stage envs.
16. **Swappable adapters behind a shared auth gateway** (§10). Read-only-in-depth data source (role grants + session defaults + reader endpoint + keyword rejection); server-side inference proxy with forced-tool structured output + a curated model allow-list; one gateway centralizes authn + group-gate.
17. **A read-only live lens on a run** (§4.3 `serve`). Optional but cheap: a loopback-only, GET-only, path-jailed web view where "the agent is the sole writer, the app the sole reader." Good for observability without coupling.

### 12. Build-your-own: a spin-off recipe

A concrete order of operations to stand up a generic, open-sourceable ML factory for a new domain. **Build the deterministic core and the contracts first; the agents are a thin skin over them.**

**Step 1 — Define the artifact contracts + lineage (L2 first).** Write pydantic (or equivalent) models for your stage artifacts, all extending one `ArtifactBase` (`artifact, version, stage, parent{path,sha256}, input_mode, verification{status,method,errors}, caveats[], backtrack_signals[]`), all `extra="forbid"`. Decide your chain (the reference chain `input → eda → feature → dataset → model` is a good default). Add a `validate-artifact --walk-lineage --probe-output` command that hashes the chain and probes on-disk outputs. **This is the spine — get it right before anything else.**

**Step 2 — Build the deterministic CLI (L3).** One binary, many `--json` subcommands: `split`, `engineer-features` (a closed transform registry, fit-on-train/apply-outward), `baseline-train`, `hp-search` (Optuna/seeded), `evaluate`, `gen-model-card`, `validate-artifact`, `profile-dataset`. Seed everything (`--seed=42`, `default_rng` only); byte-stable columnar writes; content hashing. **Ship the acceptance gate as code** (the multi-metric stability gate + the dataset quality checks) — these are pure functions with unit tests. Put domain-specific metrics (the reference's KS/ROB/score-PSI) behind a profile, defaulting to generic ML metrics.

**Step 3 — Choose your agent runtime + adapters (L1).** Pick the runtime (Claude Agent SDK or equivalent): each stage = a skill/playbook, each specialist = a subagent with a scoped tool set + JSON return. Write two adapters as MCP servers (or the SDK's tool equivalent): a **data-source adapter** (read-only-in-depth; a `probe/plan/extract/fingerprint/verify` interface; keyset not OFFSET) and, if any stage uses an LLM for enrichment/design, an **inference adapter** (server-side, forced-tool structured output, curated model allow-list). Front both with one auth gateway.

**Step 4 — Author the orchestrators (L5) + specialists (L4).** For each stage, one playbook that owns the phase flow, spawns specialists, and runs the closing `validate-artifact` gate with delete-on-failure rollback. Two specialist archetypes: **designers/analyzers** (LLM judgment — target design, leakage scan, family ranking) and **deterministic-tool wrappers** (no-retry CLI shells). Keep the DS-leads posture (propose, don't auto-decide) and the interactive checkpoints. Thread multi-target from the start.

**Step 5 — Add the self-healing + verification loops.** Any stage that emits runnable code (a notebook, a script) gets a generate→execute→regenerate ×3 loop with a deterministic executor as the gate. Every stage refuses to emit until its output verifies.

**Step 6 — (Optional) Add the control plane (substrate A).** If you want a resumable multi-stage front-door, add a run-manifest FSM (`stages[]`/`verdict`/`next_action`), the `{result, verdict, next_action, artifacts_written}` worker return, persist-before-return, and the `--run` additive hooks. Remember: the manifest is an *index over* the lineage chain, never its truth.

**Step 7 — Distribution.** Package prompts + library as content-sha bundles behind an installer; version library and prompts independently; publish-time path-rewrite + runtime path-dep pickup into isolated envs.

**Reuse-vs-rewrite at a glance:**
| Reuse verbatim (`[CORE]`) | Rewrite for your domain (`[SPECIFIC]`) |
|---|---|
| the artifact/lineage contracts + `validate-artifact` | the concrete stage/artifact names, dtype vocabulary |
| the CLI scaffold: split/engineer/baseline/hp-search/evaluate/validate/model-card | domain metrics (KS/ROB → your metrics), the family universe if you want NN/GBM |
| leakage rules, the 6 dataset checks, the 4 feature invariants, transform registry | the target-designer domain-prior tables + feature-sufficiency categories |
| the verdict vocabulary, gate hierarchy, run-manifest, worker contract | the data-source + inference adapters, auth provider, object store |
| the self-healing loop, working-result enforcement, distribution mechanics | the semantic-manifest content (concept reused, entries rewritten) |

**Minimal-viable vs full.** *MVP:* Steps 1–2 + a single hand-driven stage that consumes a parquet, splits, trains a baseline + one family, evaluates, and emits a lineage-tracked model artifact with a card — no control plane, no distribution. That already delivers the reproducibility + auditability that is the whole point. *Full:* all six stages, the EDA crown jewels, the control plane, adapters behind a gateway, and bundle distribution to a fleet.

**The single deepest lesson to carry:** the value isn't the models — it's that **every number is reproducible, every artifact is lineage-verified, and the LLM never silently substitutes for a computation.** Build the contracts and the deterministic core first; the agents are the easy part.

### 13. Appendices

#### Appendix A — Verbatim output frontmatter templates (the inter-stage contracts)
The complete CLI reference is the 18-command table in **§4.2**; the input `saved-dataset` template is in **§7**; the universal `ArtifactBase` shape is in **§5.6**. Below are the three ML-stage output contracts, verbatim (swap `<ml-cli>`, the `created_by` names, the `.data-gents/...` paths, and the domain column examples).

**A.0 — the four stage commands (signatures)** — what a builder actually types to chain the pipeline:

| Stage | Command | Input (one of) | Key options | Output |
|---|---|---|---|---|
| 3 | `/…-eda-explore` | `<saved-dataset.md>` \| `<file>` \| `<topic.table>` | `descriptive\|modeling` (asked up front); `--target-column` `--split-column` `--feature-columns` (file-entry only); **continuous session** | `eda-exploration.md` |
| 4 | `/…-feature-engineer` | `--from-artifact <eda.md>` XOR `--from-file <parquet> --feature-spec <yaml>` | — | `feature-spec.md` |
| 5 | `/…-dataset-generate` | `--from-artifact <feature-spec.md>` XOR `--from-files <train,val,test> --target-column <Y>` | `--split-strategy` · `--sample-strategy`(+`--sample-params`) | `dataset.md` |
| 6 | `/…-model-implement` | `--from-artifact <dataset.md>` XOR `--from-files … --eda-path <eda.md>` | `--min-improvement=0.05` `--max-hp-search-trials=50` `--max-hp-search-time-seconds=3600` `--seed=42` `--families-to-skip` | `model.md` + `model-card.md` |
| 2b | `/…-split` (optional; the leakage-safe path, §5.6) | `--from-artifact <saved-dataset.md>` | `--split-strategy` | `dataset.md` (`stage: 2`) |

All stages accept `--force` (bump the `-v<n>` version instead of prompting) and `--run <slug>` (additive manifest-worker mode — ETL/dashboard). Stages 4/5/6 are transactional (one run = one artifact); Stage 3 is a continuous session.

**A.1 — `feature-spec` (Stage 4 output):**
```yaml
---
artifact: feature-spec
version: "1.0"
stage: 4
created_at, created_by: "data-feature-engineer", audience_mode
parent: <parent block — null when from_file>
input_mode: <from_artifact | from_file>
input_file: <block — null when from_artifact>
verification: {status: passed, method: deterministic_script, ran_at, execution_log, errors: []}
transforms:
  - id, name, type, inputs: [...], params: {...}, output_column   # or output_columns: [...]
    probe:
      health: ok
      columns_added: [...]
      null_delta: <int>
      samples: [...]                      # 3 sample values from the new column(s)
      cv_leakage_check: {status: passed, reason: null}   # for target_encoding only
output: {path, row_count, column_count, schema_hash}
target_compatibility:
  - {target, leakage_check: passed, notes: null}
backtrack_signals: []
caveats: []
---
# schema_hash = sha256("\n".join(f"{field.name}:{field.type}:{field.nullable}" for field in schema))
# parent.artifact MUST be "eda-exploration" (enforced by _check_parent_is_eda)
# path: .data-gents/features/<topic>/<slug>-v<n>.feature-spec.md
```

**A.2 — `dataset` (Stage 5 output):**
```yaml
---
artifact: dataset
version: "1.0"
stage: 5
created_at, created_by: "data-dataset-generate", audience_mode
parent, input_mode, input_file
verification: {status: passed, method: deterministic_script, ran_at, execution_log, errors: []}
splits:
  train: {path, row_count, sha256}
  val:   null | {path, row_count, sha256}
  test:  {path, row_count, sha256}
split_strategy:
  type: <random | stratified | time_aware | grouped>
  seed, time_column, group_column, stratify_column
  train_window, val_window, test_window        # {start,end}|null
feature_schema: {hash, columns: [{name, dtype}]}   # hash = upstream feature-spec's schema_hash
targets: ["<Y>", ...]
targets_relationship: <parallel | multi_output | null>
quality_checks:
  - {check: no_row_leakage_across_splits,  status: passed, notes}
  - {check: group_leakage,                 status: passed, notes, ratios: null}
  - {check: time_ordering,                 status: passed, notes}
  - {check: train_class_balance,           status: passed, notes, ratios: {overall, train}}
  - {check: train_test_distribution_drift, status: passed, notes, ks_p_values: {<col>: <p>}}
  - {check: feature_schema_hash_match,     status: passed, notes}
backtrack_signals: []
caveats: <accumulated from split-designer + dataset-validator>
---
# path: .data-gents/dataset/<topic>/<slug>-v<n>/<slug>-v<n>.dataset.md  (next to the split parquets)
```

**A.3 — `model` (Stage 6 output):**
```yaml
---
artifact: model
version: "1.0"
stage: 6
created_at, created_by: "data-model-implement", audience_mode
parent, input_mode, input_file
verification: {status: passed, method: deterministic_script, ran_at, execution_log, errors: []}
models:
  - target: "<Y>"
    baseline: {type, metric, metric_name}
    candidates_evaluated:
      - {family, library, best_metric, hp_trials, best_params: {...}}
    winner: {family, library, params: {...}, model_path, training_duration_seconds}
    evaluation:
      test_metrics: {<metric_name>: <value>, ...}
      calibration: {ece, brier, reliability_plot_path} | null
      slices: [{dimension, slice, metric, n}, ...]
      threshold_sweep_path: <path|null>        # binary only
      residual_plots_path:  <path|null>        # regression/ordinal only
      eval_artifact_path
model_card: {path}
approval: null           # DS opens the PR manually after reviewing the card — v1 does NOT auto-open
backtrack_signals: []
caveats: <sub-agent caveats + slice warnings + family_skipped notes>
---
# path: <output_dir>/<slug>-v<n>.model.md  (rendered from a <slug>-v<n>.model.md.tmp staging file)
```

#### Appendix A.4 — Staging-input file shapes (the glue Stage 3 → Stages 4–6)
The Stage-6 orchestrator *projects* these small YAMLs from the EDA artifact's fields (A.5) and writes them under `<output_dir>/staging/`; the CLI reads them. Shapes are exact to the pydantic models (`schemas/eda.py`) — do not guess them.

```yaml
# target-spec.yaml  — projected from eda.targets[]; read by baseline-train / hp-search / evaluate
targets:
  - {column: is_default_90d, task_type: binary_classification}
targets_relationship: parallel                 # null | parallel | multi_output

# baseline-spec.yaml  — projected from eda.baseline_spec.per_target[]
per_target:
  - {target: is_default_90d, type: logreg_3feat, expected_floor_metric: 0.62, metric_name: auc}

# hp-space-<family>-<target>.yaml  — eda.hp_search_spaces.per_target[<target>].spaces[<family>] dumped
#   verbatim: a param → HpParamSpec map (hp-search also receives --family separately)
max_iter:          {type: int,         low: 100, high: 400, step: 50, reasoning: "n=42k → §8.4 band"}
learning_rate:     {type: loguniform,  low: 0.01, high: 0.3,          reasoning: "standard GBM range"}
l2_regularization: {type: loguniform,  low: 1.0,  high: 10.0,         reasoning: "severe collinearity"}
class_weight:      {type: categorical, choices: ["balanced", null],   reasoning: "minority 12% < 20%"}
# HpParamSpec rules (validated at EDA time): int/float/loguniform need low<high (loguniform low>0);
# `step` is int-only; categorical needs non-empty `choices`; `reasoning` lives INSIDE each param —
# there is NO family-level `reasoning_per_param` sibling key.

# cv-strategy.yaml  — the canonical CvStrategy (dumped from eda.cv_strategy); read by split-dataset + hp-search
type: time_aware                                # time_aware | grouped | stratified | random
reason: "defaults are time-dependent; created_at spans 18 months"
time_column: created_at                         # null unless time_aware
group_column: null                              # set only for grouped
proposed_split: {train_pct: 0.7, val_pct: 0.15, test_pct: 0.15}
# NOTE: there is no `cv_folds` field here — hp-search derives folds (default 5) from this + the split.

# slices-<target>.yaml  — evaluate --slices; a list of dimension columns to break metrics out by (may be [])
- region
- channel
```

#### Appendix A.5 — The full `eda-exploration` (Stage 3) frontmatter (the richest artifact)
Verbatim to `EdaExplorationArtifact` (`schemas/eda.py`, `extra="forbid"`). The staging YAMLs in A.4 are projections of the modeling-mode fields here.

```yaml
---
artifact: eda-exploration
version: "1.0"
stage: 3
created_at, created_by, audience_mode
parent: {artifact: saved-dataset, path, sha256, version}   # or a dataset(2b) parent on the split-aware path
input_mode: from_artifact | from_file | cold
verification: {status: passed|partial, method: viz_spec_validated, ran_at, execution_log, errors: []}
mode: descriptive | modeling
sample: {path, row_count, column_count, manifest_coverage: full|partial|none}
viz_spec_path: <path|null>            # the Plotly-JSON bundle validate-viz-spec checks (the gate of record)
notebook_path: <path|null>           # derived, legacy
# ---- modeling-mode fields (null / empty when mode=descriptive) ----
targets:                              # null when descriptive; each entry:
  - column, goal, task_type: binary_classification|multiclass|multilabel|regression|ordinal|survival
    construction_method: single_column|rule_based_boolean|weighted_composite|latent_factor|survival_pair|snorkel_weak_supervision|multi_step_pipeline
    derivation_steps: [{id, type, name, inputs:[...], params:{...}, sql?, output_column, probe:{...}}]
    validation: {probe_rows, distribution_summary:{...}, health: ok|degenerate}
targets_relationship: parallel | multi_output | null
feature_candidates: [{column, dtype, role: numeric|categorical|datetime|json|text|id_like, missingness, cardinality}]
leakage_risks: [{target, column, strength, kind: perfect_predictor|near_perfect|posterior_info|derived_from_target|id_correlated, recommendation: drop|inspect|safe-with-caveat, reason}]
stability_risks: [{column, role, comparison, index: psi|csi, score, threshold, severity: minor|major, recommendation: monitor|inspect|drop, reason}]
imputation_recommendations: [{column, role: numeric|categorical|boolean, missingness, strategy: constant|constant_by_dtype|median|mean|mode, fill_value, fit_scope: literal|train_only, rationale, alternatives:[...]}]
feature_sufficiency_audit: {per_target: [...] } | {joint: {...}} | null   # optional (opt-in)
cv_strategy: {type: time_aware|grouped|stratified|random, reason, time_column?, group_column?, proposed_split: {train_pct, val_pct, test_pct}}
recommended_model_families:          # per_target XOR joint
  per_target: [{target, families: [{family, libraries:[...], rank, reasoning, notes?}]}]
hp_search_spaces:
  per_target: [{target, spaces: {<family>: {<param>: <HpParamSpec>}}}]      # HpParamSpec per A.4
baseline_spec:
  per_target: [{target, type, expected_floor_metric, metric_name}]
backtrack_signals: []
caveats: []
---
```

#### Appendix B — The ML-pipeline sub-agent roster
(All inherit full tools + the parent model unless a port pins otherwise. Stage 3 = continuous session; 4/5/6 = transactional.)

| Stage | Sub-agents |
|---|---|
| **3 EDA** (12) | `env-bootstrapper`, `sample-designer`, **`target-designer`**, `column-profiler`, `json-column-handler`, `relationship-analyzer`, **`target-analyzer`**, `dimensionality-analyzer`, **`model-design-recommender`**, `interpreter`, `notebook-composer`, `notebook-executor` |
| **4 Feature** (3) | `feature-proposer` (opt-in), `feature-validator` (4-invariant gate), `feature-executor` (CLI wrapper) |
| **5 Dataset** (2) | `split-designer` (strategy locker), `dataset-validator` (6-check gate) |
| **6 Model** (4) | `baseline-trainer`, `hp-search-runner`, `model-evaluator`, `model-card-generator` (all CLI wrappers) |

**Bold** = the three crown jewels (target design · leakage detection · model-family recommendation), detailed in §8.2–8.4. The broader platform also ships ETL + dashboard sub-agents (~52 total across all pipelines); those are out of the ML scope.

#### Appendix C — Consolidated rename map (identifier → generic)
| Reference identifier | What it is | Generic replacement |
|---|---|---|
| `merlina-ml` / `merlina_ml_tools.cli` | the deterministic CLI | `<ml-cli>` / `<pkg>.cli` — keep the subcommand names (they're `[CORE]`) |
| `merlina-rds-mcp` / `merlina-server` MCP | data-source / inference adapters | your MCP servers (§10) |
| `.data-gents/`, `.data-gents/eda/.venv` | runtime state dir + shared venv | any app-owned state dir |
| `.claude/commands/*.md`, `.claude/agents/<g>/*.md` | stage playbooks, sub-agents | your runtime's skills + subagents |
| `/data-eda-explore`, `/data-feature-engineer`, `/data-dataset-generate`, `/data-model-implement` | the stage commands | your command namespace |
| `data`, `etl-generation`, `data-ml-tools` topics | bundles | your bundle names |
| `is_30dpd`, `days_late`, `loans`, `remesas`, cents-MXN | fintech/lending domain examples | your domain columns |
| `risk_metrics.py` (KS/ROB), `selection.py` gate thresholds | credit-scorecard metrics | your domain metrics behind a profile |
| CONDUSEF/CNBV compliance section, `mcp-readonly-prod` group | Mexican-fintech regulators / IAM group names | your compliance + IAM |
| `/tech-bz-commit`, `/tech-bz-pr`, `beloz-etl` | commit/PR/deploy hooks | your VCS/deploy hooks |

#### Appendix D — Glossary
- **artifact** — a stage's output: a Markdown file whose YAML frontmatter is a typed machine contract and whose body is human narrative.
- **lineage chain** — the `parent.sha256`-linked sequence of artifacts; substrate (B); the ML pipeline's actual state machine.
- **run manifest** — `run.json`; substrate (A); a durable control-plane FSM that *indexes* the lineage chain (used by ETL/dashboard, optional for ML).
- **verdict** — `{result, findings[severity], score}`; the shared gate vocabulary.
- **working-result enforcement** — verify (read-back + gates) before emitting; delete-on-failure rollback.
- **deterministic-tool wrapper** — a sub-agent that shells to one CLI subcommand, never retries, returns structured pass/fail.
- **the stability gate** — the multi-metric acceptance test a model must clear (floor + train→holdout stability across metrics), never a single number.
- **DS-leads** — sub-agents propose/execute but never override the human's upstream choices unless an explicit override flag is passed.
- **multi-Y** — `targets[] + targets_relationship (parallel | multi_output)`; multiple targets handled per-Y or jointly, first-class.
- **CORE / SPECIFIC / MIXED** — the reuse tags: keep as-is / swap for your stack / keep the pattern, replace identifiers.

---

#### Appendix E — An end-to-end worked example (a toy dataset through every stage)
A 200-row toy makes the file-to-file flow concrete and pins the shapes above. **Columns:** `customer_id` (id), `loan_amount` (numeric), `region` (categorical, 4 values), `applied_at` (datetime, spans 18 months), `days_late` (a post-outcome field — a *leak*), `repaid_late` (the raw signal). **Goal:** predict early default. We take the **leakage-safe split-before-EDA path** (§5.6).

**① Data acquired →** `datasets/toy/loans-v1.saved-dataset.md`
```yaml
artifact: saved-dataset, stage: 2, verification: {status: passed}
extraction: {query_sha256: 9f2…, rows_extracted: 200}
output: {path: …/loans-v1.parquet, schema_hash: a41…}
schema: {columns: [customer_id, loan_amount, region, applied_at, days_late, repaid_late]}
```
`Next: /…-split --from-artifact datasets/toy/loans-v1.saved-dataset.md --split-strategy time_aware`

**② Split before EDA (Stage 2b) →** `dataset/toy/loans-v1/…dataset.md` (`stage: 2`); train/val/test parquets cut chronologically on `applied_at` (oldest 70% train … newest 15% test).
`Next: /…-eda-explore dataset/toy/loans-v1/…dataset.md modeling`

**③ EDA →** `eda/toy/loans-v1.eda-exploration.md` (`mode: modeling`) — abbreviated:
```yaml
targets: [{column: repaid_late, task_type: binary_classification, construction_method: single_column,
           derivation_steps: [{id:1, type: select_columns, inputs:[repaid_late], output_column: repaid_late}],
           validation: {probe_rows: 200, health: ok}}]
targets_relationship: null
leakage_risks: [{target: repaid_late, column: days_late, strength: 0.98, kind: perfect_predictor,
                 recommendation: drop, reason: "days_late is computed from the outcome"}]     # ← the leak, caught
cv_strategy: {type: time_aware, reason: "applied_at spans 18mo; defaults are time-dependent",
              time_column: applied_at, proposed_split: {train_pct: 0.7, val_pct: 0.15, test_pct: 0.15}}
recommended_model_families: {per_target: [{target: repaid_late, families: [
   {family: gradient_boosting, libraries: [HistGradientBoostingClassifier], rank: 1, reasoning: "n small; nonlinear"},
   {family: regularized_linear, libraries: [LogisticRegression], rank: 2, reasoning: "n<5k → linear safer"}]}]}
hp_search_spaces: {per_target: [{target: repaid_late, spaces: {gradient_boosting: {
   max_iter: {type: int, low: 50, high: 300, step: 25, reasoning: "n=200 → small band"}, …}}}]}
baseline_spec: {per_target: [{target: repaid_late, type: majority_class, expected_floor_metric: 0.5, metric_name: auc}]}
verification: {status: passed, method: viz_spec_validated}
```
`Next: /…-feature-engineer --from-artifact eda/toy/loans-v1.eda-exploration.md`

**④ Feature-engineer (split-aware) →** `features/toy/loans-v1.feature-spec.md`. Transforms (abbrev): **drop** `customer_id` (id-leak) + `days_late` (from `leakage_risks`); `standard_scaler(loan_amount)` **fit on train**; `one_hot(region)` fit on train; `date_parts(applied_at)`. Emits `outputs: {train, val, test}` sharing one `schema_hash`.
`Next: /…-dataset-generate --from-artifact features/toy/loans-v1.feature-spec.md`

**⑤ Dataset finalize (Stage 5) →** `dataset.md` (`stage: 5`) with all 6 `quality_checks` passed: no row/group leakage · `time_ordering` (train max < test min) · class balance · KS drift ≤ tolerance · `feature_schema_hash_match`.
`Next: /…-model-implement --from-artifact dataset/toy/loans-v1/…dataset.md`

**⑥ Model →** orchestrator projects the five staging YAMLs (A.4) from the EDA fields, then: `baseline-train` (majority-class floor, AUC 0.5) → `hp-search` per family → pick the winner **(must beat 0.5 by ≥ `--min-improvement` 0.05)** → `evaluate` on the untouched test split → `gen-model-card`:
```yaml
models: [{target: repaid_late, baseline: {type: majority_class, metric: 0.5},
  winner: {family: gradient_boosting, library: HistGradientBoostingClassifier, model_path: …},
  evaluation: {test_metrics: {auc: 0.71}, calibration: {ece: 0.04}}}]
model_card: {path: …/model-card.md}
approval: null      # a human reads the card, then opens the PR — the pipeline does not auto-ship
```
Closing gate `validate-artifact --walk-lineage --probe-output` passes → the artifact stays. **Trace any number back:** `model.md.parent → dataset(5).parent → feature-spec.parent → eda-exploration.parent → dataset(2b).parent → saved-dataset` — sha256-verified at every hop. That chain, plus the reproducible CLI numbers, is the whole point.

---

*End of blueprint. Provenance: reverse-engineered from a production fintech ML pipeline (internal codename "Merlina") via a parallel read of its command playbooks, sub-agent prompts, the `merlina-ml` CLI source, the shared specs/schemas, and the MCP-adapter + bundle-build repos. Architecture and contracts are captured faithfully; all secrets, credentials, internal hostnames, and account identifiers were redacted at the source-reading stage and never entered this document.*

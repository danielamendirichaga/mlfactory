# Reuse Map — churnpilot → ML Factory

> **Purpose.** A concrete, file-by-file plan for building the ML Factory (`ml-factory-architecture.md`)
> by lifting the proven parts of **churnpilot** (`../AI&DS_lab`) instead of starting from zero. Every
> churnpilot file is tagged **KEEP / UPGRADE / REBUILD / DOMAIN / DROP**, mapped to a factory layer, with
> a target location in the new project. This is the plan the scaffolding step (B) executes.
>
> Written after a full top-to-bottom read of churnpilot (20 modules, 22 test files, all docs) and the
> complete factory blueprint (all 13 sections + appendices).

---

## 1. The relationship in one paragraph

churnpilot and the ML Factory were built to the **identical thesis** — *the LLM decides & judges; a
deterministic, unit-tested CLI does all the math; typed artifacts with lineage are the API between
stages* (churnpilot's README diagram states it almost verbatim; the factory's §1 is the same idea).
churnpilot is essentially **the factory's deterministic bottom half, built one scale down**, and two of
its scoping decisions were *deliberate cuts of exactly what the factory demands*:

- **ADR-009 — "medium contract tier"**: Pydantic artifacts + `parent_sha256` + JSON sidecars, and
  *explicitly parked* JSON-Schema/CI, versioning, and a `validate-artifact` lineage-walker. The factory
  wants precisely that parked "heavy tier."
- **ADR-001 — "single agent, no orchestration"**: a human (or Claude) drives the CLI; no skills,
  subagents, MCP, or playbooks. The factory's whole top half (L1/L4/L5/L6) is that missing orchestration.

So the factory is largely **"un-park what churnpilot consciously deferred, then add the agent layer on
top of the CLI churnpilot already built."** The deterministic core is ~50% done and high quality.

---

## 2. Layer coverage (factory L1–L6 vs churnpilot)

| Factory layer | What it is | churnpilot status | Verdict |
|---|---|---|---|
| **L3 — Deterministic CLI** | the compute core (split/engineer/train/hp-search/eval/validate) | ✅ strong — Typer CLI, all math in tested modules, 168 tests | **reuse, mostly as-is** |
| **L2 — Contracts & state** | typed lineage-tracked artifacts + `validate-artifact` | 🟡 medium tier (`ArtifactBase`+`parent_sha256`+JSON) | **upgrade to heavy tier** |
| **L1 — Adapters** | data-source MCP + inference MCP | 🟡 local loader only (`source.py`), no MCP | **keep local path; MCP is new** |
| **L4 — Specialist subagents** | target-designer, leakage-scanner, CLI-wrappers | ❌ none (rules in `recommend.py`) | **greenfield (on the CLI)** |
| **L5 — Orchestrator playbooks** | one skill/playbook per stage | ❌ none (a hard-coded `run` loop) | **greenfield** |
| **L6 — Bundle distribution** | content-sha bundles + installer | ❌ standard wheel | **greenfield (later)** |

---

## 3. Per-file reuse table (the heart of the plan)

**Tags:** **KEEP** = lift ~verbatim (rename import only) · **UPGRADE** = keep core, raise to the factory's
bar · **REBUILD** = factory wants a different shape, use churnpilot as reference/seed · **DOMAIN** =
churn-specific, keep as the reference-domain instance (see §6 decision) · **NEW** = no churnpilot
equivalent, greenfield.

### 3a. The deterministic core (L3) — the gold

| churnpilot file | LoC | → Factory piece | Tag | What to do |
|---|---|---|---|---|
| `metrics.py` | 280 | `evaluate` + `stability.py` (PSI) + `risk_metrics.py` (KS/ROB) | **KEEP** | Near-exact match — decile-KS, **PSI with frozen reference edges** (the exact "load-bearing correctness fix" the factory calls out), ROB, gain/lift, ROC (Mann-Whitney), AP, P/R/F1, log-loss, calibration/ECE. numpy/pandas only. Lift verbatim → `compute/metrics.py`. Tests lift with it. |
| `compare.py` | 70 | `select-model` — the multi-metric **stability gate** (§4.4) | **KEEP→UPGRADE** | churnpilot's `_MAX_AUC_DROP=0.05` / `_MAX_SCORE_PSI=0.2` **already is** the factory's aspirational gate. Lift → `compute/model_select.py`; upgrade thresholds into a `Thresholds`/profile object + lexicographic rank + fallback. |
| `split.py` | 149 | `split-dataset` + `dataset-validator` | **UPGRADE** | Has time/grouped/random + a leakage guard (row-disjoint, time-ordered, entity-overlap = the factory's dataset-checks 1–3). Add **`stratified`** (the factory's 4th strategy) and the other 3 dataset checks (class balance, KS drift, `feature_schema_hash_match`). |
| `evaluate.py` | 105 | `evaluate` | **KEEP→UPGRADE** | Union metric pack + per-segment slices + score-PSI → `EvalReport`. Lift; later add threshold-sweep, calibration PNG, cross-split stability (W1), SHAP/PDP (W4), drop-`n<30` slices. |
| `profile.py` | 110 | `profile-dataset` + EDA `column-profiler` | **UPGRADE** | role/null/cardinality/numeric-stats + `target_corr` leakage hint. Add the factory's richer role order (json/id_like/text) + pair-type correlations (point-biserial/η²/Cramér's V) + the **0.99/0.9/0.05 leakage tiers**. |
| `model.py` | 384 | `baseline-train` + `engineer-features` + `hp-search` | **SPLIT (see §5)** | The big reconciliation. Extract baseline floor → `compute/baseline.py` (**KEEP**). Extract the `ColumnTransformer` preprocessing → a **standalone `engineer-features` transform registry** (**REBUILD**). Extract the search (`GridSearchCV`/`LogisticRegressionCV`/ccp) → `hp-search` (**UPGRADE** to Optuna, or keep CV for MVP). Model menu (logistic/tree/rf/xgboost) KEEP; add HistGBM/lightgbm/catboost later. |
| `source.py` | 116 | L1 data-source adapter (local/file path) | **KEEP + NEW** | file/sqlite/postgres loader + the handy `_coerce_numeric_like`. Keep as the **file path** of the adapter; the read-only-in-depth **MCP adapter** (probe/plan/extract/fingerprint/verify) is NEW. |
| `config.py` | 121 | config seam → `saved-dataset` / `target-spec` | **UPGRADE** | Pydantic `churn.yaml` (source + column map, `features:auto`). Generalize to **multi-target + `task_type`** (beyond binary) and adapter-neutral dtypes; feeds the saved-dataset frontmatter. |

### 3b. The contract/state layer (L2) — the spine to build first

| churnpilot file | LoC | → Factory piece | Tag | What to do |
|---|---|---|---|---|
| `artifacts.py` | 39 | `ArtifactBase` + lineage + `validate-artifact` + `export-schemas` | **UPGRADE (heavy)** | `ArtifactBase`(`extra="forbid"`) + `content_hash(df)` is the right seed. Extend to heavy tier: **markdown-with-frontmatter** (YAML machine contract + human body), `stage/version/created_at/created_by`, `verification{status,method,errors}`, `parent{path,sha256,version}`, `input_mode`, `caveats[]`, `backtrack_signals[]`; versioned `-v<n>` paths; **delete-on-failure**. Keep `content_hash`; add `schema_hash` + `query_sha256`. |
| — | — | `validate-artifact --walk-lineage --probe-output` | **NEW** | The lineage walker (cycle→existence→sha→schema→type→status) + on-disk probe. The factory's linchpin gate; churnpilot has none. |
| — | — | `export-schemas --check` | **NEW** | Emit JSON-Schema from the Pydantic models + a CI drift check. |

### 3c. The judgment / recommendation layer

| churnpilot file | LoC | → Factory piece | Tag | What to do |
|---|---|---|---|---|
| `recommend.py` | 176 | the **deterministic gate layer** under the L4 LLM judges (§5.2) | **KEEP→REPURPOSE** | churnpilot's tested rules (leakage exclusion, stability>peak-AUC, ship go/no-go, split choice, retrain, experiment-detect) are exactly the factory's *"deterministic checks FIRST"*. Lift as the deterministic core; the LLM `target-designer`/`target-analyzer`/`model-design-recommender` sit **on top** (never instead — the factory's rubber-stamp ban). |
| `validate.py` | 185 | saved-dataset verification + EDA sanity | **UPGRADE→REPURPOSE** | Graded pass/warn/fail, **collect-all-findings** (matches factory invariant #4), panel/snapshot, experiment detection. Feed the saved-dataset `verification.status` + validate-artifact probe. |

### 3d. Reporting / viz

| churnpilot file | LoC | → Factory piece | Tag | What to do |
|---|---|---|---|---|
| `charts.py` | 192 | `build-*-viz` bundles | **KEEP→REFORMAT** | Tested matplotlib visuals in a validated palette (PNG bytes). The factory emits **Plotly-JSON viz bundles** gated by `validate-viz-spec`. Keep the chart functions for the HTML report; add the viz-spec format later if wanted. |
| `report.py` | 210 | `gen-model-card` + `serve` | **KEEP→UPGRADE** | Self-contained HTML from artifacts (no compute — pure render, the right shape). Keep the HTML report; add the factory's canonical **`gen-model-card`** (8-section *markdown* card) as the model deliverable; `serve` (loopback live lens) is NEW/optional. |

### 3e. The CLI + package

| churnpilot file | LoC | → Factory piece | Tag | What to do |
|---|---|---|---|---|
| `cli.py` | 926 | L3 CLI surface (what subagents wrap) | **UPGRADE** | One command per stage — the right surface. Add `--json` machine I/O + structured stderr errors (so subagents can wrap "one CLI = one result, no retry"); add `validate-artifact`, `gen-model-card`, `export-schemas`, `serve`; split `engineer-features` out of `train`. The human-driven `advise`/`run` become the **deterministic fallback**; the LLM orchestration is layered above, not instead. |
| `__init__.py` | 9 | package init | **KEEP** | rename package. |

### 3f. Domain layer — churn-specific (see §6 decision)

| churnpilot file | LoC | → Factory piece | Tag | What to do |
|---|---|---|---|---|
| `generate.py` | 297 | — (factory consumes a `saved-dataset`; has no generator) | **DOMAIN/KEEP** | Seeded churn panel + A/B test with planted leakage/drift/uplift. Becomes the churn **reference domain's data + the factory's end-to-end demo** (the role Appendix E's toy-loans plays). Keep as fixture/instance. |
| `policy.py` | 220 | — (beyond factory scope; it ends at the model card) | **DOMAIN/KEEP** | Cost-based retention targeting + risk-vs-uplift contrast — churnpilot's downstream *application* of the model. Keep as a churn-domain post-model stage. |
| `uplift.py` | 153 | — (beyond scope) | **DOMAIN/KEEP** | S-/T-learner meta-models over the model stack. Domain extension. |
| `qini.py` | 130 | — (beyond scope) | **DOMAIN/KEEP** | Qini curve/coefficient + uplift-by-decile (numpy/pandas). Domain extension. |
| `monitor.py` | 82 | ops/drift (partial overlap w/ `evaluate` PSI) | **DOMAIN/KEEP→UPGRADE** | Per-feature PSI drift + retrain recommendation. Factory has drift inside evaluate/stability but no standalone monitor stage. Keep as an ops extension; reuses `metrics.psi`. |

### 3g. Tests & docs

| churnpilot | → | Tag | What to do |
|---|---|---|---|
| `tests/` (22 files, 2244 LoC, 168 tests) | test suite | **KEEP→CARRY** | Lift with their modules (metric-property, artifact-lineage-roundtrip, split-guard, model-beats-floor, stability, policy, uplift, qini, recommend, capstones). **Add:** validate-artifact lineage-walk tests, engineer-features transform tests, hp-search (Optuna) tests, `export-schemas --check` CI-sync test. |
| `docs/` (ADRs, PRD, DESIGN_BRIEF, v2, context, synthetic-data) | reference | **REFERENCE** | Keep as churnpilot's record. The new project writes its own PRD/ADRs — notably **reversing ADR-001** (single→multi-agent) and **ADR-009** (medium→heavy tier). |

---

## 4. CLI subcommand mapping (factory's 18 → churnpilot seeds)

| # | Factory subcommand | churnpilot seed | Gap |
|---|---|---|---|
| 1 | `baseline-train` | `model.py` baseline floor | extract to standalone |
| 2 | `hp-search` | `_tune_xgb` / `LogisticRegressionCV` / ccp | GridSearchCV → **Optuna TPE** |
| 3 | `evaluate` | `evaluate.py` | ✓ close |
| 4 | `select-model` | `compare.py` stability gate | ✓ close (upgrade thresholds) |
| 5 | `profile-dataset` | `profile.py` | richer roles + leakage tiers |
| 6 | `split-dataset` | `split.py` | add `stratified` |
| 7 | `engineer-features` | inside `model.py` ColumnTransformer | **decouple → registry** |
| 8 | `propose-feature-selection` | ~`recommend_features` | **new** |
| 9 | `propose-supervised-selection` | — | **new** |
| 10 | `validate-artifact` | — | **new (the spine gate)** |
| 11 | `gen-model-card` | `report.py` (HTML) | **new markdown card** |
| 12 | `export-schemas` | — | **new** |
| 13 | `serve` | — | **new (optional)** |
| 14 | `validate-viz-spec` | — | **new** |
| 15–17 | `build-*-viz` | `charts.py` (PNG) | reformat → Plotly-JSON |
| 18 | `build-notebook` | — | **new** |

**~8 of 18 have strong churnpilot seeds; ~10 are new/upgrade.** Conversely, churnpilot has *extra*
downstream stages the generic factory lacks (`simulate-policy`, `train-uplift`, `uplift-eval`,
`policy-contrast`, `monitor`, `report`) — it is **broader downstream, thinner upstream** (no
data-acquisition adapter, thin EDA).

---

## 5. The four structural reconciliations (where the shapes differ)

1. **Artifact format: JSON sidecar → markdown-with-frontmatter.** churnpilot writes `*.json`; the factory
   writes `*.md` (YAML frontmatter = machine contract, body = human narrative). Adopt the factory format in
   the upgraded `ArtifactBase`. Touches every artifact-emitting module (mechanical).
2. **Contract tier: medium → heavy.** Add versioning, `verification.status`, the `validate-artifact`
   lineage-walker + probe, delete-on-failure, and JSON-Schema/CI. This is the factory's *"build the spine
   first"* (§12 Step 1).
3. **Feature-engineering: fused → separate Stage 4.** churnpilot fits preprocessing *inside* the estimator
   pipeline (`ColumnTransformer` in `train_model`). The factory makes it a **standalone stage** with a
   closed transform registry, genuine fit-on-train/apply-outward, a `feature-spec` artifact, and the
   split-before-EDA leakage-safe path. This is the single biggest rebuild.
4. **Orchestration: human-driven CLI → LLM playbooks + subagents.** churnpilot's `run`/`advise` +
   `recommend.py` become the deterministic substrate; the LLM layer (skills per stage, subagents, MCP
   adapters) is greenfield built *on top of* the CLI. `recommend.py`'s rules stay as the deterministic
   gates the judges sit on.

---

## 6. Target new-project skeleton (what B scaffolds)

```
AI&DS/                              # (or a fresh repo — see decision)
  <pkg>/                            # the deterministic core (L3 + L2)
    __init__.py
    cli.py                         # Typer — the CLI subagents shell out to (--json, structured errors)
    artifacts/
      base.py                      # UPGRADE of artifacts.py → heavy-tier ArtifactBase (md+frontmatter)
      schemas.py                   # the stage artifact models (extra="forbid")
      validate.py                  # NEW — validate-artifact: lineage walk + probe-output
      lineage.py                   # content_hash (KEEP) + schema_hash + query_sha256
    compute/                       # the math (lifted from churnpilot)
      metrics.py                   # KEEP verbatim
      model_select.py              # KEEP←compare.py (stability gate)
      split.py                     # UPGRADE←split.py (+stratified, +6 checks)
      baseline.py                  # KEEP←model.py floor
      engineer.py                  # REBUILD — transform registry (fit-on-train/apply-outward)
      hp_search.py                 # UPGRADE←model.py search (Optuna, or CV for MVP)
      evaluate.py                  # KEEP←evaluate.py
      profile.py                   # UPGRADE←profile.py
    config.py                      # UPGRADE←config.py (multi-target, task_type)
    source.py                      # KEEP←source.py (file path) + adapter seam
    recommend.py                   # KEEP←recommend.py (deterministic gates)
    report.py / charts.py          # KEEP (HTML report + visuals) + gen-model-card
  domains/churn/                   # the reference domain instance (see decision)
    generate.py policy.py uplift.py qini.py monitor.py   # DOMAIN — churn extensions
  skills/ or .claude/              # L4/L5 — agent layer (greenfield, phase 2)
    commands/                      # one playbook per stage
    agents/                        # subagents (target-designer, CLI-wrappers, validators)
  adapters/                        # L1 — MCP data-source + inference (greenfield, phase 2)
  tests/                           # carried + new (validate-artifact, engineer, hp-search)
  docs/                            # new PRD/ADRs (reverse ADR-001 + ADR-009)
  pyproject.toml
```

---

## 7. Build order (factory §12 recipe, adapted — churnpilot did the hard part)

**MVP (factory's minimal-viable = Steps 1–2 + one hand-driven stage):** churnpilot *already* delivers most
of this; the MVP is essentially *"churnpilot's pipeline re-emitting heavy-tier, lineage-verified,
markdown artifacts."*

1. **Scaffold + lift the KEEP core.** New package; carry `metrics / model_select / baseline / evaluate /
   split / profile / config / source / recommend / charts / report` + their tests. Get green.
2. **Build the spine (L2).** Upgrade `ArtifactBase` → heavy tier + `validate-artifact` walker/probe +
   `export-schemas` + delete-on-failure. *(The factory's "get this right before anything else.")*
3. **Decouple Stage 4.** Rebuild `engineer-features` as a standalone transform-registry stage emitting a
   `feature-spec` artifact (fit-on-train/apply-outward).
4. **Wire the CLI.** `--json` + structured stderr; add `validate-artifact` / `gen-model-card`; split
   `engineer` from `train`. → **End of MVP: an auditable, lineage-verified deterministic factory.**
5. **The agent layer (L4/L5).** One skill/playbook per stage + subagents (target-designer,
   leakage-scanner, model-recommender = the crown jewels; plus no-retry CLI wrappers). MCP adapters (L1).
6. **hp-search → Optuna**, GBM engines, SHAP/PDP (compute depth).
7. **Distribution (L6)** — content-sha bundles + installer — last.

---

## 8. Open decisions for B (need your call)

1. **Domain scope.** Is the new project a **generic factory** (churn = one plugged-in reference domain /
   end-to-end demo, à la the toy-loans in Appendix E) — or a **churn-specific factory** (churn done the
   factory way)? This flips `generate/policy/uplift/qini/monitor` between *example-only* and *core*.
2. **Project name / package name** (churnpilot → `?`).
3. **First target: MVP (Steps 1–4) or straight for the agent layer.** Recommended: **MVP first** — it's
   ~half-built already and is the factory's own advice ("build the contracts + deterministic core first;
   the agents are the easy part").
4. **Fresh repo vs in-place** in `AI&DS` (you chose fresh; confirming git-init here).
```

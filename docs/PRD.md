# mlfactory — PRD

An LLM-orchestrated **ML factory**: turn a dataset into a trained, validated, documented model via a
pipeline of re-runnable stages where the LLM decides and judges, a deterministic tested CLI computes,
and typed lineage artifacts are the contract. Architecture: [`../ml-factory-architecture.md`](../ml-factory-architecture.md).
Decisions: [`ADRs.md`](ADRs.md). Build plan: [`../REUSE-MAP.md`](../REUSE-MAP.md).

## 1. Goals
1. A **portfolio piece** for Data Scientist / Research / Applied / ML-platform roles — defensible
   line-by-line, and demonstrating the agent + tested-CLI + typed-artifacts pattern at factory scale.
2. A **genuinely retargetable** factory: swap the domain, keep the core. The bundled reference domain
   is **B2B SaaS account churn** (it proves the factory end-to-end; it is not the point).

## 2. Scope — the layered pipeline
`saved-dataset → eda-exploration → feature-spec → dataset → model`, on the layered model of the
blueprint: L3 deterministic CLI · L2 typed lineage artifacts + `validate-artifact` · L4/L5 orchestrator
playbooks + specialist subagents · L1 data/inference adapters · L6 bundle distribution.

## 3. Layer 1 — the deterministic ML engine
*Acceptance = unit/integration tests. The tested metric core imports only numpy/pandas.*

| Requirement | Status |
|---|---|
| Clean-room metric suite (decile-KS, PSI frozen-edges, ROB, lift/gain, ROC/PR-AUC, log-loss, calibration) | ✅ lifted |
| Leakage-safe split (time/grouped/random + stratified) with a leakage guard | ✅ lifted (stratified TODO) |
| Standalone feature-engineering: closed transform registry, fit-on-train/apply-outward, `feature-spec` | ✅ #2 |
| Model menu + baseline floor + stability-based selection | ✅ lifted |
| Held-out evaluation: union metric pack + slices + calibration | ✅ lifted |
| Hyper-parameter search (Optuna TPE) + `hist_gbm` engine | ✅ #4 |

## 4. Layer 2 — contracts & state (the spine)
*Acceptance = schema/lineage checks.*

| Requirement | Status |
|---|---|
| Heavy `ArtifactBase` (markdown-frontmatter, lineage `parent`, `verification`, versioning) | ✅ #1 |
| `validate-artifact --walk-lineage --probe-output` + delete-on-failure rollback | ✅ #1 (walker/probe) |
| `export-schemas --check` (JSON-Schema CI-synced to the pydantic source) | ✅ #1 |
| Stage artifacts: `saved-dataset` · `feature-spec` · `eda-exploration` ✅ · `dataset`/`model` heavy artifacts | ⏳ (model stage uses the lifted `ModelCard`) |

## 5. Layer 3 — the CLI tool surface
*Acceptance = CLI E2E tests.*

| Requirement | Status |
|---|---|
| One command per capability; `--json` machine output + structured errors (the tool surface subagents shell out to) | ✅ #3 (train · engineer-features · advise · leakage-scan) |
| `gen-model-card` (markdown model card — the DS go/no-go surface) | ✅ #3 |
| `validate-artifact`, `export-schemas`, `engineer-features` | ✅ #1–#2 |

## 6. Layer 4/5 — agent behavior (✅ #10–#12, epic #5 complete)
Multi-agent orchestration (ADR-001), under `.claude/`: the `/mlfactory-run` + `/mlfactory-eda`
orchestrator playbooks; judgment subagents (`leakage-scanner`, `model-recommender`, `column-profiler`)
+ no-retry CLI-wrappers; and the human-in-the-loop gates (`/mlfactory-gates`, `mlfactory-advisor`) —
**AI proposes, human decides**. The deterministic-tool boundary (ADR-002) and adversarial-verify
discipline throughout.

## 7. The reference domain (B2B SaaS account churn)
A deterministic synthetic account-month panel (seats/MRR/product-usage/logins/discounts/support) with
churn as the target, a planted leakage trap, drift, imbalance, missingness, and a randomized
retention-offer uplift layer — so every stage can be exercised end-to-end on safe, controllable data.

## 8. Non-goals
- Not general AutoML; a well-understood fixed DAG, not a free-roaming agent.
- Not a production MLOps platform (though the heavy contract tier is a clean bolt-on toward one).
- No real customer data / PII — the reference domain is synthetic.

## 9. Requirements → issues (all closed)
Contracts/CLI spine (#1, #3) · feature stage (#2) · reorg/decouple (#7) · compute depth (#4) · agent
layer (#10–#12, epic #5). Bundle distribution (#6) dropped as out of scope; MCP adapters deferred.
Live state: [`../STATUS.md`](../STATUS.md).

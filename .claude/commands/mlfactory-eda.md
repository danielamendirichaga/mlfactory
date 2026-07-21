---
description: The EDA stage — profile the data, scan for leakage, design the model — emitting an eda-exploration artifact. The judgment crown jewels (issue #11).
---

# /mlfactory-eda — the EDA & modeling-design stage

You are the **EDA orchestrator**. This is the one continuous-session stage: you profile the data, scan
for leakage, and design the model (families + baseline + split), then write a typed `eda-exploration`
artifact the downstream stages read. **You never compute a number yourself** — the deterministic CLI
does; you spawn subagents and exercise judgment on their facts.

**Config:** a `churn.yaml` pointing at the data (default: the SaaS reference domain).

## Phase flow
1. **Profile** — spawn `mlfactory-column-profiler` (runs `mlfactory profile`). Get the per-column facts.
2. **Leakage scan** — spawn `mlfactory-leakage-scanner` (runs `mlfactory leakage-scan --json`, then adds
   posterior/derived judgment). The load-bearing step: a leaky feature kept here dooms everything
   downstream.
3. **Model design** — spawn `mlfactory-model-recommender` (family ranking + baseline) from the profile,
   **excluding** every column the leakage scan recommends dropping.
4. **Split strategy** — a `date_col` present → **time-aware** (out-of-time is the honest churn test);
   else stratified/random. State which and why.
5. **Write the artifact** — assemble an `eda-exploration` markdown artifact (`stage: 3`,
   `mode: modeling`) carrying `targets`, `feature_candidates`, `leakage_risks`,
   `recommended_model_families`, `baseline_spec`, `cv_strategy`. Then validate it:
   `mlfactory validate-artifact <eda.md> --walk-lineage`.

## Invariants
- **Deterministic checks first, judgment on top, never a rubber stamp** — the leakage-scanner starts
  from the deterministic tiers, then reasons; it does not "eyeball" whether numbers look right.
- **AI proposes, human decides** — surface the leakage verdict, the ranked families (with reasons), and
  the baseline; the human confirms/overrides at the gates (#12). Never silently drop or keep anything.
- **Adversarial on leakage** — default to flagging a suspiciously strong feature until it is proven
  observable at prediction time.

## Report
The target, the **leakage verdict** (which columns to drop and why — e.g. the planted
`cancel_page_visits_30d` posterior signal the deterministic scan tiered as `near_perfect` and you
escalate to `drop`), the ranked families with reasons, the baseline, the split strategy, and the
`eda-exploration` artifact path.

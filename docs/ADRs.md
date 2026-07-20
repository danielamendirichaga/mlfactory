# mlfactory — Architecture Decision Records

Short, numbered records of the load-bearing calls. Each: context → decision → consequences.
Full architecture: [`../ml-factory-architecture.md`](../ml-factory-architecture.md). Reuse plan:
[`../REUSE-MAP.md`](../REUSE-MAP.md). Where these differ from the churnpilot ancestor, it is called out.

---

## ADR-001 — Multi-agent orchestration (one playbook per stage + specialist subagents)
> **Reverses churnpilot ADR-001 (single-agent).**

**Context:** the factory is a multi-stage pipeline (input → EDA → feature → dataset → model) where
each stage carries genuine, separable judgment — target design, leakage scanning, model-family
ranking — that benefits from isolated context and narrow tools.
**Decision:** one **orchestrator playbook/skill per stage** that spawns **narrow specialist
subagents** (designers/analyzers for judgment; no-retry CLI-wrappers for deterministic work), each
with its own context and tool allow-list.
**Consequences:** composability + testability (narrow agents, narrow tools); isolated contexts keep
the leak-hunt focused; more orchestration infrastructure than a single agent. The deterministic core
is built first; the agent layer is a thin skin over the CLI (blueprint §12).

## ADR-002 — The deterministic-tool boundary (AI decides & judges; the CLI computes)
**Context:** an LLM "eyeballing" a statistic cannot be audited or unit-tested.
**Decision:** every reproducible/numeric operation is a version-pinned CLI subcommand; the LLM only
decides, judges at gates, and invokes commands. **No number is ever reasoned out in a prompt.**
**Consequences:** reproducibility + auditability; agents read typed artifact fields, not free text.
This is the pattern the project demonstrates. (Carried from churnpilot ADR-002.)

## ADR-003 — Heavy contract tier (typed lineage artifacts + validate-artifact)
> **Reverses churnpilot ADR-009 (medium tier).**

**Context:** auditability and working-result enforcement require more than JSON sidecars.
**Decision:** the **heavy** tier — markdown-with-frontmatter artifacts (machine contract + human
body), a lineage `parent` pointer + `verification` block, versioned paths, JSON-Schema exports
(CI-checked), and the `validate-artifact --walk-lineage --probe-output` engine with delete-on-failure
rollback.
**Consequences:** any number traces back to its data and re-verifies; a failed upstream poisons the
chain. Built backward-compatibly so the lifted medium-tier artifacts keep working. (Shipped in #1.)

## ADR-004 — Generic core + swappable domain; the core hardcodes no domain columns
**Context:** a factory must retarget to any domain; a churn tool welded to churn columns cannot.
**Decision:** `compute/` is domain-agnostic and imports no domain package; columns that must never be
features (leakage / experiment-oracle) are declared in `config.exclude_columns`, which the **domain**
populates. Dependency direction is **core ← domain ← app**. The bundled reference domain is
**B2B SaaS account churn** (`domains/saas/`).
**Consequences:** the reference domain proves the factory end-to-end (as fintech did in the
blueprint) without entangling the core; a second domain is a new `domains/<x>/` package. (Shipped in #7.)

## ADR-005 — Leakage-out-at-the-source: fit-on-train / apply-outward
**Context:** target leakage is the #1 failure mode of a churn/loss model.
**Decision:** feature engineering is a standalone stage with a closed transform registry; stateful
transforms fit on **train only** and apply outward; `target_encoding` is CV-folded on train (no
self-leakage); the split-before-EDA path makes stateful transforms genuinely leakage-safe.
**Consequences:** leakage is prevented structurally, not caught late. (Shipped in #2.)

## ADR-006 — Beat-a-baseline-by-a-margin acceptance, with a multi-metric stability gate as the upgrade
**Context:** one lucky score must not decide shipping.
**Decision:** a candidate must beat a simple baseline by a set margin on the primary metric; the
stronger **multi-metric stability gate** (floor + train→holdout stability) exists as an upgrade for a
high-stakes port. The metric/threshold set lives behind a domain profile.
**Consequences:** honest, defensible acceptance; never a single number.

## ADR-007 — Swappable adapters (data-source + inference) behind one gateway
**Context:** the factory reaches the world through remote services.
**Decision:** a **read-only-in-depth** data-source adapter and a server-side **inference** adapter as
MCP servers (or SDK-tool equivalents), fronted by one auth gateway. These are the most swappable
parts; keep the interface, swap the implementation.
**Consequences:** the compute/contract spine is unaffected by the transport. (Planned — issue #5.)

## ADR-008 — Content-sha bundle distribution; library and prompts version independently
**Context:** the factory ships prompts + a Python library to many workstations.
**Decision:** deterministic content-sha bundles behind an installer; the declarative layer (prompts)
and the deterministic layer (CLI library) version separately.
**Consequences:** idempotent re-sync; ship a new CLI without touching a prompt. (Planned — issue #6.)

# mlfactory

**An LLM-orchestrated ML factory: the AI decides and judges, a deterministic, unit-tested CLI does
all the compute, and typed artifacts with lineage are the contract between stages.**

Turn a dataset into a trained, validated, documented model through a pipeline of re-runnable stages —
where every number is reproducible, every artifact is lineage-verified, and the LLM never silently
substitutes for a computation. Architecture blueprint: [`ml-factory-architecture.md`](ml-factory-architecture.md)
(plain-language intro: [`ml-factory-explained.md`](ml-factory-explained.md)).

> **Status: in progress — bootstrapping the deterministic core.** mlfactory is being built by lifting the
> proven, tested compute core of [churnpilot](../AI&DS_lab) (a churn/retention analysis tool built to the
> same thesis) and generalizing it into a domain-agnostic factory. **Churn is the bundled reference
> domain** that exercises the pipeline end-to-end.
>
> **Done:** the lifted deterministic core is green (metric suite, split + leakage guard, model menu,
> stability-based selection, held-out evaluation, typed artifacts with `parent_sha256` lineage, the churn
> reference domain) — 168 tests passing.
> **Roadmap** (see [`REUSE-MAP.md`](REUSE-MAP.md)): heavy-tier contracts (`validate-artifact` lineage
> walker + versioning + delete-on-failure), a standalone feature-engineering stage, an Optuna
> hyper-parameter search, and the LLM orchestration layer (per-stage skills + specialist subagents + MCP
> data/inference adapters).

---

## The idea

An ML result is only trustworthy if you can audit it. mlfactory splits the job so mistakes can't hide:

```
        ┌──────────────────────────┐        ┌───────────────────────────────┐
        │   AGENT (the LLM)         │        │   CLI (deterministic Python)  │
        │ • designs the target      │ calls  │ • split / engineer / train /  │
        │ • scans for leakage       │ ─────► │   search / score / validate   │
        │ • recommends & judges     │        │ • same input → same output    │
        │ • the human decides       │ ◄───── │ • unit-tested, seeded         │
        └──────────────────────────┘ reads  └───────────────────────────────┘
                     │      typed artifacts (frontmatter contract + lineage)   ▲
                     └─────────────────────────────────────────────────────────┘
```

- **The agent** handles judgment — designing the target, flagging leakage, ranking model families,
  narrating results — and never computes a number itself.
- **The tested CLI** owns every number. You can't unit-test a vibe, but you *can* unit-test
  `psi(identical) == 0`. Same input → same output, always.
- **Typed artifacts with lineage** connect the stages, so any number traces back to the data that
  produced it.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -e .
uv pip install --python .venv pytest ruff mypy types-PyYAML
.venv/bin/pytest -q                        # the lifted core, green
```

The generic factory CLI (`mlfactory <command>`) and the churn reference-domain demo are being wired up
per the roadmap. See [`REUSE-MAP.md`](REUSE-MAP.md) for the build plan.

---

## Provenance

- **Original / clean-room.** All compute is written from scratch, reimplementing standard public methods
  (decile-table KS, PSI, leakage-safe pipelines). Bootstrapped from the author's own churnpilot project.
- **Synthetic data only** in the churn reference domain — generated from a seed, no real data or PII.

## License

[MIT](LICENSE).

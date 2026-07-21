# mlfactory

**An LLM-orchestrated ML factory: the AI decides and judges, a deterministic, unit-tested CLI does
all the compute, and typed artifacts with lineage are the contract between stages.**

Turn a dataset into a trained, validated, documented model through a pipeline of re-runnable stages —
where every number is reproducible, every artifact is lineage-verified, and the LLM never silently
substitutes for a computation. Architecture blueprint: [`ml-factory-architecture.md`](ml-factory-architecture.md)
(plain-language intro: [`ml-factory-explained.md`](ml-factory-explained.md)).

> **Status: complete.** Bootstrapped from the author's own [churnpilot](../AI&DS_lab) (a churn/retention
> tool built to the same thesis) and generalized into a domain-agnostic factory. A **B2B SaaS
> account-churn** domain is the bundled reference domain that exercises the pipeline end-to-end — an
> account-month panel (product usage, seats, MRR, logins, discounts, support) with churn as the target
> and a retention-offer uplift layer.
>
> **What's built** — the deterministic compute core (metric suite, split + leakage guard, model menu,
> stability-based selection, held-out evaluation, Optuna TPE hp-search); the heavy contract tier
> (markdown-frontmatter artifacts + `validate-artifact` lineage-walk/probe + `export-schemas`); the
> standalone leakage-safe feature-engineering stage; the CLI tool surface (`--json`, `gen-model-card`,
> `leakage-scan`, `advise --json`); and the full **agent layer** under [`.claude/`](.claude/README.md)
> (per-stage orchestrator playbooks + judgment subagents + human-in-the-loop gates). **216 tests green.**
> Deferred / out of scope: MCP adapters (the local loader is the data adapter, Claude Code is the
> inference) and bundle distribution. See [`STATUS.md`](STATUS.md) and [`docs/PRD.md`](docs/PRD.md).

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

Then drive the pipeline via the CLI (or the agent layer's `/mlfactory-run` playbook):

```bash
mlfactory init                                  # scaffold churn.yaml (synthetic SaaS domain by default)
mlfactory leakage-scan --config churn.yaml      # tiered leakage risks — flags the planted trap
mlfactory split --config churn.yaml --strategy time --out-dir data/splits
mlfactory train --train data/splits/train.parquet --config churn.yaml --model logistic --json
mlfactory evaluate --model data/model.pkl --test data/splits/test.parquet --config churn.yaml
mlfactory gen-model-card --card data/model.card.json --eval data/eval-report.json
```

Plus `engineer-features` (Stage 4) + `validate-artifact` for the leakage-safe feature stage, and
`train --optuna` for a seeded Optuna search. See [`AGENTS.md`](AGENTS.md) for the full command map and
the [`.claude/`](.claude/README.md) agent layer.

---

## Provenance

- **Original / clean-room.** All compute is written from scratch, reimplementing standard public methods
  (decile-table KS, PSI, leakage-safe pipelines). Bootstrapped from the author's own churnpilot project.
- **Synthetic data only** in the B2B SaaS reference domain — generated from a seed, no real data or PII.

## License

[MIT](LICENSE).

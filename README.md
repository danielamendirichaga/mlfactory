# mlfactory

> **An LLM-orchestrated ML factory** — the AI decides and judges, a deterministic **unit-tested CLI**
> does all the compute, and **typed artifacts with lineage** are the contract between stages.

Turn a dataset into a trained, validated, documented model through a pipeline of re-runnable stages —
where **every number is reproducible, every artifact is lineage-verified, and the LLM never silently
substitutes for a computation.**

*Python 3.11+ · 284 tests green · ruff + mypy clean · MIT · runs on your own data or a bundled synthetic domain.*

New here and want the ideas before the code? Read the plain-language intro:
[**`ml-factory-explained.md`**](ml-factory-explained.md).

---

## Why this exists

An ML result is only worth as much as your ability to **audit** it. Two failure modes quietly ruin models:

- **Leakage** — the model accidentally trains on a column that encodes the answer, scores near-perfectly
  in testing, then collapses in production. It is the single most common way a churn/risk model is wrong.
- **Un-auditable automation** — letting an LLM "just do the ML" end-to-end is fast but impossible to
  reproduce or check. A language model *narrating* an AUC is not a fact you can re-run; a pinned CLI
  *computing* it is.

mlfactory's answer is a strict **division of labor** that makes mistakes hard to hide:

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

- **The agent** owns *judgment* — designing the target, hunting leakage, ranking model families,
  narrating results — and **never computes a number itself**.
- **The tested CLI** owns *every number*. You can't unit-test a vibe, but you *can* unit-test
  `psi(identical) == 0`. Same input → same output, always.
- **Typed artifacts with lineage** connect the stages, so any number traces back to the data that
  produced it — an sha256 parent chain that `validate-artifact` walks and re-verifies.

**Reach for it when** you want an auditable, reproducible data→model path; when leakage-safety,
typed contracts, or clean hand-off between steps matter; or as a **reference architecture** for the
"agent + tested CLI + typed artifacts" pattern. **It is not** general AutoML or a production MLOps
platform — it's a rigorously-built, fully-tested factory for tabular binary-classification (a bundled
synthetic domain, or point it at your own data).

---

## See it work — the leakage lesson in three commands

The bundled synthetic domain ships with a **planted leakage trap** (`cancel_page_visits_30d`), so you
can watch the factory catch a mistake that would sink a real model. All output below is real.

**1 — The EDA scan flags it before you ever train:**

```text
$ mlfactory leakage-scan --config churn.yaml
⚠ 1 leakage risk(s) — the EDA leakage-scanner should judge the posterior/derived cases:

  [near_perfect] cancel_page_visits_30d  strength +0.917 → inspect
      |corr|=0.917 in [0.9, 0.99) — verify it is observable at prediction time
```

**2 — Ignore the warning, train on `features: auto`, and the score is "too good to be true":**

```text
$ mlfactory train --model logistic --json
{"command": "train", "model": "logistic", "n_features": 21,
 "train_metrics": {"auc": 0.9999, "ks": 0.9877, "top_decile_lift": 9.903}, ...}
```

A **0.9999 AUC is not a triumph — it's the fingerprint of leakage.** The model is reading a column that
only spikes *because* the account is already cancelling.

**3 — The model card makes the weakness legible instead of shipping it:**

```text
## Performance                          ## Limitations
| test AUC | 0.9998 |                   - Trained/evaluated on the SYNTHETIC reference
| test KS  | 0.9503 |                     domain — not real production performance.
| ECE      | 0.0010 |                   - The score is a risk ranking, not a calibrated
| lift     | 9.50x  |                     business decision; pair it with the policy layer.
```

Drop the flagged column and the same pipeline reports honest, defensible numbers. **That's the whole
point: the factory surfaces the leak instead of laundering it into a great-looking score.** The full
model card renders 8 sections — Purpose · Training Data · Features · Performance · Calibration · Slices ·
Limitations · Lineage.

**And it's not just the demo:** the same pipeline runs on real data — validated end-to-end on the public
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset (held-out
**AUC 0.83**, a genuine result), with the model card honestly describing its *real-data* provenance.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -e .
uv pip install --python .venv pytest ruff mypy types-PyYAML
.venv/bin/pytest -q                     # 284 tests, green
```

Then run the whole pipeline — it works straight from `init` with **zero data setup** (the default
config points at the synthetic SaaS domain, generated deterministically on the fly):

```bash
mlfactory init                                       # scaffold churn.yaml (synthetic SaaS domain by default)
mlfactory leakage-scan --config churn.yaml           # tiered leakage risks — flags the planted trap
mlfactory split  --config churn.yaml --strategy time --out-dir data/splits
mlfactory train    --train data/splits/train.parquet --config churn.yaml --model logistic --model-out data/model.pkl --json
mlfactory evaluate --model data/model.pkl --test data/splits/test.parquet --config churn.yaml --report-out data/eval.json
mlfactory gen-model-card --card data/model.card.json --eval data/eval.json --output model-card.md
```

Every capability is one command; `--json` gives machine output for scripting; `validate-artifact`
re-checks any artifact's lineage and on-disk contents. Full command map: [`AGENTS.md`](AGENTS.md).

**Your own data:** point mlfactory at a file/DB with `mlfactory configure …` (or the guided
`/mlfactory-setup` playbook) instead of the synthetic default. **DS decisions** — the metric, operating
threshold, feature approach, ship bar, policy economics — are recorded with `mlfactory record-decision`
and read by every stage, so a call made at a gate actually propagates. Prefer to let the LLM drive? The
`/mlfactory-run` playbook orchestrates the same CLI.

---

## The two layers

**The deterministic CLI (`mlfactory/`)** — the tested engine. A clean-room metric suite, a leakage-safe
splitter, a standalone feature-engineering stage (fit-on-train / apply-outward), a model menu with a
baseline floor and a stability-based selection gate, held-out evaluation, and a seeded Optuna
hyper-parameter search — all behind typed, lineage-tracked artifacts and `validate-artifact`.

**The agent layer (`.claude/`)** — the orchestration. Per-stage **playbooks** (`/mlfactory-setup`,
`/mlfactory-eda`, `/mlfactory-run`) that spawn narrow **specialist subagents** (leakage-scanner,
column-profiler, model-recommender, no-retry CLI-wrappers). Its spine is **human-in-the-loop gates** —
*AI proposes with a deterministic recommendation, the human decides* — at every real judgment call
(target · leakage · feature approach · model · ship). Each decision is written to a **typed decision
record** (`config.decisions`) that downstream stages read, so a call at a gate actually changes what the
tool computes. The playbook *is* the program — no compiled orchestrator. See [`.claude/README.md`](.claude/README.md).

---

## Repository tour

| Path | What's there |
|---|---|
| [`mlfactory/compute/`](mlfactory/compute) | The **domain-agnostic engine** — metrics · split · model · evaluate · feature engineering · hp-search. The metric core imports only numpy/pandas. |
| [`mlfactory/artifacts/`](mlfactory/artifacts) | The **heavy contract tier** — typed lineage artifacts + `validate-artifact` (walk + sha256 probe) + JSON-Schema export. |
| [`mlfactory/domains/saas/`](mlfactory/domains/saas) | The **B2B SaaS reference domain** — a synthetic account-churn panel + policy / uplift / monitor extensions. |
| [`mlfactory/`](mlfactory) | App wiring — `config.py`, the Typer CLI (`cli.py`), `model_card.py`, reporting. |
| [`.claude/`](.claude/README.md) | The **agent layer** — orchestrator playbooks + specialist subagents. |
| [`docs/`](docs) | [`PRD.md`](docs/PRD.md) (product requirements) + [`ADRs.md`](docs/ADRs.md) (the load-bearing design decisions). |
| [`tests/`](tests) | 284 tests — metric properties, artifact-lineage round-trips, split guards, decision-record + gate coverage, a CLI E2E. |
| [`ml-factory-architecture.md`](ml-factory-architecture.md) | The full **design blueprint** this repo implements. |

---

## What this project demonstrates

- **Leakage-safe ML engineering** — a standalone feature stage with fit-on-train / apply-outward and
  CV-folded target encoding, plus a leakage scanner and a *planted trap the pipeline actually catches*.
- **Contracts & reproducibility** — typed, versioned, lineage-tracked artifacts with a lineage-walker,
  sha256 output probes, JSON-Schema exports (CI-checkable), and seeded end-to-end determinism.
- **Clean-room deterministic compute** — a metric suite (decile-KS, PSI on frozen edges, lift/gain,
  ROC/PR-AUC, log-loss, calibration/ECE) reimplemented from public methods and property-unit-tested.
- **LLM orchestration done responsibly** — agents are markdown playbooks + narrow subagents over a
  tested CLI; the deterministic-tool boundary means *no number is ever reasoned out in a prompt*.
- **AI proposes, the human decides** — every judgment call (target · leakage · features · model · ship)
  is surfaced at a gate with a deterministic recommendation and written to a typed decision record the
  downstream stages read; defaults are behaviour-preserving, so the gates add control without surprise.
- **Rigor that survives contact with real data** — validated on a real dataset (Telco churn), where the
  agent's adversarial checks surfaced genuine defects *and* verification caught a false alarm before it
  shipped — the point of separating judgment from tested computation.
- **Software discipline** — 284 tests, ruff + mypy clean, pydantic v2 contracts, a layered
  architecture (core ← domain ← app), and issue-driven delivery with recorded ADRs.

---

## Provenance & honesty

- **Original / clean-room.** All compute is written from scratch, reimplementing standard *public*
  methods (decile-table KS, PSI, leakage-safe pipelines) — no proprietary code.
- **Bootstrapped from the author's own [churnpilot](https://github.com/danielamendirichaga/churnpilot)**
  — a churn/retention tool built to the same thesis (single-agent). mlfactory generalizes its tested
  core into a domain-agnostic factory and adds the heavy contract tier + the multi-agent layer.
- **No data in the repo.** The bundled B2B SaaS reference domain is generated from a fixed seed, and run
  outputs are gitignored — no real customer data or PII is ever committed. Point the tool at your own data
  via `configure`; the synthetic numbers above are a *reference-domain demo*, not a production claim.

## Learn more

- [`ml-factory-explained.md`](ml-factory-explained.md) — plain-language: what an "ML factory" is, for a non-specialist.
- [`docs/running-with-claude-code.md`](docs/running-with-claude-code.md) — how to drive the pipeline with Claude Code (setup → eda → run, and the gates).
- [`ml-factory-architecture.md`](ml-factory-architecture.md) — the full architecture blueprint (the design this repo implements).
- [`docs/ADRs.md`](docs/ADRs.md) — why the load-bearing decisions were made (and where they reverse churnpilot's).
- [`AGENTS.md`](AGENTS.md) · [`STATUS.md`](STATUS.md) · [`CHANGELOG.md`](CHANGELOG.md) — how the project works, where it stands, what changed.

## License

[MIT](LICENSE).

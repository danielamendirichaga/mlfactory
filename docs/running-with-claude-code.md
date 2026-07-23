# Running mlfactory with Claude Code

mlfactory splits the ML job in two: **Claude decides and judges; a tested CLI does all the math; typed
artifacts connect the stages.** In Claude Code you drive the whole pipeline with **three slash commands**
and make the real calls at each **gate**.

## Setup (once)

```bash
cd path/to/mlfactory            # the cloned repo
claude                          # open Claude Code in the project
```

The `/mlfactory-*` commands live in [`.claude/`](../.claude/README.md), so they exist **only inside this
folder** (not on the claude.ai website). If the CLI isn't installed yet, see the [README](../README.md) quickstart.

## Run it (three commands)

| Type this | Claude does | You do |
|---|---|---|
| `/mlfactory-setup` | Interviews you for your data + target, writes a validated `churn.yaml` | Answer: data path, target, id/date/value columns *(skip for the synthetic demo)* |
| `/mlfactory-eda` | Profiles the data, scans for leakage, ranks model families, picks a split strategy | Confirm/override the leak drops, the split, and the model shortlist |
| `/mlfactory-run` | Runs the pipeline end-to-end (split → features → train → evaluate → card), validating each artifact | Confirm the feature approach + the ship decision; read the model card |

For your own data, run **`/mlfactory-setup` first**; on the built-in synthetic demo, skip it and start
with **`/mlfactory-eda`**, then **`/mlfactory-run`**.

## The one rule

**Claude never computes a number — the CLI does, and Claude pauses at "gates" so you decide.** Every
figure you see comes from a tested command and a lineage-checked artifact, never from Claude guessing.
Claude decides *what* to do; **you** make the judgment calls; the tool does the math.

## The stages, and where you (the DS) are in the loop

| Stage | The machine does | **You decide** |
|---|---|---|
| **1 · Get data** | Loads it, checks it's usable | What data, and what the target / "churn" means |
| **2 · EDA & design** | Profiles every column, scans leakage, ranks models | **Drop the leaks · split strategy · model shortlist** |
| **3 · Split** | Runs the split + leakage guard (no row in two splits) | *(already decided in stage 2 — you just review it)* |
| **4 · Features** | Builds a leakage-safe recipe (fit on train only); can construct ratios / interactions | **Feature approach** (skip vs. recipe) · which transforms to build |
| **5 · Train & select** | Fits candidates against a baseline floor, ranks on **stability** | **Which model to ship (stability > peak AUC)** |
| **6 · Evaluate** | Scores on the **untouched** test set (+ slices, calibration) | **Ship / don't ship** |
| **7 · Model card** | Renders the report from the artifacts (no new math) | Read it and sign off |
| **8–9 · Policy & monitor** *(SaaS extras)* | Targets who to save under a budget; flags drift — **never auto-retrains** | Set the budget · decide when to retrain |

The judgment is **front-loaded**: stage 2 (EDA) is where you make most calls, the middle stages are
largely deterministic, and stages 5–6 are the ship decisions.

## The five gates — where Claude always stops for you

**Target** (what you're predicting) → **Leakage** (which columns to drop) → **Feature approach**
(skip / recipe / hybrid) → **Model choice** (which family to ship, and the metric) → **Ship** (go / no-go
on the held-out card).

At every gate Claude shows the **evidence + a deterministic recommendation**, then **waits**. It never
drops, keeps, or ships anything on its own — *AI proposes, the human decides.*

Your gate decisions are **saved to the config** (`config.decisions`, via `mlfactory record-decision`) so
every downstream stage honors them — a choice isn't just a chat message, it changes what the tool computes.

---

*Prefer to type the commands yourself instead of letting Claude drive? Every stage is also a plain CLI
command — see the [README](../README.md) quickstart.*

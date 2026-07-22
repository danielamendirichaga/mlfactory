---
description: The human-in-the-loop gates — where the pipeline pauses for the DS to decide. AI proposes, human decides (issue #12).
---

# mlfactory — the human-in-the-loop gates

The load-bearing control principle of the whole factory: **AI proposes; the human decides.** The
orchestrators (`/mlfactory-run`, `/mlfactory-eda`) do the deterministic work autonomously but **pause
at every genuine judgment moment**, surface the evidence + a deterministic recommendation, and wait.

## The DS-leads posture (every gate obeys)
- **Propose, do not decide.** Present {what · why · the action you would take}, then **stop and wait**
  for the human's call. Never proceed past a gate on your own.
- **Never override an upstream human choice** unless the human passes an explicit override flag.
- **The recommendation is deterministic, not a vibe.** It comes from the tested `recommend.py` rules —
  surface them with `.venv/bin/mlfactory advise --json`. Judgment-*support*, not an LLM guess.
- **Non-interactive mode.** When the human opts in (as `mlfactory run --yes` does), take every
  recommendation without prompting — for scripting / CI, never the default.

## The five gates
| Gate | When | What is proposed | Evidence |
|---|---|---|---|
| **Target** | start of EDA | confirm the target column + task (from config) | the config + the target class balance |
| **Leakage exclusion** | after the leakage scan | the columns to drop (posterior / derived / perfect) | `leakage-scan` tiers + the scanner's escalation |
| **Feature approach** | before feature engineering | **skip** (train on raw) · **recipe** · **hybrid**, and which transforms | signal strength (max \|corr\|), skew, collinearity, missingness + the transform registry |
| **Model choice** | after `compare` | the family to ship — **stability over peak AUC** | the ranked `compare` table (held-out AUC + drop + score-PSI) |
| **Ship go/no-go** | after `evaluate` | ship / do-not-ship on the held-out card | AUC ≥ floor and ECE ≤ bar (`recommend_ship`) |

At each gate, spawn `mlfactory-advisor` to surface the recommendation, present it, and **wait for the
human**.

**Persisting a confirmed decision.** A gate's decision only takes effect once it is written where the
deterministic stages read it. For the **Leakage exclusion** gate that means the config, not the
artifact: on confirm, run `mlfactory exclude-columns --config churn.yaml --add <cols>`. `split`/`train`
read `config.exclude_columns` (via `feature_columns`), so **without this the model silently trains on
the "dropped" leak** — the `eda-exploration` artifact recording the decision does not change what
`train` sees. For the **Feature approach** gate, persist the choice with
`mlfactory record-decision --key features.approach --value <skip|recipe|hybrid>`; `/mlfactory-run`
branches on it (skip → train on the raw split; recipe/hybrid → `engineer-features` → `train --engineered`).

## Why this exists
A model that looks great in-sample and quietly collapses in production is the failure these gates
prevent. They put a human in the loop exactly where taste and context matter — the leak call, the
stability-vs-AUC trade-off, the go/no-go — while the machine does everything reproducible around them.

---
description: Guided config setup — interview the DS for their data + target, write a validated churn.yaml. The input-boundary gate, before /mlfactory-eda.
---

# /mlfactory-setup — point mlfactory at your data

You set up `churn.yaml` by **interviewing the human**, then writing it with the tested `configure`
command. **Never hand-edit the YAML** — the writer validates it, and `validate` confirms it against the
data. This is the input boundary, immediately before the Target gate.

## Interview (ask, do not assume)
1. **Where is the data?** A parquet/CSV file (→ `--source-kind file --path <p>`) or a Postgres/SQLite
   database (→ `--source-kind postgres --dsn <..> --table <..>`).
2. **The target** — which column to predict, and **what value means the event happened**
   (`--target <col> --positive-value <v>`). Confirm it is **binary** (a 1/0 outcome); mlfactory does not
   do regression/multiclass — flag it if the target is not binary.
3. **The id column** (`--id-col <col>`) — the unique row/entity id.
4. **Optional levers:** a **date column** (`--date-col`) unlocks the time-aware split + drift monitoring;
   a **customer-value column** (`--value-col`) unlocks the policy simulator.

## Write + validate (deterministic — the CLI does it, not you)
1. `mlfactory configure --source-kind <..> [--path <..> | --dsn <..> --table <..>] --target <col>
   --positive-value <v> --id-col <col> [--date-col <col>] [--value-col <col>]`
   — writes source+schema, validated; preserves any existing `decisions:` block.
2. `mlfactory validate --config churn.yaml` — **must report USABLE**. If it fails, surface the error and
   fix the mapping (wrong column name, missing file, degenerate/one-class target). Do not proceed on a
   bad config.

## Report + hand off
State the written source + target (+ its class balance from `validate`), then hand off:
**run `/mlfactory-eda` next** — it opens with the Target gate on the config you just wrote.

## Invariants
- **Interview, don't guess.** The data location + target need the human's knowledge.
- **The tested `configure` writer does the write; `validate` proves it.** Never free-hand-edit the YAML.
- **Binary classification only** — flag a non-binary target rather than forcing it.

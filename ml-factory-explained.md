# What an "ML Factory" Is — a plain-language guide

*A short companion to the technical blueprint. Assumes you've poked at Claude Code but don't live in AI day-to-day.*

## What it is

An "ML factory" is a pipeline that turns raw data into a machine-learning model you can actually **trust** — and it's assembled out of the exact parts you've met in Claude Code: **skills, subagents, MCP connections, and a command-line tool.** Understanding the factory and understanding those parts turn out to be the same thing, because the factory is *made of* them.

Why it needs to exist: doing ML by hand is slow and easy to get subtly wrong — the classic disaster is **leakage**, where the model secretly peeks at the answer during testing and then fails in real life. Letting an AI do the whole thing on its own is fast but impossible to audit. The factory's answer is a strict division of labor: **an AI coordinates and judges each step, the actual math is done by a plain reproducible program, and every step leaves a file the next step reads.**

## The pieces, in one breath

- **Context** — the AI's limited working memory for a task; keep it small and focused and the work comes out better.
- **A skill** — a saved playbook the AI follows the same way every time (what you invoke with a slash-command). **Each stage of the factory is one skill.**
- **Subagents** — helper AIs a skill spins up, each with its *own* fresh context and one narrow job, reporting back a tidy result. **The specialists inside each stage are subagents.**
- **MCP** — a guarded connector that lets the AI use outside systems only through a defined set of safe "tools" (e.g. a *read-only* database query). **It's how the factory touches real data and models without endangering them.**
- **A CLI** — an ordinary command-line program. **It does the math the AI must not improvise**, so the results are reproducible.

Now watch them work.

## The stages — what each is for, and who does the work

Five stations. Each is a **skill**; each hands the next a file that records exactly where it came from.

**1 · Get the data.** *Why it exists:* you can't model without a clean, traceable copy of the raw data. *What it delivers:* one data file plus a note of exactly which query produced it. The skill reaches the database only through a **read-only MCP** tool — it can read production but physically cannot change it.

**2 · Understand it, and decide what to predict.** *Why it exists:* most model failures are decided right here — a vague target or a hidden leak dooms everything downstream. *What it delivers:* an "analysis" file that names the thing to predict, lists the columns that are safe to use, and flags the dangerous ones. The skill coordinates several **subagents**, each with its own clean context:
- a **profiler** — describes every column: its type, how much is missing, anything odd.
- a **target-designer** — helps pin down *exactly* what you're predicting and how it's computed (e.g. "defaulted within 90 days," not just "bad customer").
- a **leakage-scanner** — the most important helper in the whole system: it flags any column that secretly encodes the answer, so it's dropped *before* training instead of discovered after launch.
- a **model-recommender** — proposes which model types are worth trying and roughly how to tune them, based on what the data looks like.

**3 · Prepare the features.** *Why it exists:* raw columns aren't model-ready, and preparing them carelessly is a classic way to leak future information backward into the past. *What it delivers:* a table of clean, model-ready features. The skill's **subagents** check each transformation — a **validator** confirms it didn't blow up, go all-blank, or leak — and then call the **CLI** to apply it identically every time.

**4 · Split the data.** *Why it exists:* the only honest test is on data the model has never seen, and the split has to respect reality (dates, repeat customers). *What it delivers:* three separate piles — *learn-from*, *tune-on*, *test-on* — plus a checklist proving they don't overlap. A **split-designer** subagent picks the right splitting rule; a **validator** subagent runs a battery of leakage-and-balance checks and refuses to pass a bad split.

**5 · Train, choose, and stress-test.** *Why it exists:* you want the best model that's actually trustworthy, not the luckiest one. *What it delivers:* the trained model, an honest scorecard on the untouched test pile, and a plain-language **model card** a human signs off on. Each subagent here calls the **CLI**:
- a **baseline-trainer** builds a deliberately simple model first, to set a floor the real one must beat.
- an **hp-search** helper tunes each candidate model type, trying many settings to find the best.
- an **evaluator** scores the winner on the test pile it has never seen, and checks it's *stable*, not just lucky.
- a **card-generator** writes the summary — what it is, how it performs, where it's weak.

Two rules run through all five stations: **nothing moves forward until it's checked** (fail an inspection and the half-finished work is deleted, not passed on), and **every file carries a chain of custody** — so any number can be traced back to the raw data.

## Why it's built out of exactly these parts

Each Claude Code primitive solves a concrete problem the pipeline has:
- **needs real data but must never damage it** → so it reaches the outside world only through gated, read-only **MCP** tools;
- **each stage must run identically and be re-runnable** → so each stage is a **skill**, not a one-off prompt that drifts;
- **some jobs (the leak-hunt above all) need a focused, un-distracted pass, and independent checks should run at once** → so they're **subagents**, each with its own small **context**;
- **the trustworthy parts — training, scoring, the pass/fail gate — can't be improvised by a language model** → so they live in a **CLI**;
- **you need to audit, pause, resume, or hand off the work** → so the stages connect through **files with their own history**, never a chat that fills up and gets lost.

## Two habits worth stealing

- **The AI proposes; a human decides at the moments that matter** — and the AI **never trusts code it just wrote until it has actually run it and watched it work.** Looking right isn't being right.
- **Connect steps with self-describing files, not chat memory.**

The whole system reduces to one sentence: **let the AI coordinate and judge, but push every calculation into a plain reproducible CLI, reach the outside world only through gated MCP tools, split the work into focused subagents, wrap each stage as a re-runnable skill, and connect the stages with files that carry their own history.** That's what turns "an AI did some ML" into "an ML result you can actually stand behind."

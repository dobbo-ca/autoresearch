# Autoresearch (Go) — Design

- **Date:** 2026-06-10
- **Status:** Approved (pending written-spec review)
- **Owner:** Chris Dobbyn

## 1. Summary

A local-first, autonomous overnight optimization agent written in Go. It implements
the generalized Karpathy "auto-research" loop — *baseline → hypothesize → change →
score → keep-or-revert → repeat* — over an arbitrary user-supplied asset and scorer.
The agent's "brain" is a local LLM running on the Mac's Apple Silicon GPU (Metal),
embedded in-process via cgo bindings to llama.cpp. The Go harness owns all control
flow; the model is asked only to propose one focused change per round.

This is **not** a port of Karpathy's GPU *training* code (`train.py`) to Metal. The
original repo has no orchestrator — the "researcher" is an external coding agent that
edits a single file. We are building that orchestrator in Go, generalized beyond ML
training, with a local model as the brain. Any ML-training use case becomes just one
possible *asset* whose scorer shells out to an existing trainer; nothing in the model
or optimizer needs porting.

## 2. Goals / Non-goals

### Goals (v1)
- Single Go binary (`ar`) that runs the optimization loop unattended overnight.
- Three-file project contract: locked instructions, editable asset, locked scorer.
- Local model inference on Apple Silicon GPU, embedded in-process (cgo + llama.cpp + Metal).
- Capacity-based model auto-selection with confirm-before-download and manual override.
- Harness-owned control flow; structured (grammar-constrained) model output.
- Git-backed keep/revert with a human-readable morning report.
- Robust to scorer crashes, NaNs, timeouts, and process restarts (resume).

### Non-goals (v1)
- Multiple files edited in a single round (v2).
- Parallel/concurrent rounds or multi-asset projects (v2).
- Distributed inference or multi-machine execution (v2/v3).
- Running the loop inside a Kubernetes sandbox (v3).
- Porting nanochat GPU training to Metal (out of scope entirely).
- Any web UI.

## 3. Background

Karpathy's `autoresearch` repo is three files: `train.py` (the single file an agent
edits), `prepare.py` (data + tokenizer), and `program.md` (instructions to the agent).
A general coding agent reads `program.md`, edits `train.py`, runs it, reads `val_bpb`,
keeps or discards, and loops on a fixed time budget.

The viral generalization reframes this for *any* objective: pick one asset, turn "is it
good?" into a single honest number, then loop overnight — change one thing, score it,
keep winners, trash losers. The three-file system becomes:

1. **Instructions** — human-locked. The goal in plain English plus the rules.
2. **Asset** — the only thing the agent may change.
3. **Scoring** — the objective measuring stick; readable by the agent, never editable.

This design builds an engine for that generalized loop.

## 4. Architecture

### 4.1 Components

| Component | Responsibility |
|---|---|
| **CLI** (`ar`) | Subcommands `init`, `run`, `report`. Flag/config parsing. |
| **Config** | Load/validate `autoresearch.toml`. |
| **Capacity** | Detect unified memory; map to a model tier; resolve/confirm/download GGUF. |
| **Brain** | Interface producing one `Proposal` per round. Embedded (default) and subprocess impls. |
| **Engine** | The loop. Owns control flow, comparison, keep/revert orchestration, stop conditions. |
| **Workspace** | Git-backed asset state: apply change, commit on keep, checkout on revert; enforce locks. |
| **Scorer** | Run the scoring command in a subprocess with timeout; parse a single number. |
| **Ledger** | Append-only `rounds.jsonl`; render `report.md`. |

### 4.2 Data flow (one round)

```
baseline(asset, score)
  -> Brain.Propose(instructions, asset, history, direction)
  -> validate target_file in asset globs
  -> write new_content to target_file
  -> Scorer.Run()            (subprocess, timeout, capture)
  -> better(score, baseline, direction)?
       keep  : git commit (msg = hypothesis); baseline = score
       revert: git checkout -- <asset files>
  -> Ledger.Append(round summary)
  -> stop if goal met / max_rounds / SIGINT, else repeat
```

## 5. The three-file contract

- **Instructions** (`instructions.md`, configurable): human-locked. Fed to the model
  verbatim every round as goal + rules. The engine never writes it.
- **Asset** (path globs in config): the only files the model may change. **One file per
  round in v1.** The model returns the full new content for one target file.
- **Scoring** (an executable, e.g. `score.sh` / `score.py`): run by the harness, read by
  the model, never written by the model.

### 5.1 Scorer contract

- Invoked with working directory = workspace root.
- Must exit `0` on a valid score.
- The score is read from stdout: if the last non-empty line parses as JSON containing a
  `score` field, that value is used; otherwise the last non-empty line is parsed as a
  float.
- `direction` in config is `"min"` (lower is better — e.g. latency ms, `val_bpb`) or
  `"max"` (higher is better — e.g. reply rate, CTR).
- Non-zero exit, timeout, `NaN`/`Inf`, or an unparseable score = **round failure** →
  revert and continue. Stdout/stderr are captured into the ledger for debugging.

## 6. Engine (the loop)

```
load config; resolve model (capacity or override); open Brain
ensure workspace is a git repo (init if absent); require clean tree
baseline = Scorer.Run()                 # establish starting number
Ledger.Append(baseline as round 0)

for round in 1.. :
    if SIGINT or goal_met(baseline) or max_rounds_reached: break
    asset   = read(asset_globs)
    prop    = Brain.Propose(instructions, asset, recent_history, direction)
    if prop.target_file not in asset_globs:    # model tried to escape its sandbox
        Ledger.Append(rejected); continue
    write(prop.target_file, prop.new_content)
    score, err = Scorer.Run()
    if err == nil and better(score, baseline, direction):
        git add -A; git commit -m prop.hypothesis
        baseline = score; kept = true
    else:
        git checkout -- asset_globs          # revert this round's edit
        kept = false
    Ledger.Append({round, hypothesis, target_file, before, after, kept, diffstat, logs})

render report.md
```

`better(a, b, dir)` = `a < b` for `min`, `a > b` for `max`. `goal_met` is optional
(config `goal`); when set, the loop stops once the baseline reaches/crosses it.

## 7. Brain

### 7.1 Interface

```go
type ProposeInput struct {
    Instructions string
    Asset        map[string]string // path -> current content
    History      []RoundSummary    // bounded recent rounds
    Direction    string            // "min" | "max"
}

type Proposal struct {
    Hypothesis string // one sentence: what is being changed and why
    TargetFile string // must be within the asset globs
    NewContent string // full replacement for TargetFile
}

type Brain interface {
    Propose(ctx context.Context, in ProposeInput) (Proposal, error)
    Close() error
}
```

### 7.2 Why harness-owns-the-loop + constrained output

Small local models are unreliable at open-ended agentic editing (multi-step tool use,
unified diffs, deciding when to stop). So the harness owns the loop and the model does
exactly one bounded task per round: propose one change. The model returns **full file
content** (not a diff — diffs are too fragile for local models) for **one** target file,
emitted as JSON constrained by a **GBNF grammar** so the output always parses. No regex
scraping of free-form model text.

### 7.3 Embedded implementation (default)

- Vendored llama.cpp (git submodule), built with the Metal backend, linked via cgo.
- Thin cgo wrapper exposing: model load, context create, tokenize, grammar-constrained
  decode, detokenize, free.
- GBNF grammar pins output to `{"hypothesis": string, "target_file": string,
  "new_content": string}`.
- **Known gotcha:** the Metal shader resource (`default.metallib` / `ggml-metal.metal`)
  must be locatable at runtime. We embed it and set the resource path at init so the
  single binary is self-contained.

### 7.4 Subprocess implementation (fallback)

The same `Brain` interface, backed by an OpenAI-compatible HTTP endpoint
(`llama-server` or Ollama). Selected by config `model.backend = "subprocess"`. This
de-risks the cgo work: if embedded linking fights us, we ship subprocess-first and swap
later with zero engine changes. JSON output is enforced via the server's grammar/JSON
mode where available, with a parse-and-retry guard otherwise.

## 8. Capacity-based model selection

On `ar run`, unless `model.path` is set:

1. Detect unified memory (`sysctl hw.memsize`).
2. Map to a tier (q4 weights target roughly < ⅓ of RAM, leaving room for the OS, KV
   cache, and the scorer process):

   | RAM | Default model |
   |---|---|
   | ≤ 16 GB | Qwen2.5-Coder-7B-Instruct q4 |
   | ≤ 24 GB | Qwen2.5-Coder-14B-Instruct q4 |
   | ≤ 48 GB | Qwen2.5-Coder-32B-Instruct q4 |
   | 64 GB+ | Qwen2.5-Coder-32B-Instruct q4 (70B q4 opt-in) |

3. If the chosen GGUF is already cached, load it. If not, **print the model name and
   approximate download size and ask for confirmation before downloading** (no silent
   multi-GB pull). On confirm, download to the cache and load.
4. `--model <path>` / `model.path` always overrides auto-selection.

The tier table and download URLs live in a small built-in registry. Models are cached
under `~/.cache/autoresearch/models/`.

## 9. Workspace (git-backed keep/revert)

- The workspace is a git repository; `ar run` runs `git init` if absent and requires a
  clean tree before starting.
- A **kept** change is a commit whose message is the round's hypothesis — yielding a free
  diff history and a trivial morning report.
- A **reverted** change is undone with `git checkout -- <asset globs>`.
- The model may only write paths inside the asset globs. Any proposal targeting
  `instructions.md`, the scorer, or anything else is rejected and logged — the metric
  cannot be gamed by editing the measuring stick.

## 10. Configuration

`autoresearch.toml` at the workspace root:

```toml
[project]
name        = "landing-page-speed"
instructions = "instructions.md"     # human-locked
asset        = ["index.html", "styles.css"]  # globs; ONE edited per round (v1)
scorer       = "./score.sh"          # executable; prints final number
direction    = "min"                 # "min" or "max"
goal         = 800                    # optional stop threshold; omit for none
max_rounds   = 0                      # 0 = unlimited (until goal or SIGINT)
round_timeout = "10m"                # per-round scorer wall-clock cap

[model]
backend     = "embedded"             # "embedded" | "subprocess"
path        = ""                     # explicit GGUF override; empty = auto by capacity
context     = 16384
temperature = 0.7
endpoint    = "http://127.0.0.1:8080" # used only when backend = "subprocess"

[run]
history_window = 8                    # recent rounds shown to the model
```

## 11. CLI

- `ar init` — interactive setup mirroring the viral prompt's interview: asks what asset
  to optimize, helps define the single objective metric, runs the "fit check"
  (objective number? fast feedback? write access?), and scaffolds `instructions.md`, an
  asset stub, a `score.sh` stub, and `autoresearch.toml`.
- `ar run` — resolve/confirm the model, then run the loop until goal / max_rounds /
  SIGINT. Resumable.
- `ar report` — render `report.md` from `rounds.jsonl`.

## 12. Ledger & report

- `rounds.jsonl` — one JSON object per round: `{round, ts, hypothesis, target_file,
  score_before, score_after, kept, diffstat, scorer_exit, logs_path}`.
- `report.md` — a table (round #, change, before → after, kept/reverted) plus total
  improvement from baseline. Rendered by `ar report` and at clean shutdown.

## 13. Robustness (unattended overnight)

- **Scorer isolation:** subprocess with `round_timeout`; stdout/stderr captured. Crash /
  NaN / timeout = failed round → revert, keep going.
- **Resume:** on start, read `rounds.jsonl`; the baseline score is the last kept score
  (re-run the scorer if the ledger is empty). Git is already at the last kept state.
  `ar run` continues from there.
- **Lock enforcement:** the engine only writes asset-glob paths; locked files are never
  touched.
- **Clean stop:** SIGINT / goal / max_rounds stops the loop and writes the final report.

## 14. Testing strategy

- **Engine with a fake Brain:** a deterministic `Brain` plus a trivial project — asset
  is `value.txt` holding an integer, scorer prints `abs(target - value)`, direction
  `min`. The fake proposes a closer value (engine must keep + commit) then a worse value
  (engine must revert). Assert ledger entries, git history, and resume — all without an
  LLM.
- **Scorer parser:** unit tests for JSON-score, last-float, non-zero exit, NaN, timeout,
  garbage.
- **Capacity mapping:** unit tests for RAM → tier boundaries.
- **Brain:** subprocess impl validated against a local `llama-server` first (fast to
  stand up); embedded impl validated with a tiny GGUF in CI-skippable tests.

## 15. Future roadmap

### v2
- Multiple files edited per round; multi-asset projects.
- Parallel hypotheses per round (fan out N variations, keep the best).
- Richer reporting (trend charts, per-hypothesis attribution).

### v3
- Run the loop inside a **Kubernetes sandbox**: containerize the harness and run each
  scorer in an isolated pod. The real win is treating the scorer as untrusted code and
  isolating it, plus scaling rounds out across a cluster. The `Brain`, `Scorer`, and
  `Workspace` boundaries in v1 are drawn so this is an execution-substrate swap, not a
  rewrite.

## 16. Risks & open questions

- **cgo + Metal embedding** is the highest-effort, highest-risk piece (build flags,
  metallib resource path, cross-version llama.cpp API drift). Mitigation: the subprocess
  `Brain` fallback behind the same interface.
- **Local-model change quality**: a 32B coder may propose weak or repetitive
  hypotheses. Mitigation: feed bounded history to discourage repeats; full-file
  replacement keeps edits well-formed. Revisit prompt/temperature after first runs.
- **Scorer trust**: v1 runs the scorer with the user's own privileges (it's their
  machine and project). True isolation is the v3 Kubernetes-sandbox goal.

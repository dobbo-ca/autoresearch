# Autoresearch (Go) — Design

- **Date:** 2026-06-10
- **Status:** Approved (managed-runtime backend revision)
- **Owner:** Chris Dobbyn

## 1. Summary

A local-first, autonomous overnight optimization agent written in Go. It implements
the generalized Karpathy "auto-research" loop — *baseline → hypothesize → change →
score → keep-or-revert → repeat* — over an arbitrary user-supplied asset and scorer.
The agent's "brain" is a local LLM running on the Mac's Apple Silicon GPU (Metal). The
Go program **owns the model runtime**: it downloads a `llama-server` binary and a GGUF
model, launches the server as a supervised child process, talks to it over a local
OpenAI-compatible HTTP API, and shuts it down on exit. Nothing is compiled into the
binary (no cgo) and the user never runs a separate tool (no LM Studio / Ollama). The Go
harness owns all control flow; the model is asked only to propose one focused change per
round.

This is **not** a port of Karpathy's GPU *training* code (`train.py`) to Metal. The
original repo has no orchestrator — the "researcher" is an external coding agent that
edits a single file. We are building that orchestrator in Go, generalized beyond ML
training. Any ML-training use case becomes just one possible *asset* whose scorer shells
out to an existing trainer; nothing in the model or optimizer needs porting.

## 2. Goals / Non-goals

### Goals (v1)
- Single Go binary (`ar`) that runs the optimization loop unattended overnight.
- Three-file project contract: locked instructions, editable asset, locked scorer.
- Local model inference on Apple Silicon GPU via a **Go-managed `llama-server`
  subprocess** — `ar` downloads, launches, health-checks, and shuts it down. No cgo,
  nothing compiled in, no external tool to start.
- Capacity-based model auto-selection with confirm-before-download and manual override.
- Harness-owned control flow; structured (grammar-constrained) model output.
- Git-backed keep/revert with a human-readable morning report.
- Robust to scorer crashes, NaNs, timeouts, and process restarts (resume).

### Non-goals (v1)
- Multiple files edited in a single round (v2).
- Parallel/concurrent rounds or multi-asset projects (v2).
- Distributed inference or multi-machine execution (v2/v3).
- Running the loop inside a Kubernetes sandbox (v3).
- Compiling the model runtime into the binary (cgo/embedded llama.cpp). The runtime is a
  downloaded, supervised child process; a single static cgo binary is a possible v2
  nicety, not a v1 requirement.
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
| **Capacity** | Detect unified memory; map to a model tier; resolve/confirm/download the GGUF. |
| **Runtime** | Resolve/download the `llama-server` binary, launch it on the GPU, health-check readiness, expose its endpoint, and shut it down. |
| **Brain** | OpenAI-compatible HTTP client that asks the model for one `Proposal` per round. |
| **Engine** | The loop. Owns control flow, comparison, keep/revert orchestration, stop conditions. |
| **Workspace** | Git-backed asset state: apply change, commit on keep, checkout on revert; enforce locks; ignore ledger/report/logs. |
| **Scorer** | Run the scoring command in a subprocess with timeout; parse a single number; capture stdout/stderr. |
| **Ledger** | Append-only `rounds.jsonl` (incl. `logs_path`, `diffstat`); render `report.md`. |

### 4.2 Data flow (one round)

```
baseline(asset, score)
  -> Brain.Propose(instructions, asset, history, direction)   [HTTP -> managed llama-server]
  -> validate target_file in asset globs
  -> write new_content to target_file
  -> Scorer.Run()            (subprocess, timeout, capture stdout/stderr)
  -> better(score, baseline, direction)?
       keep  : git commit (msg = hypothesis); baseline = score
       revert: git checkout -- <asset files>
  -> Ledger.Append(round summary + per-round log path + diffstat)
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
  revert and continue. Stdout/stderr are written to a per-round log file
  (`logs/round-NNNN.log`) and its path recorded in the ledger for debugging.

## 6. Engine (the loop)

```
load config; resolve model (capacity or override); start runtime (managed llama-server)
ensure workspace is a git repo (init if absent); ensure .gitignore covers ledger/report/logs
require clean tree (ignored files don't count); baseline = Scorer.Run()
Ledger.Append(baseline as round 0)

for round in 1.. :
    if SIGINT or goal_met(baseline) or max_rounds_reached: break
    prevBaseline = baseline
    asset   = read(asset_globs)
    prop    = Brain.Propose(instructions, asset, recent_history, direction)
    if prop.target_file not in asset_globs:    # model tried to escape its sandbox
        Ledger.Append(rejected, ScoreBefore=prevBaseline); continue
    write(prop.target_file, prop.new_content)
    res = Scorer.Run(); write res.stdout/stderr to logs/round-NNNN.log
    if res.ok and better(res.score, baseline, direction):
        git add -A; git commit -m prop.hypothesis; baseline = res.score; kept = true
    else:
        git checkout -- asset_globs; kept = false
    Ledger.Append({round, ts, hypothesis, target_file, prevBaseline, after, kept,
                   scorer_exit, logs_path, diffstat})

render report.md
```

`better(a, b, dir)` = `a < b` for `min`, `a > b` for `max`. `goal_met` is optional
(config `goal`); when set, the loop stops once the baseline reaches/crosses it.

## 7. Runtime + Brain

### 7.1 Brain interface

```go
type ProposeInput struct {
    Instructions string
    Asset        map[string]string // path -> current content
    History      []RoundSummary
    Direction    string            // "min" | "max"
}
type Proposal struct {
    Hypothesis string `json:"hypothesis"`
    TargetFile string `json:"target_file"`
    NewContent string `json:"new_content"`
}
type Brain interface {
    Propose(ctx context.Context, in ProposeInput) (Proposal, error)
    Close() error
}
```

There is one Brain implementation: an OpenAI-compatible HTTP client. It posts a
system+user prompt to `{endpoint}/v1/chat/completions` with a `grammar` field (GBNF) and
`response_format: json_object`, then parses the returned content into a `Proposal`.
`llama-server` applies the model's own chat template server-side, so no per-model prompt
formatting is needed — Qwen2.5, Qwen3.6, Llama, etc. all work unchanged.

### 7.2 Why harness-owns-the-loop + constrained output

Small local models are unreliable at open-ended agentic editing. So the harness owns the
loop and the model does exactly one bounded task per round: propose one change. It returns
**full file content** (not a diff) for **one** target file, emitted as JSON constrained by
a GBNF grammar so the output always parses. `ParseProposal` additionally strips any
`<think>…</think>` block (Qwen3.6 retains thinking context) before parsing, as a defense
for the case where the server does not hard-enforce the grammar.

### 7.3 Runtime (managed llama-server)

The Go program owns the model server's lifecycle:

1. **Resolve the server binary.** If not cached, confirm-and-download the llama.cpp
   release artifact for macOS arm64 (`llama-bNNNN-bin-macos-arm64.zip`, Metal-enabled,
   shaders embedded), unzip it, and locate `llama-server`. Cache under
   `~/.cache/autoresearch/bin/`.
2. **Resolve the model.** Capacity-based GGUF selection with confirm-before-download
   (see §8), or an explicit `model.path` override.
3. **Launch.** Exec `llama-server --model <gguf> --host 127.0.0.1 --port <p>
   --n-gpu-layers 999 --ctx-size <n>` as a child process on a free port.
4. **Wait until healthy.** Poll `GET /health` until it returns 200 (or time out).
5. **Serve.** Hand the endpoint to the Brain for the duration of the run.
6. **Shut down.** On engine exit / SIGINT, terminate the child (SIGTERM, then SIGKILL on
   a grace timeout).

If `model.endpoint` is set in config, the runtime is skipped entirely and the Brain talks
to that already-running server (escape hatch for power users).

## 8. Capacity-based selection + downloads

On `ar run`, unless `model.path` is set:

1. Detect unified memory (`sysctl hw.memsize`).
2. Map to a tier (q4 weights target well under available RAM, leaving room for the OS, KV
   cache, and the scorer process):

   | RAM | Default model |
   |---|---|
   | ≤ 16 GB | Qwen2.5-Coder-7B-Instruct q4 |
   | ≤ 24 GB | Qwen2.5-Coder-14B-Instruct q4 |
   | > 24 GB | Qwen3.6-27B q4 (default for 32/64 GB machines) |

3. If the chosen GGUF is cached, use it. Otherwise **print the model name and approximate
   download size and ask for confirmation before downloading** (no silent multi-GB pull).
   On confirm, download to the cache and use it.
4. `--model <path>` / `model.path` always overrides auto-selection.

The same confirm-before-download applies to the `llama-server` binary (§7.3). Tier table
and download URLs live in a small built-in registry; the registry is overridable via
config so new models need no recompile. Models cache under
`~/.cache/autoresearch/models/`; the server binary under `~/.cache/autoresearch/bin/`.

## 9. Workspace (git-backed keep/revert)

- The workspace is a git repository; `ar run` runs `git init` if absent.
- On first setup it writes/commits a `.gitignore` covering `rounds.jsonl`, `report.md`,
  and `logs/` so the ledger, report, and per-round logs are **never** tracked. This is
  essential: it keeps `git add -A` (kept rounds) from tracking them and
  `git checkout -- .` (reverted rounds) from deleting accumulated ledger lines, and keeps
  the clean-tree check passing on resume.
- `RequireClean` errors only on changes to tracked files (ignored files don't count).
- A **kept** change is a commit whose message is the round's hypothesis — a free diff
  history and a trivial morning report.
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
backend     = "managed"              # "managed" (download+launch llama-server) | "external"
path        = ""                     # explicit GGUF override; empty = auto by capacity
endpoint    = ""                     # set => use this running server; empty + managed => launch one
context     = 16384
temperature = 0.7

[run]
history_window = 8                    # recent rounds shown to the model
```

## 11. CLI

- `ar init` — interactive setup mirroring the viral prompt's interview: asks what asset
  to optimize, helps define the single objective metric, runs the "fit check"
  (objective number? fast feedback? write access?), and scaffolds `instructions.md`, an
  asset stub, a `score.sh` stub, `.gitignore`, and `autoresearch.toml`.
- `ar run` — start the runtime, then run the loop until goal / max_rounds / SIGINT.
  Resumable.
- `ar report` — render `report.md` from `rounds.jsonl`.

## 12. Ledger & report

- `rounds.jsonl` — one JSON object per round: `{round, ts, hypothesis, target_file,
  score_before, score_after, kept, scorer_exit, logs_path, diffstat}`.
- `logs/round-NNNN.log` — captured scorer stdout/stderr for that round.
- `report.md` — a table (round #, change, before → after, kept/reverted) plus total
  improvement from baseline. Rendered by `ar report` and at clean shutdown.

## 13. Robustness (unattended overnight)

- **Scorer isolation:** subprocess with `round_timeout`; stdout/stderr captured to a
  per-round log file. Crash / NaN / timeout = failed round → revert, keep going.
- **Resume:** ledger/report/logs are gitignored, so a restarted `ar run` passes the
  clean-tree check, reads `rounds.jsonl`, sets the baseline to the last kept score, and
  continues. Git is already at the last kept state.
- **Lock enforcement:** the engine only writes asset-glob paths; locked files are never
  touched.
- **Clean stop:** SIGINT / goal / max_rounds stops the loop, shuts down the runtime, and
  writes the final report.

## 14. Testing strategy

- **Engine with a fake Brain:** a deterministic `Brain` plus a trivial project — asset
  is `value.txt` holding a number, scorer prints `abs(target - value)`, direction `min`.
  Tests: converges and keeps improvements; reverts regressions; an **interleaving**
  keep/revert run does not clobber the ledger; a **resume** run (engine started twice in
  the same dir) continues from the last kept score. All without an LLM.
- **Scorer parser:** unit tests for JSON-score, last-float, non-zero exit, NaN, timeout,
  garbage.
- **Capacity mapping:** unit tests for RAM → tier boundaries.
- **Runtime:** unit-test the health-wait against an `httptest` server that flips
  503→200, and the binary/model resolution with an injected downloader. The real
  exec+launch is a skippable integration test gated on a present `llama-server`.
- **Brain:** HTTP client tested against an `httptest` server returning canned content,
  including `<think>`-prefixed output to verify stripping.

## 15. Future roadmap

### v2
- Multiple files edited per round; multi-asset projects.
- Parallel hypotheses per round (fan out N variations, keep the best).
- Richer reporting (trend charts, per-hypothesis attribution).
- Optional embedded single-binary runtime (cgo + llama.cpp + Metal) for users who want
  zero external files — behind a build tag, not the default.

### v3
- Run the loop inside a **Kubernetes sandbox**: containerize the harness and run each
  scorer in an isolated pod. The real win is treating the scorer as untrusted code and
  isolating it, plus scaling rounds out across a cluster. The `Brain`, `Scorer`,
  `Runtime`, and `Workspace` boundaries in v1 are drawn so this is an execution-substrate
  swap, not a rewrite.

## 16. Risks & open questions

- **llama-server version/flag drift**: release artifact names and server flags
  (`--n-gpu-layers`, `--ctx-size`, `/health`) can change across llama.cpp releases.
  Mitigation: pin a known release build number in the registry; the `external` backend
  lets a user point at any server they manage if a pinned build misbehaves.
- **Process supervision**: a crashed/hung `llama-server` must not wedge the overnight
  run. Mitigation: health-check with timeout on launch; terminate-and-restart policy;
  surface server stderr into the run log.
- **Local-model change quality**: a 27B model may propose weak or repetitive
  hypotheses. Mitigation: feed bounded history to discourage repeats; full-file
  replacement keeps edits well-formed. Revisit prompt/temperature after first runs.
- **Scorer trust**: v1 runs the scorer with the user's own privileges (it's their
  machine and project). True isolation is the v3 Kubernetes-sandbox goal.

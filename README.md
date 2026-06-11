# autoresearch

A local, overnight optimization agent. It runs the generalized Karpathy auto-research
loop — baseline → propose one change → score → keep or revert → repeat — over an asset
you choose, scored by a measuring stick you control, driven by a local LLM that `ar`
itself downloads and runs on your Mac's GPU.

## The three files
- `instructions.md` — your goal and rules (the agent reads this; never edits it).
- asset file(s) — the only thing the agent changes, one file per round.
- `score.sh` — prints a single number; the agent reads it but never edits it.

## Quick start
    make build
    ./bin/ar init        # scaffold instructions.md, score.sh, asset, .gitignore, autoresearch.toml
    # edit instructions.md and score.sh for your objective
    ./bin/ar run         # downloads + launches the model, then loops until goal / max_rounds / Ctrl-C
    ./bin/ar report      # render report.md

## The model
`ar` owns the model runtime. On first `run` it asks to download (a) a prebuilt
`llama-server` (Metal) binary and (b) a GGUF model auto-selected for your unified memory,
then launches and supervises the server itself — no LM Studio, Ollama, or manual server.

- 64 GB Macs default to **Qwen3.6-27B q4**. Override with `--model /path/to/model.gguf`
  or `model.path` in config (download any GGUF yourself and point at it).
- Already running a server? Set `model.backend = "external"` and `model.endpoint`.

## Roadmap
- v2: multiple files per round, parallel hypotheses, richer reports, optional embedded
  single-binary runtime.
- v3: run the loop inside a Kubernetes sandbox for isolated scorer execution.

# Autoresearch (Go) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ar`, a Go binary that runs the generalized Karpathy auto-research loop (baseline → propose one change → score → keep/revert → repeat) over a user-supplied asset and scorer, driven by a local LLM that `ar` itself downloads, launches, and supervises on the Mac's Apple Silicon GPU.

**Architecture:** The Go harness owns all control flow; the model is asked only to propose one focused change per round and returns grammar-constrained JSON. Keep/revert is git-backed (with the ledger/report/logs gitignored so they survive reverts and enable resume). The model runs as a **Go-managed `llama-server` subprocess**: `ar` resolves/downloads a prebuilt `llama-server` (Metal) binary and a GGUF model, launches the server, health-checks it, talks to it over the local OpenAI-compatible HTTP API, and shuts it down on exit. No cgo, nothing compiled in, no external tool to start. An `external` backend lets a power user point at a server they already run.

**Tech Stack:** Go 1.23, stdlib (`net/http`, `os/exec`, `archive/zip`, `flag`), `github.com/BurntSushi/toml`, and the `git` CLI. No cgo.

---

## File Structure

```
go.mod
Makefile
cmd/ar/main.go                  # CLI dispatch (run/init/report)
cmd/ar/run.go                   # `ar run`: start runtime, wire engine, signals
cmd/ar/init.go                  # `ar init`: interview + scaffold 3-file project + .gitignore
cmd/ar/report.go                # `ar report`: render report.md
internal/config/config.go       # TOML config (+ managed/external backend) + Load + Validate
internal/scorer/scorer.go       # run scoring command, parse a single number, capture logs
internal/capacity/capacity.go   # RAM detection + model tier table (Qwen3.6 default)
internal/capacity/resolve.go    # resolve GGUF: cache lookup, confirm, download
internal/runtime/runtime.go     # resolve/download llama-server, launch, health-poll, shutdown
internal/brain/brain.go         # Brain interface, types, prompt builder, GBNF, ParseProposal
internal/brain/subprocess.go    # OpenAI-compatible HTTP Brain
internal/workspace/workspace.go # git apply/commit/revert + .gitignore + locks + diffstat
internal/ledger/ledger.go       # rounds.jsonl (+logs_path,+diffstat) append + report.md render
internal/engine/engine.go       # the loop, Better(), per-round logs, resume, stop conditions
```

Each file has one responsibility. `internal/brain` owns the shared `RoundSummary`/`Proposal`/`ProposeInput` types so the Brain and the engine reference one definition. There is **no cgo** anywhere; the model is a child process managed by `internal/runtime`.

---

## Task 1: Module scaffold + test loop

**Files:**
- Create: `go.mod`, `Makefile`, `internal/version/version.go`
- Test: `internal/version/version_test.go`

- [ ] **Step 1: Initialize the module**

Run:
```bash
go mod init github.com/dobbo-ca/autoresearch
go mod edit -go=1.23
```

- [ ] **Step 2: Write the failing test**

Create `internal/version/version_test.go`:
```go
package version

import "testing"

func TestString(t *testing.T) {
	if String() != "0.1.0-dev" {
		t.Fatalf("got %q, want %q", String(), "0.1.0-dev")
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `go test ./internal/version/`
Expected: FAIL — `undefined: String`.

- [ ] **Step 4: Write minimal implementation**

Create `internal/version/version.go`:
```go
// Package version reports the build version of ar.
package version

const v = "0.1.0-dev"

// String returns the current ar version.
func String() string { return v }
```

- [ ] **Step 5: Add a Makefile**

Create `Makefile`:
```makefile
.PHONY: test build
test:
	go test ./...
build:
	go build -o bin/ar ./cmd/ar
```

- [ ] **Step 6: Run test to verify it passes**

Run: `go test ./...`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add go.mod Makefile internal/version/
git commit -m "feat: module scaffold and test loop"
```

---

## Task 2: Config package

**Files:**
- Create: `internal/config/config.go`
- Test: `internal/config/config_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/config/config_test.go`:
```go
package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeTOML(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "autoresearch.toml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadValidManaged(t *testing.T) {
	p := writeTOML(t, `
[project]
name = "demo"
instructions = "instructions.md"
asset = ["value.txt"]
scorer = "./score.sh"
direction = "min"
goal = 0.0
max_rounds = 100
round_timeout = "30s"

[model]
backend = "managed"
context = 16384
temperature = 0.7

[run]
history_window = 8
`)
	cfg, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Project.Direction != "min" {
		t.Fatalf("direction = %q", cfg.Project.Direction)
	}
	if cfg.Timeout() != 30*time.Second {
		t.Fatalf("timeout = %v", cfg.Timeout())
	}
	if cfg.Project.Goal == nil || *cfg.Project.Goal != 0.0 {
		t.Fatalf("goal = %v", cfg.Project.Goal)
	}
}

func TestExternalBackendRequiresEndpoint(t *testing.T) {
	p := writeTOML(t, `
[project]
name = "demo"
instructions = "instructions.md"
asset = ["value.txt"]
scorer = "./score.sh"
direction = "min"
round_timeout = "30s"
[model]
backend = "external"
[run]
history_window = 8
`)
	if _, err := Load(p); err == nil {
		t.Fatal("expected error: external backend without endpoint")
	}
}

func TestValidateRejectsBadDirection(t *testing.T) {
	p := writeTOML(t, `
[project]
name = "demo"
instructions = "instructions.md"
asset = ["value.txt"]
scorer = "./score.sh"
direction = "sideways"
round_timeout = "30s"
[model]
backend = "managed"
[run]
history_window = 8
`)
	if _, err := Load(p); err == nil {
		t.Fatal("expected validation error for bad direction")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/config/`
Expected: FAIL — `undefined: Load`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/config/config.go`:
```go
// Package config loads and validates the autoresearch.toml project file.
package config

import (
	"fmt"
	"time"

	"github.com/BurntSushi/toml"
)

type Project struct {
	Name         string   `toml:"name"`
	Instructions string   `toml:"instructions"`
	Asset        []string `toml:"asset"`
	Scorer       string   `toml:"scorer"`
	Direction    string   `toml:"direction"`
	Goal         *float64 `toml:"goal"`
	MaxRounds    int      `toml:"max_rounds"`
	RoundTimeout string   `toml:"round_timeout"`
}

type Model struct {
	Backend     string  `toml:"backend"` // "managed" (default) | "external"
	Path        string  `toml:"path"`    // explicit GGUF override
	Endpoint    string  `toml:"endpoint"`// required when backend == "external"
	Context     int     `toml:"context"`
	Temperature float64 `toml:"temperature"`
}

type Run struct {
	HistoryWindow int `toml:"history_window"`
}

type Config struct {
	Project Project `toml:"project"`
	Model   Model   `toml:"model"`
	Run     Run     `toml:"run"`
}

// Backend returns the effective backend, defaulting to "managed".
func (c Config) Backend() string {
	if c.Model.Backend == "" {
		return "managed"
	}
	return c.Model.Backend
}

// Timeout returns the per-round scorer timeout. Defaults to 10m if unset/invalid.
func (c Config) Timeout() time.Duration {
	d, err := time.ParseDuration(c.Project.RoundTimeout)
	if err != nil || d <= 0 {
		return 10 * time.Minute
	}
	return d
}

// Load reads and validates a config file.
func Load(path string) (Config, error) {
	var c Config
	if _, err := toml.DecodeFile(path, &c); err != nil {
		return Config{}, fmt.Errorf("decode %s: %w", path, err)
	}
	if err := c.validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

func (c Config) validate() error {
	if c.Project.Name == "" {
		return fmt.Errorf("project.name is required")
	}
	if c.Project.Instructions == "" {
		return fmt.Errorf("project.instructions is required")
	}
	if len(c.Project.Asset) == 0 {
		return fmt.Errorf("project.asset must list at least one path/glob")
	}
	if c.Project.Scorer == "" {
		return fmt.Errorf("project.scorer is required")
	}
	if c.Project.Direction != "min" && c.Project.Direction != "max" {
		return fmt.Errorf("project.direction must be \"min\" or \"max\", got %q", c.Project.Direction)
	}
	switch c.Backend() {
	case "managed":
	case "external":
		if c.Model.Endpoint == "" {
			return fmt.Errorf("model.endpoint is required when model.backend = \"external\"")
		}
	default:
		return fmt.Errorf("model.backend must be \"managed\" or \"external\", got %q", c.Model.Backend)
	}
	return nil
}
```

- [ ] **Step 4: Add the dependency and run tests**

Run:
```bash
go get github.com/BurntSushi/toml@v1.4.0
go test ./internal/config/
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/config/ go.mod go.sum
git commit -m "feat: config loading and validation"
```

---

## Task 3: Scorer package

**Files:**
- Create: `internal/scorer/scorer.go`
- Test: `internal/scorer/scorer_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/scorer/scorer_test.go`:
```go
package scorer

import (
	"context"
	"testing"
	"time"
)

func TestParseScoreFloat(t *testing.T) {
	got, err := parseScore("noise line\n42.5\n")
	if err != nil || got != 42.5 {
		t.Fatalf("got %v err %v", got, err)
	}
}

func TestParseScoreJSON(t *testing.T) {
	// NOTE: interpreted string with a REAL newline so the JSON is the last line.
	got, err := parseScore("some log\n{\"score\": 0.873}\n")
	if err != nil || got != 0.873 {
		t.Fatalf("got %v err %v", got, err)
	}
}

func TestParseScoreRejectsNaN(t *testing.T) {
	if _, err := parseScore("NaN\n"); err == nil {
		t.Fatal("expected error for NaN")
	}
}

func TestParseScoreRejectsGarbage(t *testing.T) {
	if _, err := parseScore("not a number\n"); err == nil {
		t.Fatal("expected error for garbage")
	}
}

func TestRunSuccess(t *testing.T) {
	r := Run(context.Background(), "echo 7.0", t.TempDir(), 5*time.Second)
	if r.Err != nil || r.Score != 7.0 {
		t.Fatalf("score %v err %v", r.Score, r.Err)
	}
}

func TestRunCapturesStderr(t *testing.T) {
	r := Run(context.Background(), "echo oops 1>&2; echo 3.0", t.TempDir(), 5*time.Second)
	if r.Err != nil || r.Score != 3.0 {
		t.Fatalf("score %v err %v", r.Score, r.Err)
	}
	if r.Stderr == "" {
		t.Fatal("expected captured stderr")
	}
}

func TestRunNonZeroExitFails(t *testing.T) {
	r := Run(context.Background(), "echo 1.0; exit 3", t.TempDir(), 5*time.Second)
	if r.Err == nil {
		t.Fatal("expected error on non-zero exit")
	}
}

func TestRunTimeoutFails(t *testing.T) {
	r := Run(context.Background(), "sleep 5; echo 1.0", t.TempDir(), 200*time.Millisecond)
	if r.Err == nil {
		t.Fatal("expected timeout error")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/scorer/`
Expected: FAIL — `undefined: parseScore`, `undefined: Run`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/scorer/scorer.go`:
```go
// Package scorer runs the project's scoring command and extracts a single number.
package scorer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Result is the outcome of one scorer invocation.
type Result struct {
	Score    float64
	Stdout   string
	Stderr   string
	ExitCode int
	Err      error // non-nil means the round failed (revert and continue)
}

// Run executes command via `sh -c` in dir, with a wall-clock timeout, capturing
// stdout/stderr and parsing the score from stdout.
func Run(ctx context.Context, command, dir string, timeout time.Duration) Result {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	res := Result{Stdout: stdout.String(), Stderr: stderr.String()}
	if cmd.ProcessState != nil {
		res.ExitCode = cmd.ProcessState.ExitCode()
	}

	if ctx.Err() == context.DeadlineExceeded {
		res.Err = fmt.Errorf("scorer timed out after %s", timeout)
		return res
	}
	if runErr != nil {
		res.Err = fmt.Errorf("scorer exited non-zero: %w", runErr)
		return res
	}
	score, err := parseScore(res.Stdout)
	if err != nil {
		res.Err = err
		return res
	}
	res.Score = score
	return res
}

func parseScore(stdout string) (float64, error) {
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	last := ""
	for i := len(lines) - 1; i >= 0; i-- {
		if s := strings.TrimSpace(lines[i]); s != "" {
			last = s
			break
		}
	}
	if last == "" {
		return 0, fmt.Errorf("scorer produced no output")
	}
	var obj struct {
		Score *float64 `json:"score"`
	}
	if err := json.Unmarshal([]byte(last), &obj); err == nil && obj.Score != nil {
		return checkFinite(*obj.Score)
	}
	f, err := strconv.ParseFloat(last, 64)
	if err != nil {
		return 0, fmt.Errorf("could not parse score from %q", last)
	}
	return checkFinite(f)
}

func checkFinite(f float64) (float64, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, fmt.Errorf("score is not finite: %v", f)
	}
	return f, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/scorer/`
Expected: PASS (all seven tests).

- [ ] **Step 5: Commit**

```bash
git add internal/scorer/
git commit -m "feat: scorer runner, number parsing, log capture"
```

---

## Task 4: Capacity detection + model tiers

**Files:**
- Create: `internal/capacity/capacity.go`
- Test: `internal/capacity/capacity_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/capacity/capacity_test.go`:
```go
package capacity

import "testing"

func TestSelectTier(t *testing.T) {
	cases := []struct {
		gb   float64
		want string
	}{
		{8, "qwen2.5-coder-7b"},
		{16, "qwen2.5-coder-7b"},
		{24, "qwen2.5-coder-14b"},
		{32, "qwen3.6-27b"},
		{64, "qwen3.6-27b"},
		{128, "qwen3.6-27b"},
	}
	for _, c := range cases {
		if got := SelectTier(c.gb).ID; got != c.want {
			t.Errorf("SelectTier(%v) = %q, want %q", c.gb, got, c.want)
		}
	}
}

func TestTiersSortedAscending(t *testing.T) {
	for i := 1; i < len(Tiers); i++ {
		if Tiers[i].MaxGB <= Tiers[i-1].MaxGB {
			t.Fatalf("tiers not strictly ascending at %d", i)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/capacity/`
Expected: FAIL — `undefined: SelectTier`, `undefined: Tiers`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/capacity/capacity.go`:
```go
// Package capacity detects machine memory and maps it to a default model tier.
package capacity

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Tier maps an upper RAM bound to a default GGUF model.
type Tier struct {
	MaxGB  int     // inclusive upper bound; last tier is the catch-all
	ID     string  // stable model id
	Repo   string  // Hugging Face repo
	File   string  // GGUF filename within the repo
	SizeGB float64 // approximate download size, for the confirm prompt
}

// Tiers are evaluated in ascending MaxGB order. The final tier is the catch-all.
// Default for 32/64 GB machines is Qwen3.6-27B (dense, ~16 GB q4 — highest raw quality).
var Tiers = []Tier{
	{MaxGB: 16, ID: "qwen2.5-coder-7b", Repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", File: "qwen2.5-coder-7b-instruct-q4_k_m.gguf", SizeGB: 4.7},
	{MaxGB: 24, ID: "qwen2.5-coder-14b", Repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", File: "qwen2.5-coder-14b-instruct-q4_k_m.gguf", SizeGB: 9.0},
	{MaxGB: 1 << 30, ID: "qwen3.6-27b", Repo: "Qwen/Qwen3.6-27B-GGUF", File: "qwen3.6-27b-q4_k_m.gguf", SizeGB: 16.5},
}

// SelectTier returns the first tier whose MaxGB is >= ramGB.
func SelectTier(ramGB float64) Tier {
	for _, t := range Tiers {
		if ramGB <= float64(t.MaxGB) {
			return t
		}
	}
	return Tiers[len(Tiers)-1]
}

// DetectRAMGB returns total unified memory in GiB via `sysctl hw.memsize` (macOS).
func DetectRAMGB() (float64, error) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0, fmt.Errorf("sysctl hw.memsize: %w", err)
	}
	bytes, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse hw.memsize: %w", err)
	}
	return float64(bytes) / (1 << 30), nil
}
```

> **Note:** verify the exact `Repo`/`File` for the Qwen3.6-27B GGUF at implementation time
> (e.g. `Qwen/Qwen3.6-27B-GGUF` or an `unsloth/...-GGUF` mirror). The model is swappable
> via `model.path`, so an imperfect default never blocks a run.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/capacity/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/capacity/capacity.go internal/capacity/capacity_test.go
git commit -m "feat: capacity detection and model tier selection"
```

---

## Task 5: Brain types, prompt builder, grammar, ParseProposal

**Files:**
- Create: `internal/brain/brain.go`
- Test: `internal/brain/brain_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/brain/brain_test.go`:
```go
package brain

import (
	"strings"
	"testing"
)

func TestBuildMessagesIncludesAssetAndHistory(t *testing.T) {
	in := ProposeInput{
		Instructions: "Make value.txt approach 0.",
		Asset:        map[string]string{"value.txt": "10"},
		Direction:    "min",
		History: []RoundSummary{
			{Round: 1, Hypothesis: "try 8", TargetFile: "value.txt", Before: 10, After: 8, Kept: true},
		},
	}
	sys, user := BuildMessages(in)
	if !strings.Contains(sys, "lower is better") {
		t.Errorf("system prompt missing direction guidance:\n%s", sys)
	}
	if !strings.Contains(user, "value.txt") || !strings.Contains(user, "Make value.txt approach 0.") {
		t.Errorf("user prompt missing asset/instructions:\n%s", user)
	}
	if !strings.Contains(user, "try 8") {
		t.Errorf("user prompt missing history:\n%s", user)
	}
}

func TestGrammarMentionsAllFields(t *testing.T) {
	g := Grammar()
	for _, f := range []string{"hypothesis", "target_file", "new_content"} {
		if !strings.Contains(g, f) {
			t.Errorf("grammar missing field %q", f)
		}
	}
}

func TestParseProposalPlainJSON(t *testing.T) {
	p, err := ParseProposal(`{"hypothesis":"h","target_file":"a.txt","new_content":"x"}`)
	if err != nil || p.TargetFile != "a.txt" || p.NewContent != "x" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

func TestParseProposalStripsThinkBlock(t *testing.T) {
	in := "<think>let me reason {not json}</think>\n{\"hypothesis\":\"h\",\"target_file\":\"a\",\"new_content\":\"b\"}"
	p, err := ParseProposal(in)
	if err != nil || p.TargetFile != "a" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

func TestParseProposalExtractsEmbeddedJSON(t *testing.T) {
	p, err := ParseProposal("Sure!\n{\"hypothesis\":\"h\",\"target_file\":\"a\",\"new_content\":\"b\"}\nDone.")
	if err != nil || p.TargetFile != "a" {
		t.Fatalf("p %+v err %v", p, err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/brain/`
Expected: FAIL — `undefined: ProposeInput`, etc.

- [ ] **Step 3: Write minimal implementation**

Create `internal/brain/brain.go`:
```go
// Package brain defines the model interface that proposes one change per round,
// plus the shared types, prompt builder, JSON grammar, and proposal parser.
package brain

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// RoundSummary is a compact record of a past round, shown to the model as history.
type RoundSummary struct {
	Round      int
	Hypothesis string
	TargetFile string
	Before     float64
	After      float64
	Kept       bool
}

// ProposeInput is everything the model sees for one round.
type ProposeInput struct {
	Instructions string
	Asset        map[string]string // path -> current content
	History      []RoundSummary
	Direction    string // "min" | "max"
}

// Proposal is the model's single change for one round.
type Proposal struct {
	Hypothesis string `json:"hypothesis"`
	TargetFile string `json:"target_file"`
	NewContent string `json:"new_content"`
}

// Brain proposes one change per round.
type Brain interface {
	Propose(ctx context.Context, in ProposeInput) (Proposal, error)
	Close() error
}

// BuildMessages renders the system and user prompts for one round.
func BuildMessages(in ProposeInput) (system, user string) {
	goal := "lower is better"
	if in.Direction == "max" {
		goal = "higher is better"
	}
	system = fmt.Sprintf(`You are an optimization engineer running an overnight research loop.
Each round you propose exactly ONE change to ONE asset file to improve a single objective score (%s).
Rules:
- Change only one file, chosen from the asset files shown.
- Return the FULL new content of that file, not a diff.
- Make one focused, testable hypothesis per round; do not repeat changes that were already reverted.
- Reply ONLY with a JSON object: {"hypothesis": string, "target_file": string, "new_content": string}.`, goal)

	var b strings.Builder
	b.WriteString("# Instructions (locked, human-authored)\n")
	b.WriteString(in.Instructions)
	b.WriteString("\n\n# Current asset files\n")
	paths := make([]string, 0, len(in.Asset))
	for p := range in.Asset {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, p := range paths {
		fmt.Fprintf(&b, "\n## %s\n```\n%s\n```\n", p, in.Asset[p])
	}
	if len(in.History) > 0 {
		b.WriteString("\n# Recent rounds (do not repeat reverted ideas)\n")
		for _, h := range in.History {
			status := "reverted"
			if h.Kept {
				status = "kept"
			}
			fmt.Fprintf(&b, "- round %d [%s] %s (%s): %.6f -> %.6f\n",
				h.Round, status, h.Hypothesis, h.TargetFile, h.Before, h.After)
		}
	}
	b.WriteString("\nPropose the next single change now.")
	return system, b.String()
}

// Grammar returns a GBNF grammar that constrains output to the Proposal JSON shape.
func Grammar() string {
	return `root   ::= "{" ws "\"hypothesis\"" ws ":" ws string ws "," ws "\"target_file\"" ws ":" ws string ws "," ws "\"new_content\"" ws ":" ws string ws "}"
string ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* "\""
ws     ::= [ \t\n]*`
}

var thinkRE = regexp.MustCompile(`(?s)<think>.*?</think>`)

// ParseProposal extracts a Proposal from model text. It strips any <think>...</think>
// reasoning block (e.g. Qwen3.6) and tolerates surrounding prose by slicing to the
// outermost JSON object.
func ParseProposal(content string) (Proposal, error) {
	content = thinkRE.ReplaceAllString(content, "")
	var p Proposal
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &p); err == nil && p.TargetFile != "" {
		return p, nil
	}
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start >= 0 && end > start {
		if err := json.Unmarshal([]byte(content[start:end+1]), &p); err == nil && p.TargetFile != "" {
			return p, nil
		}
	}
	return Proposal{}, fmt.Errorf("could not parse proposal JSON from model output")
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/brain/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/brain/brain.go internal/brain/brain_test.go
git commit -m "feat: brain interface, prompt, grammar, proposal parser"
```

---

## Task 6: Subprocess Brain (HTTP, OpenAI-compatible)

**Files:**
- Create: `internal/brain/subprocess.go`
- Test: `internal/brain/subprocess_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/brain/subprocess_test.go`:
```go
package brain

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func canned(content string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}}},
		})
	}))
}

func TestSubprocessProposeParsesResponse(t *testing.T) {
	srv := canned(`{"hypothesis":"set to 5","target_file":"value.txt","new_content":"5"}`)
	defer srv.Close()
	b := NewSubprocess(srv.URL, "test-model", 0.7)
	p, err := b.Propose(context.Background(), ProposeInput{
		Instructions: "approach 0", Asset: map[string]string{"value.txt": "10"}, Direction: "min",
	})
	if err != nil || p.TargetFile != "value.txt" || p.NewContent != "5" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

func TestSubprocessStripsThinkAndExtracts(t *testing.T) {
	srv := canned("<think>hmm</think>\n{\"hypothesis\":\"x\",\"target_file\":\"a\",\"new_content\":\"b\"}")
	defer srv.Close()
	b := NewSubprocess(srv.URL, "m", 0.0)
	p, err := b.Propose(context.Background(), ProposeInput{Direction: "min"})
	if err != nil || p.TargetFile != "a" {
		t.Fatalf("p %+v err %v", p, err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/brain/ -run Subprocess`
Expected: FAIL — `undefined: NewSubprocess`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/brain/subprocess.go`:
```go
package brain

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Subprocess is a Brain backed by an OpenAI-compatible HTTP server
// (a managed or external llama-server running the model on the GPU).
type Subprocess struct {
	endpoint    string
	model       string
	temperature float64
	client      *http.Client
}

// NewSubprocess builds a Brain that calls {endpoint}/v1/chat/completions.
func NewSubprocess(endpoint, model string, temperature float64) *Subprocess {
	return &Subprocess{
		endpoint:    strings.TrimRight(endpoint, "/"),
		model:       model,
		temperature: temperature,
		client:      &http.Client{Timeout: 10 * time.Minute},
	}
}

func (s *Subprocess) Close() error { return nil }

func (s *Subprocess) Propose(ctx context.Context, in ProposeInput) (Proposal, error) {
	system, user := BuildMessages(in)
	body := map[string]any{
		"model":       s.model,
		"temperature": s.temperature,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"response_format": map[string]string{"type": "json_object"},
		"grammar":         Grammar(), // llama-server enforces; ignored elsewhere
	}
	buf, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.endpoint+"/v1/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return Proposal{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return Proposal{}, fmt.Errorf("call model: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Proposal{}, fmt.Errorf("model returned status %d", resp.StatusCode)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return Proposal{}, fmt.Errorf("decode response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return Proposal{}, fmt.Errorf("model returned no choices")
	}
	return ParseProposal(parsed.Choices[0].Message.Content)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/brain/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/brain/subprocess.go internal/brain/subprocess_test.go
git commit -m "feat: OpenAI-compatible HTTP brain backend"
```

---

## Task 7: Workspace (git apply/commit/revert + .gitignore + locks)

**Files:**
- Create: `internal/workspace/workspace.go`
- Test: `internal/workspace/workspace_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/workspace/workspace_test.go`:
```go
package workspace

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func newRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"}, {"config", "user.email", "t@example.com"}, {"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %s", args, out)
		}
	}
	return dir
}

func TestAllowedRejectsLockedAndOutside(t *testing.T) {
	w := New(newRepo(t), []string{"value.txt"}, []string{"instructions.md", "score.sh"})
	if !w.Allowed("value.txt") {
		t.Error("value.txt should be allowed")
	}
	if w.Allowed("instructions.md") || w.Allowed("score.sh") {
		t.Error("locked files must be rejected")
	}
	if w.Allowed("../escape.txt") {
		t.Error("path traversal must be rejected")
	}
}

func TestEnsureRepoWritesGitignore(t *testing.T) {
	dir := t.TempDir()
	w := New(dir, []string{"value.txt"}, nil)
	if err := w.EnsureRepo(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, ".gitignore"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"rounds.jsonl", "report.md", "logs/"} {
		if !contains(string(b), want) {
			t.Errorf(".gitignore missing %q:\n%s", want, b)
		}
	}
}

// Regression: an untracked, gitignored ledger must NOT make RequireClean fail (resume).
func TestRequireCleanIgnoresLedger(t *testing.T) {
	dir := t.TempDir()
	w := New(dir, []string{"value.txt"}, nil)
	if err := w.EnsureRepo(); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(dir, "value.txt"), []byte("1"), 0o644)
	if err := w.Commit("baseline"); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(dir, "rounds.jsonl"), []byte("{}\n"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "report.md"), []byte("# r\n"), 0o644)
	if err := w.RequireClean(); err != nil {
		t.Fatalf("RequireClean must ignore gitignored files: %v", err)
	}
}

func TestApplyCommitRevert(t *testing.T) {
	dir := newRepo(t)
	_ = os.WriteFile(filepath.Join(dir, "value.txt"), []byte("10"), 0o644)
	w := New(dir, []string{"value.txt"}, nil)
	if err := w.Commit("baseline"); err != nil {
		t.Fatal(err)
	}
	if err := w.Apply("value.txt", "5"); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "value.txt")); string(got) != "5" {
		t.Fatalf("apply failed: %q", got)
	}
	if err := w.RevertAsset(); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "value.txt")); string(got) != "10" {
		t.Fatalf("revert failed: %q", got)
	}
}

func TestReadAssetResolvesGlobs(t *testing.T) {
	dir := newRepo(t)
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("A"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "b.txt"), []byte("B"), 0o644)
	w := New(dir, []string{"*.txt"}, nil)
	m, err := w.ReadAsset()
	if err != nil {
		t.Fatal(err)
	}
	if m["a.txt"] != "A" || m["b.txt"] != "B" {
		t.Fatalf("globs not resolved: %+v", m)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/workspace/`
Expected: FAIL — `undefined: New`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/workspace/workspace.go`:
```go
// Package workspace manages the git-backed asset tree: applying the model's change,
// committing kept changes, reverting losers, enforcing writable files, and keeping the
// ledger/report/logs out of the tree (gitignored) so reverts and resume work.
package workspace

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// IgnoredPaths are written to .gitignore so the engine's working files are never tracked
// (a kept round's `git add -A` must not track them; a reverted round's `git checkout`
// must not delete them; and they must not trip the clean-tree check on resume).
var IgnoredPaths = []string{"rounds.jsonl", "report.md", "logs/"}

type Workspace struct {
	dir    string
	asset  []string // globs the model may write
	locked []string // files the model must never write
}

// New builds a Workspace over dir. asset and locked are paths/globs relative to dir.
func New(dir string, asset, locked []string) *Workspace {
	return &Workspace{dir: dir, asset: asset, locked: locked}
}

func (w *Workspace) git(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = w.dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, out)
	}
	return string(out), nil
}

// EnsureRepo runs `git init` if needed and guarantees .gitignore lists IgnoredPaths.
func (w *Workspace) EnsureRepo() error {
	if _, err := os.Stat(filepath.Join(w.dir, ".git")); err != nil {
		if _, err := w.git("init", "-q"); err != nil {
			return err
		}
	}
	return w.ensureGitignore()
}

func (w *Workspace) ensureGitignore() error {
	p := filepath.Join(w.dir, ".gitignore")
	existing, _ := os.ReadFile(p)
	lines := string(existing)
	var add []string
	for _, ig := range IgnoredPaths {
		if !lineContains(lines, ig) {
			add = append(add, ig)
		}
	}
	if len(add) == 0 {
		return nil
	}
	if len(lines) > 0 && !strings.HasSuffix(lines, "\n") {
		lines += "\n"
	}
	lines += strings.Join(add, "\n") + "\n"
	return os.WriteFile(p, []byte(lines), 0o644)
}

func lineContains(body, want string) bool {
	for _, l := range strings.Split(body, "\n") {
		if strings.TrimSpace(l) == want {
			return true
		}
	}
	return false
}

// RequireClean errors if any TRACKED file has uncommitted changes. Untracked and
// gitignored files (the ledger/report/logs) do not count.
func (w *Workspace) RequireClean() error {
	out, err := w.git("status", "--porcelain", "--untracked-files=no")
	if err != nil {
		return err
	}
	if strings.TrimSpace(out) != "" {
		return fmt.Errorf("workspace has uncommitted changes; commit or stash before running")
	}
	return nil
}

// Allowed reports whether the model may write rel.
func (w *Workspace) Allowed(rel string) bool {
	clean := filepath.Clean(rel)
	if strings.HasPrefix(clean, "..") || filepath.IsAbs(clean) {
		return false
	}
	for _, l := range w.locked {
		if clean == filepath.Clean(l) {
			return false
		}
	}
	for _, g := range w.asset {
		if ok, _ := filepath.Match(filepath.Clean(g), clean); ok {
			return true
		}
	}
	return false
}

// ReadAsset returns rel-path -> content for every file matching an asset glob.
func (w *Workspace) ReadAsset() (map[string]string, error) {
	out := map[string]string{}
	for _, g := range w.asset {
		matches, err := filepath.Glob(filepath.Join(w.dir, g))
		if err != nil {
			return nil, err
		}
		for _, m := range matches {
			b, err := os.ReadFile(m)
			if err != nil {
				return nil, err
			}
			rel, _ := filepath.Rel(w.dir, m)
			out[rel] = string(b)
		}
	}
	return out, nil
}

// Apply writes content to rel (caller must have checked Allowed).
func (w *Workspace) Apply(rel, content string) error {
	return os.WriteFile(filepath.Join(w.dir, rel), []byte(content), 0o644)
}

// Diffstat returns `git diff --stat` for the current uncommitted change (the round's edit).
func (w *Workspace) Diffstat() string {
	out, _ := w.git("diff", "--stat")
	return strings.TrimSpace(out)
}

// Commit stages everything (respecting .gitignore) and commits with msg.
func (w *Workspace) Commit(msg string) error {
	if _, err := w.git("add", "-A"); err != nil {
		return err
	}
	_, err := w.git("commit", "-q", "-m", msg)
	return err
}

// RevertAsset restores tracked files to the last commit.
func (w *Workspace) RevertAsset() error {
	_, err := w.git("checkout", "--", ".")
	return err
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/workspace/`
Expected: PASS (all tests, including the gitignore + RequireClean regression).

- [ ] **Step 5: Commit**

```bash
git add internal/workspace/
git commit -m "feat: git-backed workspace with gitignore, locks, diffstat"
```

---

## Task 8: Ledger (jsonl + logs_path/diffstat + report)

**Files:**
- Create: `internal/ledger/ledger.go`
- Test: `internal/ledger/ledger_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/ledger/ledger_test.go`:
```go
package ledger

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAppendAndAll(t *testing.T) {
	p := filepath.Join(t.TempDir(), "rounds.jsonl")
	l := Open(p)
	if err := l.Append(Record{Round: 1, Hypothesis: "h1", Kept: true, ScoreBefore: 10, ScoreAfter: 8, LogsPath: "logs/round-0001.log"}); err != nil {
		t.Fatal(err)
	}
	if err := l.Append(Record{Round: 2, Hypothesis: "h2", Kept: false, ScoreBefore: 8, ScoreAfter: 9}); err != nil {
		t.Fatal(err)
	}
	recs, err := l.All()
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 2 || recs[0].LogsPath != "logs/round-0001.log" || recs[1].Hypothesis != "h2" {
		t.Fatalf("got %+v", recs)
	}
}

func TestRenderShowsImprovement(t *testing.T) {
	out := Render([]Record{
		{Round: 0, Kept: true, ScoreAfter: 10},
		{Round: 1, Hypothesis: "h1", Kept: true, ScoreBefore: 10, ScoreAfter: 8},
	}, "min")
	if !strings.Contains(out, "h1") || !strings.Contains(out, "kept") {
		t.Fatalf("report missing row:\n%s", out)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/ledger/`
Expected: FAIL — `undefined: Open`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/ledger/ledger.go`:
```go
// Package ledger records each round to rounds.jsonl and renders a morning report.
package ledger

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Record is one round's full disk entry (spec §12).
type Record struct {
	Round       int     `json:"round"`
	TS          string  `json:"ts"`
	Hypothesis  string  `json:"hypothesis"`
	TargetFile  string  `json:"target_file"`
	ScoreBefore float64 `json:"score_before"`
	ScoreAfter  float64 `json:"score_after"`
	Kept        bool    `json:"kept"`
	ScorerExit  int     `json:"scorer_exit"`
	LogsPath    string  `json:"logs_path"`
	Diffstat    string  `json:"diffstat"`
}

type Ledger struct{ path string }

// Open returns a ledger appending to path (created on first Append).
func Open(path string) *Ledger { return &Ledger{path: path} }

// Append writes one record as a JSON line.
func (l *Ledger) Append(r Record) error {
	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	_, err = f.Write(append(line, '\n'))
	return err
}

// All reads every record back in order.
func (l *Ledger) All() ([]Record, error) {
	f, err := os.Open(l.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var recs []Record
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r Record
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return nil, err
		}
		recs = append(recs, r)
	}
	return recs, sc.Err()
}

// Render produces report.md content from records.
func Render(recs []Record, direction string) string {
	var b strings.Builder
	b.WriteString("# Autoresearch report\n\n")
	if len(recs) == 0 {
		b.WriteString("No rounds recorded yet.\n")
		return b.String()
	}
	baseline := recs[0].ScoreAfter
	best := baseline
	for _, r := range recs {
		if r.Kept {
			best = r.ScoreAfter
		}
	}
	fmt.Fprintf(&b, "Baseline: %.6f  →  Best: %.6f  (%s is better)\n\n", baseline, best, direction)
	b.WriteString("| Round | Change | Before | After | Result |\n")
	b.WriteString("|------:|--------|-------:|------:|--------|\n")
	for _, r := range recs {
		result := "reverted"
		if r.Kept {
			result = "kept"
		}
		fmt.Fprintf(&b, "| %d | %s | %.6f | %.6f | %s |\n",
			r.Round, escapePipes(r.Hypothesis), r.ScoreBefore, r.ScoreAfter, result)
	}
	return b.String()
}

func escapePipes(s string) string { return strings.ReplaceAll(s, "|", "\\|") }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/ledger/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/ledger/
git commit -m "feat: ledger with logs_path/diffstat and report rendering"
```

---

## Task 9: Engine (the loop)

**Files:**
- Create: `internal/engine/engine.go`
- Test: `internal/engine/engine_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/engine/engine_test.go`:
```go
package engine

import (
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/dobbo-ca/autoresearch/internal/brain"
	"github.com/dobbo-ca/autoresearch/internal/ledger"
	"github.com/dobbo-ca/autoresearch/internal/scorer"
	"github.com/dobbo-ca/autoresearch/internal/workspace"
)

func TestBetter(t *testing.T) {
	if !Better(1, 2, "min") || Better(2, 1, "min") {
		t.Error("min comparison wrong")
	}
	if !Better(2, 1, "max") || Better(1, 2, "max") {
		t.Error("max comparison wrong")
	}
}

// scoreSh writes a scorer that prints abs(value.txt) (direction "min").
func scoreSh(t *testing.T, dir string) {
	t.Helper()
	_ = os.WriteFile(filepath.Join(dir, "score.sh"),
		[]byte("#!/bin/sh\nawk '{x=$1; if(x<0)x=-x; print x}' value.txt\n"), 0o755)
}

func setupRepo(t *testing.T, start string) (string, *workspace.Workspace, *ledger.Ledger) {
	t.Helper()
	dir := t.TempDir()
	for _, a := range [][]string{{"init", "-q"}, {"config", "user.email", "t@t"}, {"config", "user.name", "t"}} {
		c := exec.Command("git", a...)
		c.Dir = dir
		_ = c.Run()
	}
	_ = os.WriteFile(filepath.Join(dir, "value.txt"), []byte(start), 0o644)
	scoreSh(t, dir)
	w := workspace.New(dir, []string{"value.txt"}, []string{"score.sh"})
	if err := w.EnsureRepo(); err != nil { // writes .gitignore
		t.Fatal(err)
	}
	if err := w.Commit("baseline"); err != nil {
		t.Fatal(err)
	}
	return dir, w, ledger.Open(filepath.Join(dir, "rounds.jsonl"))
}

func newEngine(dir string, w *workspace.Workspace, led *ledger.Ledger, b brain.Brain, goal *float64, max int) *Engine {
	return &Engine{
		Brain: b, Workspace: w, Ledger: led, Instructions: "drive to 0",
		Direction: "min", Goal: goal, MaxRounds: max, History: 8,
		LogsDir: filepath.Join(dir, "logs"), Now: func() time.Time { return time.Unix(0, 0) },
		RunScorer: func(ctx context.Context) scorer.Result { return scorer.Run(ctx, "sh score.sh", dir, 5*time.Second) },
	}
}

type halveBrain struct{}

func (halveBrain) Close() error { return nil }
func (halveBrain) Propose(_ context.Context, in brain.ProposeInput) (brain.Proposal, error) {
	cur, _ := strconv.ParseFloat(in.Asset["value.txt"], 64)
	return brain.Proposal{Hypothesis: "halve", TargetFile: "value.txt", NewContent: strconv.FormatFloat(cur/2, 'f', -1, 64)}, nil
}

func TestEngineKeepsImprovements(t *testing.T) {
	dir, w, led := setupRepo(t, "16")
	goal := 0.5
	if err := newEngine(dir, w, led, halveBrain{}, &goal, 20).Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(filepath.Join(dir, "value.txt"))
	if v, _ := strconv.ParseFloat(string(got), 64); math.Abs(v) > 0.5 {
		t.Fatalf("did not converge: %v", v)
	}
}

type doubleBrain struct{}

func (doubleBrain) Close() error { return nil }
func (doubleBrain) Propose(_ context.Context, in brain.ProposeInput) (brain.Proposal, error) {
	cur, _ := strconv.ParseFloat(in.Asset["value.txt"], 64)
	return brain.Proposal{Hypothesis: "double", TargetFile: "value.txt", NewContent: strconv.FormatFloat(cur*2, 'f', -1, 64)}, nil
}

func TestEngineRevertsRegressions(t *testing.T) {
	dir, w, led := setupRepo(t, "4")
	if err := newEngine(dir, w, led, doubleBrain{}, nil, 3).Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "value.txt")); string(got) != "4" {
		t.Fatalf("regressions not reverted: %q", got)
	}
}

// alternateBrain improves on odd rounds, regresses on even rounds, exercising
// interleaved keep/revert — the case that previously clobbered the ledger.
type alternateBrain struct{ n int }

func (a *alternateBrain) Close() error { return nil }
func (a *alternateBrain) Propose(_ context.Context, in brain.ProposeInput) (brain.Proposal, error) {
	a.n++
	cur, _ := strconv.ParseFloat(in.Asset["value.txt"], 64)
	next := cur / 2
	if a.n%2 == 0 {
		next = cur * 2 // regress -> must be reverted
	}
	return brain.Proposal{Hypothesis: "alt", TargetFile: "value.txt", NewContent: strconv.FormatFloat(next, 'f', -1, 64)}, nil
}

func TestEngineDoesNotClobberLedgerOnInterleave(t *testing.T) {
	dir, w, led := setupRepo(t, "64")
	if err := newEngine(dir, w, led, &alternateBrain{}, nil, 8).Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	recs, err := led.All()
	if err != nil {
		t.Fatal(err)
	}
	kept := 0
	for _, r := range recs {
		if r.Kept && r.Round > 0 {
			kept++
		}
	}
	if kept < 3 {
		t.Fatalf("kept rounds clobbered: only %d survived in %d records", kept, len(recs))
	}
	// And the ledger must not have broken the clean-tree check (resume gate).
	if err := w.RequireClean(); err != nil {
		t.Fatalf("RequireClean failed after run (resume would break): %v", err)
	}
}

func TestEngineResumesFromLastKept(t *testing.T) {
	dir, w, led := setupRepo(t, "16")
	goal := 1.0
	if err := newEngine(dir, w, led, halveBrain{}, &goal, 2).Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	first, _ := led.All()
	// Resume: a fresh engine in the same dir continues from the last kept score.
	led2 := ledger.Open(filepath.Join(dir, "rounds.jsonl"))
	goal2 := 0.5
	if err := newEngine(dir, w, led2, halveBrain{}, &goal2, 10).Run(context.Background()); err != nil {
		t.Fatalf("resume run failed: %v", err)
	}
	second, _ := led2.All()
	if len(second) <= len(first) {
		t.Fatalf("resume did not append rounds: first=%d second=%d", len(first), len(second))
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "value.txt")); func() bool { v, _ := strconv.ParseFloat(string(got), 64); return math.Abs(v) > 0.5 }() {
		t.Fatalf("resume did not converge: %q", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/engine/`
Expected: FAIL — `undefined: Engine`, `undefined: Better`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/engine/engine.go`:
```go
// Package engine runs the optimization loop: propose, score, keep-or-revert, log.
package engine

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/dobbo-ca/autoresearch/internal/brain"
	"github.com/dobbo-ca/autoresearch/internal/ledger"
	"github.com/dobbo-ca/autoresearch/internal/scorer"
	"github.com/dobbo-ca/autoresearch/internal/workspace"
)

// Engine wires the loop's collaborators.
type Engine struct {
	Brain        brain.Brain
	Workspace    *workspace.Workspace
	Ledger       *ledger.Ledger
	Instructions string
	Direction    string   // "min" | "max"
	Goal         *float64 // optional stop threshold
	MaxRounds    int      // 0 = unlimited
	History      int      // rounds of history shown to the model
	LogsDir      string   // per-round scorer logs
	Now          func() time.Time
	RunScorer    func(ctx context.Context) scorer.Result
	Log          func(string) // optional progress sink
}

// Better reports whether a improves on b for the given direction.
func Better(a, b float64, direction string) bool {
	if direction == "max" {
		return a > b
	}
	return a < b
}

func (e *Engine) logf(format string, args ...any) {
	if e.Log != nil {
		e.Log(fmt.Sprintf(format, args...))
	}
}

// Run executes the loop until the goal is met, MaxRounds is reached, or ctx is canceled.
func (e *Engine) Run(ctx context.Context) error {
	recs, err := e.Ledger.All()
	if err != nil {
		return err
	}
	baseline, round, err := e.resumeBaseline(ctx, recs)
	if err != nil {
		return err
	}

	for {
		if ctx.Err() != nil {
			e.logf("stopping: %v", ctx.Err())
			return nil
		}
		if e.goalMet(baseline) {
			e.logf("goal reached: %.6f", baseline)
			return nil
		}
		if e.MaxRounds > 0 && round >= e.MaxRounds {
			e.logf("max rounds reached (%d)", e.MaxRounds)
			return nil
		}
		round++
		prevBaseline := baseline

		asset, err := e.Workspace.ReadAsset()
		if err != nil {
			return err
		}
		prop, err := e.Brain.Propose(ctx, brain.ProposeInput{
			Instructions: e.Instructions,
			Asset:        asset,
			History:      e.recentHistory(recs),
			Direction:    e.Direction,
		})
		if err != nil {
			e.logf("round %d: propose failed: %v", round, err)
			recs = e.record(recs, ledger.Record{Round: round, TS: e.ts(), Hypothesis: "(propose failed)", Kept: false, ScoreBefore: prevBaseline, ScoreAfter: prevBaseline})
			continue
		}
		if !e.Workspace.Allowed(prop.TargetFile) {
			e.logf("round %d: rejected target %q (locked/outside asset)", round, prop.TargetFile)
			recs = e.record(recs, ledger.Record{Round: round, TS: e.ts(), Hypothesis: prop.Hypothesis, TargetFile: prop.TargetFile, Kept: false, ScoreBefore: prevBaseline, ScoreAfter: prevBaseline})
			continue
		}
		if err := e.Workspace.Apply(prop.TargetFile, prop.NewContent); err != nil {
			return err
		}
		res := e.RunScorer(ctx)
		logsPath := e.writeLog(round, res)
		diffstat := e.Workspace.Diffstat()

		kept := res.Err == nil && Better(res.Score, baseline, e.Direction)
		after := prevBaseline
		if res.Err == nil {
			after = res.Score
		}
		if kept {
			if err := e.Workspace.Commit(prop.Hypothesis); err != nil {
				return err
			}
			baseline = res.Score
			e.logf("round %d: KEPT %.6f  (%s)", round, baseline, prop.Hypothesis)
		} else {
			if err := e.Workspace.RevertAsset(); err != nil {
				return err
			}
			e.logf("round %d: revert (%s)", round, prop.Hypothesis)
		}
		recs = e.record(recs, ledger.Record{
			Round: round, TS: e.ts(), Hypothesis: prop.Hypothesis, TargetFile: prop.TargetFile,
			ScoreBefore: prevBaseline, ScoreAfter: after, Kept: kept,
			ScorerExit: res.ExitCode, LogsPath: logsPath, Diffstat: diffstat,
		})
	}
}

func (e *Engine) goalMet(baseline float64) bool {
	if e.Goal == nil {
		return false
	}
	if e.Direction == "max" {
		return baseline >= *e.Goal
	}
	return baseline <= *e.Goal
}

func (e *Engine) resumeBaseline(ctx context.Context, recs []ledger.Record) (float64, int, error) {
	round := 0
	for _, r := range recs {
		if r.Round > round {
			round = r.Round
		}
	}
	for i := len(recs) - 1; i >= 0; i-- {
		if recs[i].Kept {
			return recs[i].ScoreAfter, round, nil
		}
	}
	res := e.RunScorer(ctx)
	if res.Err != nil {
		return 0, 0, fmt.Errorf("baseline scorer failed: %w", res.Err)
	}
	if err := e.Ledger.Append(ledger.Record{Round: 0, TS: e.ts(), Hypothesis: "(baseline)", Kept: true, ScoreAfter: res.Score, ScoreBefore: res.Score}); err != nil {
		return 0, 0, err
	}
	return res.Score, 0, nil
}

func (e *Engine) writeLog(round int, res scorer.Result) string {
	if e.LogsDir == "" {
		return ""
	}
	if err := os.MkdirAll(e.LogsDir, 0o755); err != nil {
		return ""
	}
	name := fmt.Sprintf("round-%04d.log", round)
	body := "# stdout\n" + res.Stdout + "\n# stderr\n" + res.Stderr
	if res.Err != nil {
		body += "\n# error\n" + res.Err.Error()
	}
	if err := os.WriteFile(filepath.Join(e.LogsDir, name), []byte(body), 0o644); err != nil {
		return ""
	}
	return filepath.Join("logs", name)
}

func (e *Engine) recentHistory(recs []ledger.Record) []brain.RoundSummary {
	n := e.History
	if n <= 0 {
		n = 8
	}
	start := 0
	if len(recs) > n {
		start = len(recs) - n
	}
	out := make([]brain.RoundSummary, 0, len(recs)-start)
	for _, r := range recs[start:] {
		out = append(out, brain.RoundSummary{
			Round: r.Round, Hypothesis: r.Hypothesis, TargetFile: r.TargetFile,
			Before: r.ScoreBefore, After: r.ScoreAfter, Kept: r.Kept,
		})
	}
	return out
}

func (e *Engine) record(recs []ledger.Record, r ledger.Record) []ledger.Record {
	_ = e.Ledger.Append(r)
	return append(recs, r)
}

func (e *Engine) ts() string {
	if e.Now == nil {
		return ""
	}
	return e.Now().UTC().Format(time.RFC3339)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/engine/`
Expected: PASS — Better, KeepsImprovements, RevertsRegressions, **DoesNotClobberLedgerOnInterleave**, **ResumesFromLastKept**.

- [ ] **Step 5: Commit**

```bash
git add internal/engine/
git commit -m "feat: optimization loop with per-round logs, diffstat, resume"
```

---

## Task 10: Model resolution (cache, confirm, download)

**Files:**
- Create: `internal/capacity/resolve.go`
- Test: `internal/capacity/resolve_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/capacity/resolve_test.go`:
```go
package capacity

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveUsesExplicitPath(t *testing.T) {
	r, err := Resolve(Options{ExplicitPath: "/models/foo.gguf"})
	if err != nil || r.Path != "/models/foo.gguf" {
		t.Fatalf("r %+v err %v", r, err)
	}
}

func TestResolveReturnsCachedWithoutDownload(t *testing.T) {
	cache := t.TempDir()
	tier := SelectTier(64)
	dst := filepath.Join(cache, tier.File)
	if err := os.WriteFile(dst, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := Resolve(Options{CacheDir: cache, RAMGB: 64})
	if err != nil || r.Path != dst || r.Downloaded {
		t.Fatalf("r %+v err %v", r, err)
	}
}

func TestResolveDeclinedDownloadErrors(t *testing.T) {
	if _, err := Resolve(Options{CacheDir: t.TempDir(), RAMGB: 64, Confirm: func(Tier) bool { return false }}); err == nil {
		t.Fatal("expected error when download declined")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/capacity/ -run Resolve`
Expected: FAIL — `undefined: Resolve`, `undefined: Options`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/capacity/resolve.go`:
```go
package capacity

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// Options configures model resolution.
type Options struct {
	ExplicitPath string          // if set, used verbatim (no detection/download)
	CacheDir     string          // default ~/.cache/autoresearch/models
	RAMGB        float64         // if 0, detected via DetectRAMGB
	Confirm      func(Tier) bool // ask before downloading; default refuses
	Download     func(url, dst string) error
}

// Resolved is the chosen model.
type Resolved struct {
	Path       string
	Tier       Tier
	Downloaded bool
}

// Resolve picks a model path: explicit override, else cached tier model, else
// confirm-and-download.
func Resolve(o Options) (Resolved, error) {
	if o.ExplicitPath != "" {
		return Resolved{Path: o.ExplicitPath}, nil
	}
	cache := o.CacheDir
	if cache == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return Resolved{}, err
		}
		cache = filepath.Join(home, ".cache", "autoresearch", "models")
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return Resolved{}, err
	}
	ram := o.RAMGB
	if ram == 0 {
		r, err := DetectRAMGB()
		if err != nil {
			return Resolved{}, err
		}
		ram = r
	}
	tier := SelectTier(ram)
	dst := filepath.Join(cache, tier.File)
	if _, err := os.Stat(dst); err == nil {
		return Resolved{Path: dst, Tier: tier}, nil
	}
	confirm := o.Confirm
	if confirm == nil {
		confirm = func(Tier) bool { return false }
	}
	if !confirm(tier) {
		return Resolved{}, fmt.Errorf("model %s not present and download declined", tier.ID)
	}
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", tier.Repo, tier.File)
	download := o.Download
	if download == nil {
		download = httpDownload
	}
	if err := download(url, dst); err != nil {
		return Resolved{}, fmt.Errorf("download %s: %w", tier.ID, err)
	}
	return Resolved{Path: dst, Tier: tier, Downloaded: true}, nil
}

func httpDownload(url, dst string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/capacity/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/capacity/resolve.go internal/capacity/resolve_test.go
git commit -m "feat: model resolution with cache, confirm, download"
```

---

## Task 11: Runtime (managed llama-server)

**Files:**
- Create: `internal/runtime/runtime.go`
- Test: `internal/runtime/runtime_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/runtime/runtime_test.go`:
```go
package runtime

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestWaitHealthyFlips(t *testing.T) {
	var ready atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" && ready.Load() {
			w.WriteHeader(200)
			return
		}
		w.WriteHeader(503)
	}))
	defer srv.Close()
	go func() { time.Sleep(150 * time.Millisecond); ready.Store(true) }()
	if err := waitHealthy(context.Background(), srv.URL, srv.Client(), 3*time.Second); err != nil {
		t.Fatalf("waitHealthy: %v", err)
	}
}

func TestWaitHealthyTimesOut(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(503) }))
	defer srv.Close()
	if err := waitHealthy(context.Background(), srv.URL, srv.Client(), 300*time.Millisecond); err == nil {
		t.Fatal("expected timeout")
	}
}

func TestResolveBinaryUsesCache(t *testing.T) {
	cache := t.TempDir()
	dst := filepath.Join(cache, "llama-server")
	if err := os.WriteFile(dst, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := ResolveBinary(BinaryOptions{CacheDir: cache})
	if err != nil || got != dst {
		t.Fatalf("got %q err %v", got, err)
	}
}

func TestResolveBinaryDeclinedErrors(t *testing.T) {
	if _, err := ResolveBinary(BinaryOptions{CacheDir: t.TempDir(), Confirm: func(float64) bool { return false }}); err == nil {
		t.Fatal("expected error when download declined")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/runtime/`
Expected: FAIL — `undefined: waitHealthy`, `undefined: ResolveBinary`.

- [ ] **Step 3: Write minimal implementation**

Create `internal/runtime/runtime.go`:
```go
// Package runtime resolves, launches, health-checks, and shuts down a local
// llama-server process so the Go program owns the model runtime end-to-end.
package runtime

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ServerBuild pins the llama.cpp release used for the prebuilt macOS arm64 binary.
const ServerBuild = "b4823"

// BinaryOptions configures resolution of the llama-server binary.
type BinaryOptions struct {
	CacheDir string             // default ~/.cache/autoresearch/bin
	Confirm  func(sizeGB float64) bool
	Download func(url, dst string) error // returns the downloaded zip path content at dst
}

// ResolveBinary returns a path to a llama-server binary, downloading+unzipping the
// pinned llama.cpp macOS arm64 release on first use (after confirmation).
func ResolveBinary(o BinaryOptions) (string, error) {
	cache := o.CacheDir
	if cache == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		cache = filepath.Join(home, ".cache", "autoresearch", "bin")
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return "", err
	}
	bin := filepath.Join(cache, "llama-server")
	if _, err := os.Stat(bin); err == nil {
		return bin, nil
	}
	confirm := o.Confirm
	if confirm == nil {
		confirm = func(float64) bool { return false }
	}
	if !confirm(0.05) {
		return "", fmt.Errorf("llama-server not present and download declined")
	}
	url := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-macos-arm64.zip", ServerBuild, ServerBuild)
	zipPath := filepath.Join(cache, "llama.zip")
	download := o.Download
	if download == nil {
		download = httpDownload
	}
	if err := download(url, zipPath); err != nil {
		return "", fmt.Errorf("download llama-server: %w", err)
	}
	if err := unzipServer(zipPath, cache); err != nil {
		return "", err
	}
	if _, err := os.Stat(bin); err != nil {
		return "", fmt.Errorf("llama-server not found after unzip")
	}
	return bin, nil
}

// unzipServer extracts the archive and places a llama-server executable at dir/llama-server.
func unzipServer(zipPath, dir string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, f := range r.File {
		if !strings.HasSuffix(f.Name, "llama-server") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(filepath.Join(dir, "llama-server"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			rc.Close()
			return err
		}
		_, cErr := io.Copy(out, rc)
		rc.Close()
		out.Close()
		return cErr
	}
	return fmt.Errorf("llama-server entry not found in %s", zipPath)
}

func httpDownload(url, dst string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

// Options configures launching the server.
type Options struct {
	Binary       string
	ModelPath    string
	ContextLen   int
	StartTimeout time.Duration
}

// Server is a running, supervised llama-server child process.
type Server struct {
	cmd      *exec.Cmd
	endpoint string
}

// Start launches llama-server on a free port and waits until it is healthy.
func Start(ctx context.Context, o Options) (*Server, error) {
	port, err := freePort()
	if err != nil {
		return nil, err
	}
	ctxLen := o.ContextLen
	if ctxLen == 0 {
		ctxLen = 16384
	}
	endpoint := fmt.Sprintf("http://127.0.0.1:%d", port)
	cmd := exec.Command(o.Binary,
		"--model", o.ModelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"--n-gpu-layers", "999",
		"--ctx-size", fmt.Sprintf("%d", ctxLen),
	)
	cmd.Stdout = os.Stderr // surface server logs into the run log
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}
	timeout := o.StartTimeout
	if timeout == 0 {
		timeout = 2 * time.Minute
	}
	if err := waitHealthy(ctx, endpoint, &http.Client{Timeout: 2 * time.Second}, timeout); err != nil {
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("llama-server did not become healthy: %w", err)
	}
	return &Server{cmd: cmd, endpoint: endpoint}, nil
}

// Endpoint returns the base URL of the running server.
func (s *Server) Endpoint() string { return s.endpoint }

// Shutdown stops the server (SIGINT, then Kill on grace timeout).
func (s *Server) Shutdown() error {
	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}
	_ = s.cmd.Process.Signal(os.Interrupt)
	done := make(chan struct{})
	go func() { _ = s.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
	}
	return nil
}

func waitHealthy(ctx context.Context, endpoint string, client *http.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/health", nil)
		if resp, err := client.Do(req); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("not healthy within %s", timeout)
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}
```

> **Implementer note:** the actual `llama-server` flags and the release zip layout can
> drift across llama.cpp builds. After wiring `ar run`, do one real launch and confirm
> `--n-gpu-layers`, `--ctx-size`, `/health`, and the `llama-server` path inside the zip
> for the pinned `ServerBuild`; adjust the constant/flags if needed. The `external`
> backend is the fallback if a pinned build misbehaves.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/runtime/`
Expected: PASS (health-wait + binary-resolve unit tests; real launch is exercised via `ar run`).

- [ ] **Step 5: Commit**

```bash
git add internal/runtime/
git commit -m "feat: managed llama-server runtime (resolve, launch, health, shutdown)"
```

---

## Task 12: CLI dispatch + `ar report`

**Files:**
- Create: `cmd/ar/main.go`, `cmd/ar/report.go`

- [ ] **Step 1: Write the dispatch entrypoint**

Create `cmd/ar/main.go`:
```go
// Command ar runs the autoresearch optimization loop.
package main

import (
	"fmt"
	"os"

	"github.com/dobbo-ca/autoresearch/internal/version"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = cmdRun(os.Args[2:])
	case "init":
		err = cmdInit(os.Args[2:])
	case "report":
		err = cmdReport(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Println("ar", version.String())
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `ar — autoresearch optimization loop

Usage:
  ar init      scaffold a 3-file project (instructions/asset/scorer)
  ar run       run the overnight loop (downloads + launches the model)
  ar report    render report.md from rounds.jsonl
  ar version   print version`)
}
```

- [ ] **Step 2: Write `ar report`**

Create `cmd/ar/report.go`:
```go
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dobbo-ca/autoresearch/internal/config"
	"github.com/dobbo-ca/autoresearch/internal/ledger"
)

func cmdReport(args []string) error {
	fs := flag.NewFlagSet("report", flag.ContinueOnError)
	cfgPath := fs.String("config", "autoresearch.toml", "path to config")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	dir := filepath.Dir(*cfgPath)
	recs, err := ledger.Open(filepath.Join(dir, "rounds.jsonl")).All()
	if err != nil {
		return err
	}
	out := ledger.Render(recs, cfg.Project.Direction)
	if err := os.WriteFile(filepath.Join(dir, "report.md"), []byte(out), 0o644); err != nil {
		return err
	}
	fmt.Print(out)
	return nil
}
```

- [ ] **Step 3: Verify internal packages still vet**

Run: `go vet ./internal/...`
Expected: no errors. (`cmd/ar` does not link until Task 13 adds `cmdRun`/`cmdInit`.)

- [ ] **Step 4: Commit**

```bash
git add cmd/ar/main.go cmd/ar/report.go
git commit -m "feat: CLI dispatch and ar report"
```

---

## Task 13: `ar run` and `ar init`

**Files:**
- Create: `cmd/ar/run.go`, `cmd/ar/init.go`

- [ ] **Step 1: Write `ar run`**

Create `cmd/ar/run.go`:
```go
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/dobbo-ca/autoresearch/internal/brain"
	"github.com/dobbo-ca/autoresearch/internal/capacity"
	"github.com/dobbo-ca/autoresearch/internal/config"
	"github.com/dobbo-ca/autoresearch/internal/engine"
	"github.com/dobbo-ca/autoresearch/internal/ledger"
	"github.com/dobbo-ca/autoresearch/internal/runtime"
	"github.com/dobbo-ca/autoresearch/internal/scorer"
	"github.com/dobbo-ca/autoresearch/internal/workspace"
)

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	cfgPath := fs.String("config", "autoresearch.toml", "path to config")
	modelOverride := fs.String("model", "", "explicit GGUF path (overrides capacity auto-select)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	dir, err := filepath.Abs(filepath.Dir(*cfgPath))
	if err != nil {
		return err
	}
	instr, err := os.ReadFile(filepath.Join(dir, cfg.Project.Instructions))
	if err != nil {
		return fmt.Errorf("read instructions: %w", err)
	}

	w := workspace.New(dir, cfg.Project.Asset, []string{cfg.Project.Instructions, cfg.Project.Scorer})
	if err := w.EnsureRepo(); err != nil {
		return err
	}
	if err := w.RequireClean(); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	b, shutdown, err := startBrain(ctx, cfg, *modelOverride)
	if err != nil {
		return err
	}
	defer shutdown()
	defer b.Close()

	led := ledger.Open(filepath.Join(dir, "rounds.jsonl"))
	e := &engine.Engine{
		Brain: b, Workspace: w, Ledger: led, Instructions: string(instr),
		Direction: cfg.Project.Direction, Goal: cfg.Project.Goal, MaxRounds: cfg.Project.MaxRounds,
		History: cfg.Run.HistoryWindow, LogsDir: filepath.Join(dir, "logs"), Now: time.Now,
		Log: func(s string) { fmt.Println(s) },
		RunScorer: func(ctx context.Context) scorer.Result {
			return scorer.Run(ctx, cfg.Project.Scorer, dir, cfg.Timeout())
		},
	}
	if err := e.Run(ctx); err != nil {
		return err
	}
	recs, _ := led.All()
	_ = os.WriteFile(filepath.Join(dir, "report.md"), []byte(ledger.Render(recs, cfg.Project.Direction)), 0o644)
	fmt.Println("done; report.md written")
	return nil
}

// startBrain returns a Brain and a shutdown func. For the managed backend it downloads
// (with confirmation) and launches llama-server; for external it just points at the URL.
func startBrain(ctx context.Context, cfg config.Config, modelOverride string) (brain.Brain, func(), error) {
	if cfg.Backend() == "external" {
		return brain.NewSubprocess(cfg.Model.Endpoint, "local", cfg.Model.Temperature), func() {}, nil
	}
	res, err := capacity.Resolve(capacity.Options{
		ExplicitPath: pick(modelOverride, cfg.Model.Path),
		Confirm:      confirmModel,
	})
	if err != nil {
		return nil, nil, err
	}
	bin, err := runtime.ResolveBinary(runtime.BinaryOptions{Confirm: confirmBinary})
	if err != nil {
		return nil, nil, err
	}
	srv, err := runtime.Start(ctx, runtime.Options{Binary: bin, ModelPath: res.Path, ContextLen: cfg.Model.Context})
	if err != nil {
		return nil, nil, err
	}
	return brain.NewSubprocess(srv.Endpoint(), "local", cfg.Model.Temperature), func() { _ = srv.Shutdown() }, nil
}

func pick(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func confirmModel(t capacity.Tier) bool {
	return askYesNo(fmt.Sprintf("Download model %s (~%.1f GB) from %s?", t.ID, t.SizeGB, t.Repo))
}

func confirmBinary(sizeGB float64) bool {
	return askYesNo(fmt.Sprintf("Download llama-server (%s, ~%.0f MB) from llama.cpp releases?", runtime.ServerBuild, sizeGB*1000))
}

func askYesNo(prompt string) bool {
	fmt.Printf("%s [y/N]: ", prompt)
	line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	line = strings.ToLower(strings.TrimSpace(line))
	return line == "y" || line == "yes"
}
```

- [ ] **Step 2: Write `ar init`**

Create `cmd/ar/init.go`:
```go
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const instructionsStub = `# Goal

Describe in plain English what you are optimizing and why.

# Rules
- Run in short loops, overnight, until the goal is hit or I stop you.
- Change only the asset file(s). Never change the scorer or these instructions.
`

const scorerStub = `#!/bin/sh
# Print ONE number on the last line: the objective score for the current asset.
# Lower or higher is better per "direction" in autoresearch.toml.
echo 1.0
`

const gitignoreStub = "rounds.jsonl\nreport.md\nlogs/\n"

func cmdInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	dir := fs.String("dir", ".", "project directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	in := bufio.NewReader(os.Stdin)
	ask := func(q, def string) string {
		fmt.Printf("%s [%s]: ", q, def)
		line, _ := in.ReadString('\n')
		if line = strings.TrimSpace(line); line == "" {
			return def
		}
		return line
	}

	fmt.Println("Auto Research Engineer — project setup")
	fmt.Println("We pick ONE asset, turn \"is it good?\" into a single number, and optimize it overnight.")
	name := ask("Project name", "my-research")
	asset := ask("Asset file to optimize (the ONLY thing I may change)", "asset.txt")
	direction := ask("Is the score min (lower better) or max (higher better)?", "min")

	fmt.Println("\nFit check (all three required):")
	fmt.Println("  a) Is the score an objective number?")
	fmt.Println("  b) Does it return in minutes/hours, not weeks?")
	fmt.Println("  c) Can I actually change the asset file?")
	_ = ask("Press enter to acknowledge", "")

	files := map[string]string{
		filepath.Join(*dir, "instructions.md"): instructionsStub,
		filepath.Join(*dir, "score.sh"):        scorerStub,
		filepath.Join(*dir, ".gitignore"):      gitignoreStub,
		filepath.Join(*dir, asset):             "",
		filepath.Join(*dir, "autoresearch.toml"): fmt.Sprintf(`[project]
name = %q
instructions = "instructions.md"
asset = [%q]
scorer = "./score.sh"
direction = %q
max_rounds = 0
round_timeout = "10m"

[model]
backend = "managed"
context = 16384
temperature = 0.7

[run]
history_window = 8
`, name, asset, direction),
	}
	for path, body := range files {
		if _, err := os.Stat(path); err == nil {
			fmt.Printf("skip existing %s\n", path)
			continue
		}
		if err := os.WriteFile(path, []byte(body), modeFor(path)); err != nil {
			return err
		}
		fmt.Printf("wrote %s\n", path)
	}
	fmt.Println("\nNext: edit instructions.md and score.sh, then run `ar run`.")
	return nil
}

func modeFor(path string) os.FileMode {
	if strings.HasSuffix(path, ".sh") {
		return 0o755
	}
	return 0o644
}
```

- [ ] **Step 3: Build the binary**

Run: `go build -o bin/ar ./cmd/ar`
Expected: builds cleanly (no cgo).

- [ ] **Step 4: Vet and test everything**

Run:
```bash
go vet ./...
go test ./...
```
Expected: all packages PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/ar/run.go cmd/ar/init.go
git commit -m "feat: ar run (managed runtime) and ar init"
```

---

## Task 14: End-to-end loop with a server-shaped Brain

Proves the whole loop converges against the HTTP Brain using a fake OpenAI-compatible
server — no model or binary download required.

**Files:**
- Test: `cmd/ar/e2e_test.go`

- [ ] **Step 1: Write the end-to-end test**

Create `cmd/ar/e2e_test.go`:
```go
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/dobbo-ca/autoresearch/internal/brain"
	"github.com/dobbo-ca/autoresearch/internal/engine"
	"github.com/dobbo-ca/autoresearch/internal/ledger"
	"github.com/dobbo-ca/autoresearch/internal/scorer"
	"github.com/dobbo-ca/autoresearch/internal/workspace"
)

func TestEndToEndConverges(t *testing.T) {
	dir := t.TempDir()
	for _, a := range [][]string{{"init", "-q"}, {"config", "user.email", "t@t"}, {"config", "user.name", "t"}} {
		c := exec.Command("git", a...)
		c.Dir = dir
		_ = c.Run()
	}
	_ = os.WriteFile(filepath.Join(dir, "value.txt"), []byte("16"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "score.sh"), []byte("#!/bin/sh\nawk '{x=$1; if(x<0)x=-x; print x}' value.txt\n"), 0o755)
	_ = os.WriteFile(filepath.Join(dir, "instructions.md"), []byte("Drive value.txt to 0."), 0o644)

	// Fake server: parse current value out of the prompt, return half of it.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct{ Content string `json:"content"` } `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		cur := 16.0
		for _, m := range req.Messages {
			if v, ok := valueFromPrompt(m.Content); ok {
				cur = v
			}
		}
		content, _ := json.Marshal(brain.Proposal{Hypothesis: "halve", TargetFile: "value.txt", NewContent: strconv.FormatFloat(cur/2, 'f', -1, 64)})
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": string(content)}}},
		})
	}))
	defer srv.Close()

	w := workspace.New(dir, []string{"value.txt"}, []string{"score.sh", "instructions.md"})
	if err := w.EnsureRepo(); err != nil {
		t.Fatal(err)
	}
	_ = w.Commit("baseline")
	led := ledger.Open(filepath.Join(dir, "rounds.jsonl"))
	goal := 0.5
	e := &engine.Engine{
		Brain: brain.NewSubprocess(srv.URL, "fake", 0.0), Workspace: w, Ledger: led,
		Instructions: "Drive value.txt to 0.", Direction: "min", Goal: &goal, MaxRounds: 30,
		History: 8, LogsDir: filepath.Join(dir, "logs"), Now: time.Now,
		RunScorer: func(ctx context.Context) scorer.Result { return scorer.Run(ctx, "sh score.sh", dir, 5*time.Second) },
	}
	if err := e.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(filepath.Join(dir, "value.txt"))
	if v, _ := strconv.ParseFloat(string(got), 64); v > 0.5 {
		t.Fatalf("did not converge: %v", v)
	}
}

// valueFromPrompt extracts the value.txt content the prompt renders as "## value.txt\n```\n<v>\n```".
func valueFromPrompt(content string) (float64, bool) {
	const marker = "## value.txt\n```\n"
	i := strings.Index(content, marker)
	if i < 0 {
		return 0, false
	}
	rest := content[i+len(marker):]
	j := strings.Index(rest, "\n")
	if j < 0 {
		return 0, false
	}
	v, err := strconv.ParseFloat(rest[:j], 64)
	return v, err == nil
}
```

- [ ] **Step 2: Run the test**

Run: `go test ./cmd/ar/ -run EndToEnd -v`
Expected: PASS — the loop converges value.txt below 0.5.

- [ ] **Step 3: Commit**

```bash
git add cmd/ar/e2e_test.go
git commit -m "test: end-to-end loop convergence via HTTP brain"
```

---

## Task 15: README + full-build verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Verify the whole build and test suite**

Run:
```bash
go build -o bin/ar ./cmd/ar
go test ./...
```
Expected: builds without cgo; all tests PASS.

- [ ] **Step 2: Write the README**

Create `README.md`:
```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: project README and usage"
```

---

## Self-Review

**Spec coverage:**
- §4 components → Tasks 2–13 (config, scorer, capacity, brain, runtime, workspace, ledger, engine, CLI). ✓
- §5 three-file + scorer contract → config (T2), scorer parse + log capture (T3), workspace locks (T7). ✓
- §6 loop → engine (T9). ✓
- §7 runtime + brain → runtime (T11), HTTP brain (T6), ParseProposal `<think>` strip (T5). ✓
- §8 capacity auto-select + confirm-before-download (model + server binary) → T4/T10/T11, wired T13. ✓
- §9 git keep/revert + **.gitignore for ledger/report/logs** + locks → T7, used T9. ✓
- §10 config (managed/external) → T2, scaffolded T13. ✓
- §11 CLI init/run/report → T12/T13. ✓
- §12 ledger (incl. logs_path, diffstat) + report → T8, written by engine T9. ✓
- §13 robustness (timeout, NaN, **resume**, locks, clean stop, supervised shutdown) → T3/T7/T9/T11/T13. ✓
- §14 testing (fake brain, interleave-no-clobber, resume, parser units, capacity units, runtime health, e2e) → T3/T4/T7/T9/T11/T14. ✓
- §15 roadmap → README (T15). ✓

**Review fixes applied:**
- Ledger clobber + broken resume (confirmed major/blocker): `.gitignore` for `rounds.jsonl`/`report.md`/`logs/` written by `workspace.EnsureRepo` (T7) and `ar init` (T13); `RequireClean` uses `--untracked-files=no`; regression tests `TestEngineDoesNotClobberLedgerOnInterleave` + `TestEngineResumesFromLastKept` (T9) + `TestRequireCleanIgnoresLedger` (T7). ✓
- Scorer `TestParseScoreJSON` raw-string bug: now an interpreted string with a real newline (T3). ✓
- Ledger missing `logs_path`/`diffstat` + dropped stdout/stderr: added to `Record`; engine writes `logs/round-NNNN.log` and `Diffstat()` per round (T8/T9). ✓
- `prevBaseline` used at all three record sites; `ProcessState` nil-guarded (T3/T9). ✓
- All embedded-cgo blockers (double-accept, `-lggml-cpu`, KV reset, `GoStringN`): eliminated — embedded backend removed in favor of the managed `llama-server` subprocess. ✓

**Placeholder scan:** No "TBD/TODO/later". The two implementer notes (Qwen3.6 GGUF repo name in T4; `llama-server` flag/zip-layout verification in T11) are explicit one-line confirmations against live upstream artifacts, each with a working fallback (`model.path` / `external` backend), not deferred work.

**Type consistency:** `Proposal`/`ProposeInput`/`RoundSummary`, `ledger.Record` (with `LogsPath`/`Diffstat`), `scorer.Result`, `capacity.Tier`/`Options`/`Resolved`, `runtime.Options`/`BinaryOptions`/`Server`, and `engine.Engine` fields (incl. `LogsDir`) are referenced identically across tasks. `NewSubprocess`, `capacity.Resolve`, `runtime.ResolveBinary`/`Start`, and `engine.Engine{...}` call sites in T13/T14 match their definitions.
```

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

// Mirrors production wiring: EnsureRepo only (NO manual baseline commit), and the first
// round regresses. Before the fix this aborted with "git checkout -- .: pathspec ... did
// not match" because no HEAD existed. EnsureRepo must establish the baseline commit itself.
func TestEngineFirstRoundRegressionOnFreshRepo(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "value.txt"), []byte("4"), 0o644)
	scoreSh(t, dir)
	w := workspace.New(dir, []string{"value.txt"}, []string{"score.sh"})
	if err := w.EnsureRepo(); err != nil {
		t.Fatal(err)
	}
	// Intentionally NO w.Commit("baseline") here — EnsureRepo must establish HEAD.
	led := ledger.Open(filepath.Join(dir, "rounds.jsonl"))
	if err := newEngine(dir, w, led, doubleBrain{}, nil, 2).Run(context.Background()); err != nil {
		t.Fatalf("engine aborted on fresh-repo first-round regression: %v", err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "value.txt")); string(got) != "4" {
		t.Fatalf("regression not reverted on fresh repo: %q", got)
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

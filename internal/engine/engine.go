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
	Clock        func() time.Time // measures model-call latency; defaults to time.Now
	RunScorer    func(ctx context.Context) scorer.Result
	Log          func(string) // optional progress sink
}

func (e *Engine) clock() time.Time {
	if e.Clock != nil {
		return e.Clock()
	}
	return time.Now()
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
		modelStart := e.clock()
		prop, err := e.Brain.Propose(ctx, brain.ProposeInput{
			Instructions: e.Instructions,
			Asset:        asset,
			History:      e.recentHistory(recs),
			Direction:    e.Direction,
		})
		modelMS := e.clock().Sub(modelStart).Milliseconds()
		if err != nil {
			e.logf("round %d: propose failed: %v", round, err)
			recs = e.record(recs, ledger.Record{Round: round, TS: e.ts(), Hypothesis: "(propose failed)", Kept: false, ScoreBefore: prevBaseline, ScoreAfter: prevBaseline, ModelMS: modelMS})
			continue
		}
		if !e.Workspace.Allowed(prop.TargetFile) {
			e.logf("round %d: rejected target %q (locked/outside asset)", round, prop.TargetFile)
			recs = e.record(recs, ledger.Record{Round: round, TS: e.ts(), Hypothesis: prop.Hypothesis, TargetFile: prop.TargetFile, Kept: false, ScoreBefore: prevBaseline, ScoreAfter: prevBaseline, ModelMS: modelMS})
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
			ScorerExit: res.ExitCode, LogsPath: logsPath, Diffstat: diffstat, ModelMS: modelMS,
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

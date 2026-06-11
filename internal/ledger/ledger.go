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

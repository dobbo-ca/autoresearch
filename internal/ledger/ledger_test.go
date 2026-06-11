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

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

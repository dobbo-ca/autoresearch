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
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
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

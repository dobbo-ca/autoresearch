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

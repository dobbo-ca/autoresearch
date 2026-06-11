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

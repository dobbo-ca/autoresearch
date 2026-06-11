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

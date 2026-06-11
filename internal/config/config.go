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
	Backend     string  `toml:"backend"`  // "managed" (default) | "external"
	Path        string  `toml:"path"`     // explicit GGUF override
	Endpoint    string  `toml:"endpoint"` // required when backend == "external"
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

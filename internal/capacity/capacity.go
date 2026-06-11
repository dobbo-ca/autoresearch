// Package capacity detects machine memory and maps it to a default model tier.
package capacity

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// Tier maps an upper RAM bound to a default GGUF model.
type Tier struct {
	MaxGB  int     // inclusive upper bound; last tier is the catch-all
	ID     string  // stable model id
	Repo   string  // Hugging Face repo
	File   string  // GGUF filename within the repo
	SizeGB float64 // approximate download size, for the confirm prompt
}

// Tiers are evaluated in ascending MaxGB order. The final tier is the catch-all.
// Default for 32/64 GB machines is Qwen2.5-Coder-32B q4 (~18.5 GB, single-file GGUF
// at repo root — highest raw quality of the coder line). Repos/filenames here are
// live-verified against Hugging Face (resolve/main/<File> returns the file).
var Tiers = []Tier{
	{MaxGB: 16, ID: "qwen2.5-coder-7b", Repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", File: "qwen2.5-coder-7b-instruct-q4_k_m.gguf", SizeGB: 4.7},
	{MaxGB: 24, ID: "qwen2.5-coder-14b", Repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", File: "qwen2.5-coder-14b-instruct-q4_k_m.gguf", SizeGB: 9.0},
	{MaxGB: 1 << 30, ID: "qwen2.5-coder-32b", Repo: "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF", File: "qwen2.5-coder-32b-instruct-q4_k_m.gguf", SizeGB: 18.5},
}

// SelectTier returns the first tier whose MaxGB is >= ramGB.
func SelectTier(ramGB float64) Tier {
	for _, t := range Tiers {
		if ramGB <= float64(t.MaxGB) {
			return t
		}
	}
	return Tiers[len(Tiers)-1]
}

// DetectRAMGB returns total unified memory in GiB via `sysctl hw.memsize` (macOS).
func DetectRAMGB() (float64, error) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0, fmt.Errorf("sysctl hw.memsize: %w", err)
	}
	bytes, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse hw.memsize: %w", err)
	}
	return float64(bytes) / (1 << 30), nil
}

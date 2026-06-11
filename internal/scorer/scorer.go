// Package scorer runs the project's scoring command and extracts a single number.
package scorer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Result is the outcome of one scorer invocation.
type Result struct {
	Score    float64
	Stdout   string
	Stderr   string
	ExitCode int
	Err      error // non-nil means the round failed (revert and continue)
}

// Run executes command via `sh -c` in dir, with a wall-clock timeout, capturing
// stdout/stderr and parsing the score from stdout.
func Run(ctx context.Context, command, dir string, timeout time.Duration) Result {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	res := Result{Stdout: stdout.String(), Stderr: stderr.String()}
	if cmd.ProcessState != nil {
		res.ExitCode = cmd.ProcessState.ExitCode()
	}

	if ctx.Err() == context.DeadlineExceeded {
		res.Err = fmt.Errorf("scorer timed out after %s", timeout)
		return res
	}
	if runErr != nil {
		res.Err = fmt.Errorf("scorer exited non-zero: %w", runErr)
		return res
	}
	score, err := parseScore(res.Stdout)
	if err != nil {
		res.Err = err
		return res
	}
	res.Score = score
	return res
}

func parseScore(stdout string) (float64, error) {
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	last := ""
	for i := len(lines) - 1; i >= 0; i-- {
		if s := strings.TrimSpace(lines[i]); s != "" {
			last = s
			break
		}
	}
	if last == "" {
		return 0, fmt.Errorf("scorer produced no output")
	}
	var obj struct {
		Score *float64 `json:"score"`
	}
	if err := json.Unmarshal([]byte(last), &obj); err == nil && obj.Score != nil {
		return checkFinite(*obj.Score)
	}
	f, err := strconv.ParseFloat(last, 64)
	if err != nil {
		return 0, fmt.Errorf("could not parse score from %q", last)
	}
	return checkFinite(f)
}

func checkFinite(f float64) (float64, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, fmt.Errorf("score is not finite: %v", f)
	}
	return f, nil
}

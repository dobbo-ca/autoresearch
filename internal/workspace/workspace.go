// Package workspace manages the git-backed asset tree: applying the model's change,
// committing kept changes, reverting losers, enforcing writable files, and keeping the
// ledger/report/logs out of the tree (gitignored) so reverts and resume work.
package workspace

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// IgnoredPaths are written to .gitignore so the engine's working files are never tracked
// (a kept round's `git add -A` must not track them; a reverted round's `git checkout`
// must not delete them; and they must not trip the clean-tree check on resume).
var IgnoredPaths = []string{"rounds.jsonl", "report.md", "logs/"}

type Workspace struct {
	dir    string
	asset  []string // globs the model may write
	locked []string // files the model must never write
}

// New builds a Workspace over dir. asset and locked are paths/globs relative to dir.
func New(dir string, asset, locked []string) *Workspace {
	return &Workspace{dir: dir, asset: asset, locked: locked}
}

func (w *Workspace) git(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = w.dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, out)
	}
	return string(out), nil
}

// EnsureRepo runs `git init` if needed, guarantees .gitignore lists IgnoredPaths,
// and guarantees at least one commit exists (a HEAD) so the first reverted round
// has something to `git checkout`.
func (w *Workspace) EnsureRepo() error {
	if _, err := os.Stat(filepath.Join(w.dir, ".git")); err != nil {
		if _, err := w.git("init", "-q"); err != nil {
			return err
		}
	}
	if err := w.ensureGitignore(); err != nil {
		return err
	}
	// If there is no HEAD yet, create a baseline commit of the current tree so that
	// RevertAsset (git checkout) works from round 1 on a freshly-initialized project.
	if _, err := w.git("rev-parse", "--verify", "-q", "HEAD"); err != nil {
		w.ensureIdentity()
		if _, err := w.git("add", "-A"); err != nil {
			return err
		}
		if _, err := w.git("commit", "-q", "-m", "autoresearch baseline"); err != nil {
			return err
		}
	}
	return nil
}

// ensureIdentity sets a fallback git identity only when none is configured, so commits
// work on a machine without a global identity without clobbering an existing one.
func (w *Workspace) ensureIdentity() {
	if out, _ := w.git("config", "user.email"); strings.TrimSpace(out) == "" {
		_, _ = w.git("config", "user.email", "autoresearch@localhost")
		_, _ = w.git("config", "user.name", "autoresearch")
	}
}

func (w *Workspace) ensureGitignore() error {
	p := filepath.Join(w.dir, ".gitignore")
	existing, _ := os.ReadFile(p)
	lines := string(existing)
	var add []string
	for _, ig := range IgnoredPaths {
		if !lineContains(lines, ig) {
			add = append(add, ig)
		}
	}
	if len(add) == 0 {
		return nil
	}
	if len(lines) > 0 && !strings.HasSuffix(lines, "\n") {
		lines += "\n"
	}
	lines += strings.Join(add, "\n") + "\n"
	return os.WriteFile(p, []byte(lines), 0o644)
}

func lineContains(body, want string) bool {
	for _, l := range strings.Split(body, "\n") {
		if strings.TrimSpace(l) == want {
			return true
		}
	}
	return false
}

// RequireClean errors if any TRACKED file has uncommitted changes. Untracked and
// gitignored files (the ledger/report/logs) do not count.
func (w *Workspace) RequireClean() error {
	out, err := w.git("status", "--porcelain", "--untracked-files=no")
	if err != nil {
		return err
	}
	if strings.TrimSpace(out) != "" {
		return fmt.Errorf("workspace has uncommitted changes; commit or stash before running")
	}
	return nil
}

// Allowed reports whether the model may write rel.
func (w *Workspace) Allowed(rel string) bool {
	clean := filepath.Clean(rel)
	if strings.HasPrefix(clean, "..") || filepath.IsAbs(clean) {
		return false
	}
	for _, l := range w.locked {
		if clean == filepath.Clean(l) {
			return false
		}
	}
	for _, g := range w.asset {
		if ok, _ := filepath.Match(filepath.Clean(g), clean); ok {
			return true
		}
	}
	return false
}

// ReadAsset returns rel-path -> content for every file matching an asset glob.
func (w *Workspace) ReadAsset() (map[string]string, error) {
	out := map[string]string{}
	for _, g := range w.asset {
		matches, err := filepath.Glob(filepath.Join(w.dir, g))
		if err != nil {
			return nil, err
		}
		for _, m := range matches {
			b, err := os.ReadFile(m)
			if err != nil {
				return nil, err
			}
			rel, _ := filepath.Rel(w.dir, m)
			out[rel] = string(b)
		}
	}
	return out, nil
}

// Apply writes content to rel (caller must have checked Allowed).
func (w *Workspace) Apply(rel, content string) error {
	return os.WriteFile(filepath.Join(w.dir, rel), []byte(content), 0o644)
}

// Diffstat returns `git diff --stat` for the current uncommitted change (the round's edit).
func (w *Workspace) Diffstat() string {
	out, _ := w.git("diff", "--stat")
	return strings.TrimSpace(out)
}

// Commit stages everything (respecting .gitignore) and commits with msg. It is a no-op
// when there is nothing staged to commit (e.g. a redundant baseline commit).
func (w *Workspace) Commit(msg string) error {
	if _, err := w.git("add", "-A"); err != nil {
		return err
	}
	if _, err := w.git("diff", "--cached", "--quiet"); err == nil {
		return nil // nothing staged
	}
	_, err := w.git("commit", "-q", "-m", msg)
	return err
}

// RevertAsset restores tracked files to the last commit.
func (w *Workspace) RevertAsset() error {
	_, err := w.git("checkout", "--", ".")
	return err
}

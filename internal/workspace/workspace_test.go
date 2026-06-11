package workspace

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func newRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"}, {"config", "user.email", "t@example.com"}, {"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %s", args, out)
		}
	}
	return dir
}

func TestAllowedRejectsLockedAndOutside(t *testing.T) {
	w := New(newRepo(t), []string{"value.txt"}, []string{"instructions.md", "score.sh"})
	if !w.Allowed("value.txt") {
		t.Error("value.txt should be allowed")
	}
	if w.Allowed("instructions.md") || w.Allowed("score.sh") {
		t.Error("locked files must be rejected")
	}
	if w.Allowed("../escape.txt") {
		t.Error("path traversal must be rejected")
	}
}

func TestEnsureRepoWritesGitignore(t *testing.T) {
	dir := t.TempDir()
	w := New(dir, []string{"value.txt"}, nil)
	if err := w.EnsureRepo(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, ".gitignore"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"rounds.jsonl", "report.md", "logs/"} {
		if !contains(string(b), want) {
			t.Errorf(".gitignore missing %q:\n%s", want, b)
		}
	}
}

// Regression: an untracked, gitignored ledger must NOT make RequireClean fail (resume).
func TestRequireCleanIgnoresLedger(t *testing.T) {
	dir := t.TempDir()
	w := New(dir, []string{"value.txt"}, nil)
	if err := w.EnsureRepo(); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(dir, "value.txt"), []byte("1"), 0o644)
	if err := w.Commit("baseline"); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(dir, "rounds.jsonl"), []byte("{}\n"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "report.md"), []byte("# r\n"), 0o644)
	if err := w.RequireClean(); err != nil {
		t.Fatalf("RequireClean must ignore gitignored files: %v", err)
	}
}

func TestApplyCommitRevert(t *testing.T) {
	dir := newRepo(t)
	_ = os.WriteFile(filepath.Join(dir, "value.txt"), []byte("10"), 0o644)
	w := New(dir, []string{"value.txt"}, nil)
	if err := w.Commit("baseline"); err != nil {
		t.Fatal(err)
	}
	if err := w.Apply("value.txt", "5"); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "value.txt")); string(got) != "5" {
		t.Fatalf("apply failed: %q", got)
	}
	if err := w.RevertAsset(); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "value.txt")); string(got) != "10" {
		t.Fatalf("revert failed: %q", got)
	}
}

func TestReadAssetResolvesGlobs(t *testing.T) {
	dir := newRepo(t)
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("A"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "b.txt"), []byte("B"), 0o644)
	w := New(dir, []string{"*.txt"}, nil)
	m, err := w.ReadAsset()
	if err != nil {
		t.Fatal(err)
	}
	if m["a.txt"] != "A" || m["b.txt"] != "B" {
		t.Fatalf("globs not resolved: %+v", m)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

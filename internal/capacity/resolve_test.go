package capacity

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveUsesExplicitPath(t *testing.T) {
	r, err := Resolve(Options{ExplicitPath: "/models/foo.gguf"})
	if err != nil || r.Path != "/models/foo.gguf" {
		t.Fatalf("r %+v err %v", r, err)
	}
}

func TestResolveReturnsCachedWithoutDownload(t *testing.T) {
	cache := t.TempDir()
	tier := SelectTier(64)
	dst := filepath.Join(cache, tier.File)
	if err := os.WriteFile(dst, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := Resolve(Options{CacheDir: cache, RAMGB: 64})
	if err != nil || r.Path != dst || r.Downloaded {
		t.Fatalf("r %+v err %v", r, err)
	}
}

func TestResolveDeclinedDownloadErrors(t *testing.T) {
	if _, err := Resolve(Options{CacheDir: t.TempDir(), RAMGB: 64, Confirm: func(Tier) bool { return false }}); err == nil {
		t.Fatal("expected error when download declined")
	}
}

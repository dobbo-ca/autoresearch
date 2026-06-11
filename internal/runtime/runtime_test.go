package runtime

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestWaitHealthyFlips(t *testing.T) {
	var ready atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" && ready.Load() {
			w.WriteHeader(200)
			return
		}
		w.WriteHeader(503)
	}))
	defer srv.Close()
	go func() { time.Sleep(150 * time.Millisecond); ready.Store(true) }()
	if err := waitHealthy(context.Background(), srv.URL, srv.Client(), 3*time.Second); err != nil {
		t.Fatalf("waitHealthy: %v", err)
	}
}

func TestWaitHealthyTimesOut(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(503) }))
	defer srv.Close()
	if err := waitHealthy(context.Background(), srv.URL, srv.Client(), 300*time.Millisecond); err == nil {
		t.Fatal("expected timeout")
	}
}

func TestResolveBinaryUsesCache(t *testing.T) {
	cache := t.TempDir()
	install := filepath.Join(cache, ServerBuild)
	if err := os.MkdirAll(install, 0o755); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(install, "llama-server")
	if err := os.WriteFile(dst, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := ResolveBinary(BinaryOptions{CacheDir: cache})
	if err != nil || got != dst {
		t.Fatalf("got %q err %v", got, err)
	}
}

func TestResolveBinaryDeclinedErrors(t *testing.T) {
	if _, err := ResolveBinary(BinaryOptions{CacheDir: t.TempDir(), Confirm: func(float64) bool { return false }}); err == nil {
		t.Fatal("expected error when download declined")
	}
}

// TestResolveBinaryExtractsWholeArchive verifies the whole release archive is
// extracted (not just the binary), so llama-server keeps its sibling dylibs, and
// that the nested binary is located after extraction.
func TestResolveBinaryExtractsWholeArchive(t *testing.T) {
	cache := t.TempDir()
	// A fake release tar.gz mirroring the real layout: top dir with the binary
	// plus a sibling dylib it would dlopen via @loader_path.
	src := filepath.Join(cache, "fake.tar.gz")
	files := map[string]string{
		"llama-" + ServerBuild + "/llama-server":  "#!/bin/sh\n",
		"llama-" + ServerBuild + "/libllama.dylib": "stub",
	}
	writeTarGz(t, src, files)

	got, err := ResolveBinary(BinaryOptions{
		CacheDir: cache,
		Confirm:  func(float64) bool { return true },
		Download: func(_, dst string) error { return copyFile(src, dst) },
	})
	if err != nil {
		t.Fatalf("ResolveBinary: %v", err)
	}
	if filepath.Base(got) != "llama-server" {
		t.Fatalf("expected llama-server, got %q", got)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(got), "libllama.dylib")); err != nil {
		t.Fatalf("sibling dylib not extracted next to binary: %v", err)
	}
}

func writeTarGz(t *testing.T, path string, files map[string]string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for name, content := range files {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(content)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
}

func copyFile(src, dst string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o644)
}

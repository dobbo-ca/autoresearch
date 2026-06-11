package runtime

import (
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
	dst := filepath.Join(cache, "llama-server")
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

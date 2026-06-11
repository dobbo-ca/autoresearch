// Package runtime resolves, launches, health-checks, and shuts down a local
// llama-server process so the Go program owns the model runtime end-to-end.
package runtime

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ServerBuild pins the llama.cpp release used for the prebuilt macOS arm64 binary.
const ServerBuild = "b4823"

// BinaryOptions configures resolution of the llama-server binary.
type BinaryOptions struct {
	CacheDir string             // default ~/.cache/autoresearch/bin
	Confirm  func(sizeGB float64) bool
	Download func(url, dst string) error // returns the downloaded zip path content at dst
}

// ResolveBinary returns a path to a llama-server binary, downloading+unzipping the
// pinned llama.cpp macOS arm64 release on first use (after confirmation).
func ResolveBinary(o BinaryOptions) (string, error) {
	cache := o.CacheDir
	if cache == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		cache = filepath.Join(home, ".cache", "autoresearch", "bin")
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return "", err
	}
	bin := filepath.Join(cache, "llama-server")
	if _, err := os.Stat(bin); err == nil {
		return bin, nil
	}
	confirm := o.Confirm
	if confirm == nil {
		confirm = func(float64) bool { return false }
	}
	if !confirm(0.05) {
		return "", fmt.Errorf("llama-server not present and download declined")
	}
	url := fmt.Sprintf("https://github.com/ggml-org/llama.cpp/releases/download/%s/llama-%s-bin-macos-arm64.zip", ServerBuild, ServerBuild)
	zipPath := filepath.Join(cache, "llama.zip")
	download := o.Download
	if download == nil {
		download = httpDownload
	}
	if err := download(url, zipPath); err != nil {
		return "", fmt.Errorf("download llama-server: %w", err)
	}
	if err := unzipServer(zipPath, cache); err != nil {
		return "", err
	}
	if _, err := os.Stat(bin); err != nil {
		return "", fmt.Errorf("llama-server not found after unzip")
	}
	return bin, nil
}

// unzipServer extracts the archive and places a llama-server executable at dir/llama-server.
func unzipServer(zipPath, dir string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, f := range r.File {
		if !strings.HasSuffix(f.Name, "llama-server") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(filepath.Join(dir, "llama-server"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			rc.Close()
			return err
		}
		_, cErr := io.Copy(out, rc)
		rc.Close()
		out.Close()
		return cErr
	}
	return fmt.Errorf("llama-server entry not found in %s", zipPath)
}

func httpDownload(url, dst string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

// Options configures launching the server.
type Options struct {
	Binary       string
	ModelPath    string
	ContextLen   int
	StartTimeout time.Duration
}

// Server is a running, supervised llama-server child process.
type Server struct {
	cmd      *exec.Cmd
	endpoint string
}

// Start launches llama-server on a free port and waits until it is healthy.
func Start(ctx context.Context, o Options) (*Server, error) {
	port, err := freePort()
	if err != nil {
		return nil, err
	}
	ctxLen := o.ContextLen
	if ctxLen == 0 {
		ctxLen = 16384
	}
	endpoint := fmt.Sprintf("http://127.0.0.1:%d", port)
	cmd := exec.Command(o.Binary,
		"--model", o.ModelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"--n-gpu-layers", "999",
		"--ctx-size", fmt.Sprintf("%d", ctxLen),
	)
	cmd.Stdout = os.Stderr // surface server logs into the run log
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}
	timeout := o.StartTimeout
	if timeout == 0 {
		timeout = 2 * time.Minute
	}
	if err := waitHealthy(ctx, endpoint, &http.Client{Timeout: 2 * time.Second}, timeout); err != nil {
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("llama-server did not become healthy: %w", err)
	}
	return &Server{cmd: cmd, endpoint: endpoint}, nil
}

// Endpoint returns the base URL of the running server.
func (s *Server) Endpoint() string { return s.endpoint }

// Shutdown stops the server (SIGINT, then Kill on grace timeout).
func (s *Server) Shutdown() error {
	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}
	_ = s.cmd.Process.Signal(os.Interrupt)
	done := make(chan struct{})
	go func() { _ = s.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
	}
	return nil
}

func waitHealthy(ctx context.Context, endpoint string, client *http.Client, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/health", nil)
		if resp, err := client.Do(req); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("not healthy within %s", timeout)
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

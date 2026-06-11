package capacity

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// Options configures model resolution.
type Options struct {
	ExplicitPath string          // if set, used verbatim (no detection/download)
	CacheDir     string          // default ~/.cache/autoresearch/models
	RAMGB        float64         // if 0, detected via DetectRAMGB
	Confirm      func(Tier) bool // ask before downloading; default refuses
	Download     func(url, dst string) error
}

// Resolved is the chosen model.
type Resolved struct {
	Path       string
	Tier       Tier
	Downloaded bool
}

// Resolve picks a model path: explicit override, else cached tier model, else
// confirm-and-download.
func Resolve(o Options) (Resolved, error) {
	if o.ExplicitPath != "" {
		return Resolved{Path: o.ExplicitPath}, nil
	}
	cache := o.CacheDir
	if cache == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return Resolved{}, err
		}
		cache = filepath.Join(home, ".cache", "autoresearch", "models")
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return Resolved{}, err
	}
	ram := o.RAMGB
	if ram == 0 {
		r, err := DetectRAMGB()
		if err != nil {
			return Resolved{}, err
		}
		ram = r
	}
	tier := SelectTier(ram)
	dst := filepath.Join(cache, tier.File)
	if _, err := os.Stat(dst); err == nil {
		return Resolved{Path: dst, Tier: tier}, nil
	}
	confirm := o.Confirm
	if confirm == nil {
		confirm = func(Tier) bool { return false }
	}
	if !confirm(tier) {
		return Resolved{}, fmt.Errorf("model %s not present and download declined", tier.ID)
	}
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", tier.Repo, tier.File)
	download := o.Download
	if download == nil {
		download = httpDownload
	}
	if err := download(url, dst); err != nil {
		return Resolved{}, fmt.Errorf("download %s: %w", tier.ID, err)
	}
	return Resolved{Path: dst, Tier: tier, Downloaded: true}, nil
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
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

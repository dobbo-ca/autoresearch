// Package version reports the build version of ar.
package version

// These are set at build time via -ldflags "-X .../internal/version.Version=...".
// The defaults apply to local/dev builds.
var (
	Version = "0.1.0-dev"
	Commit  = "none"
	Date    = "unknown"
)

// String returns the current ar version.
func String() string { return Version }

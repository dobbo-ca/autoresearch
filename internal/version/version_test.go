package version

import "testing"

func TestString(t *testing.T) {
	if String() != "0.1.0-dev" {
		t.Fatalf("got %q, want %q", String(), "0.1.0-dev")
	}
}

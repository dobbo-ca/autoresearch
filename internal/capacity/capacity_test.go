package capacity

import "testing"

func TestSelectTier(t *testing.T) {
	cases := []struct {
		gb   float64
		want string
	}{
		{8, "qwen2.5-coder-7b"},
		{16, "qwen2.5-coder-7b"},
		{24, "qwen2.5-coder-14b"},
		{32, "qwen2.5-coder-32b"},
		{64, "qwen2.5-coder-32b"},
		{128, "qwen2.5-coder-32b"},
	}
	for _, c := range cases {
		if got := SelectTier(c.gb).ID; got != c.want {
			t.Errorf("SelectTier(%v) = %q, want %q", c.gb, got, c.want)
		}
	}
}

func TestTiersSortedAscending(t *testing.T) {
	for i := 1; i < len(Tiers); i++ {
		if Tiers[i].MaxGB <= Tiers[i-1].MaxGB {
			t.Fatalf("tiers not strictly ascending at %d", i)
		}
	}
}

package brain

import (
	"strings"
	"testing"
)

func TestBuildMessagesIncludesAssetAndHistory(t *testing.T) {
	in := ProposeInput{
		Instructions: "Make value.txt approach 0.",
		Asset:        map[string]string{"value.txt": "10"},
		Direction:    "min",
		History: []RoundSummary{
			{Round: 1, Hypothesis: "try 8", TargetFile: "value.txt", Before: 10, After: 8, Kept: true},
		},
	}
	sys, user := BuildMessages(in)
	if !strings.Contains(sys, "lower is better") {
		t.Errorf("system prompt missing direction guidance:\n%s", sys)
	}
	if !strings.Contains(user, "value.txt") || !strings.Contains(user, "Make value.txt approach 0.") {
		t.Errorf("user prompt missing asset/instructions:\n%s", user)
	}
	if !strings.Contains(user, "try 8") {
		t.Errorf("user prompt missing history:\n%s", user)
	}
}

// assetIndex returns the byte offset of an asset file's "## <path>" header in the
// rendered user prompt, or -1 if absent.
func assetIndex(user, path string) int {
	return strings.Index(user, "## "+path+"\n")
}

// commonPrefixLen returns the length of the longest shared leading substring.
func commonPrefixLen(a, b string) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// TestBuildMessagesPlacesChangedFileLast covers acceptance (b): the asset file
// changed in the previous (kept) round is rendered last among the asset files,
// so the stable prefix (system + instructions + unchanged assets) leads.
func TestBuildMessagesPlacesChangedFileLast(t *testing.T) {
	in := ProposeInput{
		Instructions: "optimize",
		Asset:        map[string]string{"a.txt": "A", "b.txt": "B", "c.txt": "C"},
		Direction:    "min",
		History: []RoundSummary{
			{Round: 1, Hypothesis: "tweak b", TargetFile: "b.txt", Kept: true},
		},
	}
	_, user := BuildMessages(in)
	ia, ib, ic := assetIndex(user, "a.txt"), assetIndex(user, "b.txt"), assetIndex(user, "c.txt")
	if ia < 0 || ib < 0 || ic < 0 {
		t.Fatalf("missing asset header(s): a=%d b=%d c=%d\n%s", ia, ib, ic, user)
	}
	if !(ib > ia && ib > ic) {
		t.Errorf("changed file b.txt not last among assets: a=%d b=%d c=%d", ia, ib, ic)
	}
}

// TestBuildMessagesStablePrefixWhenOneFileChanges covers acceptance (a): two
// consecutive inputs differing only in the changed asset file share a long
// identical prefix up to that file's content.
func TestBuildMessagesStablePrefixWhenOneFileChanges(t *testing.T) {
	hist := []RoundSummary{{Round: 1, Hypothesis: "tweak b", TargetFile: "b.txt", Kept: true}}
	in1 := ProposeInput{
		Instructions: "optimize",
		Asset:        map[string]string{"a.txt": "STABLE", "b.txt": "OLD"},
		Direction:    "min",
		History:      hist,
	}
	in2 := ProposeInput{
		Instructions: "optimize",
		Asset:        map[string]string{"a.txt": "STABLE", "b.txt": "NEW"},
		Direction:    "min",
		History:      hist,
	}
	_, u1 := BuildMessages(in1)
	_, u2 := BuildMessages(in2)
	cpl := commonPrefixLen(u1, u2)
	prefix := u1[:cpl]
	// The shared prefix must include the unchanged asset and the changed file's
	// header — i.e. everything is identical up to b.txt's body.
	if !strings.Contains(prefix, "## a.txt\n") || !strings.Contains(prefix, "STABLE") {
		t.Errorf("shared prefix missing unchanged asset a.txt:\n%q", prefix)
	}
	if !strings.Contains(prefix, "## b.txt\n") {
		t.Errorf("shared prefix should extend through b.txt header (changed file last):\n%q", prefix)
	}
	// Divergence is the changed content itself.
	if strings.Contains(prefix, "OLD") || strings.Contains(prefix, "NEW") {
		t.Errorf("shared prefix should stop at changed content, got:\n%q", prefix)
	}
}

// TestBuildMessagesOrdersByKeptRecency guards the cross-target case: when several
// files were kept in different rounds, they cluster toward the end in order of how
// recently each was kept (least-recent first, most-recent last). Plain "move the one
// last-kept file to the end" would reorder the others alphabetically and break the
// prefix earlier when the kept target changes between rounds.
func TestBuildMessagesOrdersByKeptRecency(t *testing.T) {
	in := ProposeInput{
		Instructions: "optimize",
		Asset:        map[string]string{"a.txt": "A", "b.txt": "B", "c.txt": "C"},
		Direction:    "min",
		History: []RoundSummary{
			{Round: 1, Hypothesis: "kept a", TargetFile: "a.txt", Kept: true},
			{Round: 2, Hypothesis: "kept b", TargetFile: "b.txt", Kept: true},
		},
	}
	_, user := BuildMessages(in)
	ia, ib, ic := assetIndex(user, "a.txt"), assetIndex(user, "b.txt"), assetIndex(user, "c.txt")
	// Never-kept c first, then a (kept round 1), then b (kept round 2) last.
	if !(ic < ia && ia < ib) {
		t.Errorf("assets not ordered by kept-recency (want c<a<b): a=%d b=%d c=%d", ia, ib, ic)
	}
}

// TestBuildMessagesRevertedRoundDoesNotReorder guards the common revert case:
// ordering keys off the last KEPT target, not the most recent (possibly
// reverted) target, so a reverted round leaves the asset block byte-stable.
func TestBuildMessagesRevertedRoundDoesNotReorder(t *testing.T) {
	in := ProposeInput{
		Instructions: "optimize",
		Asset:        map[string]string{"a.txt": "A", "b.txt": "B", "c.txt": "C"},
		Direction:    "min",
		History: []RoundSummary{
			{Round: 1, Hypothesis: "tweak b", TargetFile: "b.txt", Kept: true},
			{Round: 2, Hypothesis: "try a", TargetFile: "a.txt", Kept: false},
		},
	}
	_, user := BuildMessages(in)
	ia, ib, ic := assetIndex(user, "a.txt"), assetIndex(user, "b.txt"), assetIndex(user, "c.txt")
	// Last KEPT target is b.txt; the reverted a.txt must NOT be forced last.
	if !(ib > ia && ib > ic) {
		t.Errorf("last-kept b.txt should be last; reverted a.txt must not reorder: a=%d b=%d c=%d", ia, ib, ic)
	}
}

func TestGrammarMentionsAllFields(t *testing.T) {
	g := Grammar()
	for _, f := range []string{"hypothesis", "target_file", "new_content"} {
		if !strings.Contains(g, f) {
			t.Errorf("grammar missing field %q", f)
		}
	}
}

func TestParseProposalPlainJSON(t *testing.T) {
	p, err := ParseProposal(`{"hypothesis":"h","target_file":"a.txt","new_content":"x"}`)
	if err != nil || p.TargetFile != "a.txt" || p.NewContent != "x" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

func TestParseProposalStripsThinkBlock(t *testing.T) {
	in := "<think>let me reason {not json}</think>\n{\"hypothesis\":\"h\",\"target_file\":\"a\",\"new_content\":\"b\"}"
	p, err := ParseProposal(in)
	if err != nil || p.TargetFile != "a" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

func TestParseProposalExtractsEmbeddedJSON(t *testing.T) {
	p, err := ParseProposal("Sure!\n{\"hypothesis\":\"h\",\"target_file\":\"a\",\"new_content\":\"b\"}\nDone.")
	if err != nil || p.TargetFile != "a" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

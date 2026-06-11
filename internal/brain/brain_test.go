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

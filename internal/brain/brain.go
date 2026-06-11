// Package brain defines the model interface that proposes one change per round,
// plus the shared types, prompt builder, JSON grammar, and proposal parser.
package brain

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// RoundSummary is a compact record of a past round, shown to the model as history.
type RoundSummary struct {
	Round      int
	Hypothesis string
	TargetFile string
	Before     float64
	After      float64
	Kept       bool
}

// ProposeInput is everything the model sees for one round.
type ProposeInput struct {
	Instructions string
	Asset        map[string]string // path -> current content
	History      []RoundSummary
	Direction    string // "min" | "max"
}

// Proposal is the model's single change for one round.
type Proposal struct {
	Hypothesis string `json:"hypothesis"`
	TargetFile string `json:"target_file"`
	NewContent string `json:"new_content"`
}

// Brain proposes one change per round.
type Brain interface {
	Propose(ctx context.Context, in ProposeInput) (Proposal, error)
	Close() error
}

// BuildMessages renders the system and user prompts for one round.
func BuildMessages(in ProposeInput) (system, user string) {
	goal := "lower is better"
	if in.Direction == "max" {
		goal = "higher is better"
	}
	system = fmt.Sprintf(`You are an optimization engineer running an overnight research loop.
Each round you propose exactly ONE change to ONE asset file to improve a single objective score (%s).
Rules:
- Change only one file, chosen from the asset files shown.
- Return the FULL new content of that file, not a diff.
- Make one focused, testable hypothesis per round; do not repeat changes that were already reverted.
- Reply ONLY with a JSON object: {"hypothesis": string, "target_file": string, "new_content": string}.`, goal)

	var b strings.Builder
	b.WriteString("# Instructions (locked, human-authored)\n")
	b.WriteString(in.Instructions)
	b.WriteString("\n\n# Current asset files\n")
	paths := make([]string, 0, len(in.Asset))
	for p := range in.Asset {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, p := range paths {
		fmt.Fprintf(&b, "\n## %s\n```\n%s\n```\n", p, in.Asset[p])
	}
	if len(in.History) > 0 {
		b.WriteString("\n# Recent rounds (do not repeat reverted ideas)\n")
		for _, h := range in.History {
			status := "reverted"
			if h.Kept {
				status = "kept"
			}
			fmt.Fprintf(&b, "- round %d [%s] %s (%s): %.6f -> %.6f\n",
				h.Round, status, h.Hypothesis, h.TargetFile, h.Before, h.After)
		}
	}
	b.WriteString("\nPropose the next single change now.")
	return system, b.String()
}

// Grammar returns a GBNF grammar that constrains output to the Proposal JSON shape.
func Grammar() string {
	return `root   ::= "{" ws "\"hypothesis\"" ws ":" ws string ws "," ws "\"target_file\"" ws ":" ws string ws "," ws "\"new_content\"" ws ":" ws string ws "}"
string ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )* "\""
ws     ::= [ \t\n]*`
}

var thinkRE = regexp.MustCompile(`(?s)<think>.*?</think>`)

// ParseProposal extracts a Proposal from model text. It strips any <think>...</think>
// reasoning block (e.g. Qwen3.6) and tolerates surrounding prose by slicing to the
// outermost JSON object.
func ParseProposal(content string) (Proposal, error) {
	content = thinkRE.ReplaceAllString(content, "")
	var p Proposal
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &p); err == nil && p.TargetFile != "" {
		return p, nil
	}
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start >= 0 && end > start {
		if err := json.Unmarshal([]byte(content[start:end+1]), &p); err == nil && p.TargetFile != "" {
			return p, nil
		}
	}
	return Proposal{}, fmt.Errorf("could not parse proposal JSON from model output")
}

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const instructionsStub = `# Goal

Describe in plain English what you are optimizing and why.

# Rules
- Run in short loops, overnight, until the goal is hit or I stop you.
- Change only the asset file(s). Never change the scorer or these instructions.
`

const scorerStub = `#!/bin/sh
# Print ONE number on the last line: the objective score for the current asset.
# Lower or higher is better per "direction" in autoresearch.toml.
echo 1.0
`

const gitignoreStub = "rounds.jsonl\nreport.md\nlogs/\n"

func cmdInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	dir := fs.String("dir", ".", "project directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	in := bufio.NewReader(os.Stdin)
	ask := func(q, def string) string {
		fmt.Printf("%s [%s]: ", q, def)
		line, _ := in.ReadString('\n')
		if line = strings.TrimSpace(line); line == "" {
			return def
		}
		return line
	}

	fmt.Println("Auto Research Engineer — project setup")
	fmt.Println("We pick ONE asset, turn \"is it good?\" into a single number, and optimize it overnight.")
	name := ask("Project name", "my-research")
	asset := ask("Asset file to optimize (the ONLY thing I may change)", "asset.txt")
	direction := ask("Is the score min (lower better) or max (higher better)?", "min")

	fmt.Println("\nFit check (all three required):")
	fmt.Println("  a) Is the score an objective number?")
	fmt.Println("  b) Does it return in minutes/hours, not weeks?")
	fmt.Println("  c) Can I actually change the asset file?")
	_ = ask("Press enter to acknowledge", "")

	files := map[string]string{
		filepath.Join(*dir, "instructions.md"): instructionsStub,
		filepath.Join(*dir, "score.sh"):        scorerStub,
		filepath.Join(*dir, ".gitignore"):      gitignoreStub,
		filepath.Join(*dir, asset):             "",
		filepath.Join(*dir, "autoresearch.toml"): fmt.Sprintf(`[project]
name = %q
instructions = "instructions.md"
asset = [%q]
scorer = "./score.sh"
direction = %q
max_rounds = 0
round_timeout = "10m"

[model]
backend = "managed"
context = 16384
temperature = 0.7

[run]
history_window = 8
`, name, asset, direction),
	}
	for path, body := range files {
		if _, err := os.Stat(path); err == nil {
			fmt.Printf("skip existing %s\n", path)
			continue
		}
		if err := os.WriteFile(path, []byte(body), modeFor(path)); err != nil {
			return err
		}
		fmt.Printf("wrote %s\n", path)
	}
	fmt.Println("\nNext: edit instructions.md and score.sh, then run `karp run`.")
	return nil
}

func modeFor(path string) os.FileMode {
	if strings.HasSuffix(path, ".sh") {
		return 0o755
	}
	return 0o644
}

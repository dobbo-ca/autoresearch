// Command karp runs the autoresearch optimization loop.
package main

import (
	"fmt"
	"os"

	"github.com/dobbo-ca/autoresearch/internal/version"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = cmdRun(os.Args[2:])
	case "init":
		err = cmdInit(os.Args[2:])
	case "report":
		err = cmdReport(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Println("karp", version.String())
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `karp — autoresearch optimization loop

Usage:
  karp init      scaffold a 3-file project (instructions/asset/scorer)
  karp run       run the overnight loop (downloads + launches the model)
  karp report    render report.md from rounds.jsonl
  karp version   print version`)
}

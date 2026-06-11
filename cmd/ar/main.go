// Command ar runs the autoresearch optimization loop.
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
		fmt.Println("ar", version.String())
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
	fmt.Fprintln(os.Stderr, `ar — autoresearch optimization loop

Usage:
  ar init      scaffold a 3-file project (instructions/asset/scorer)
  ar run       run the overnight loop (downloads + launches the model)
  ar report    render report.md from rounds.jsonl
  ar version   print version`)
}

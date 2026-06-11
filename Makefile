.PHONY: test build dist clean

VERSION_PKG=github.com/dobbo-ca/autoresearch/internal/version
VERSION?=$(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE=$(shell date -u '+%Y-%m-%d_%H:%M:%S')
LDFLAGS=-ldflags "-X $(VERSION_PKG).Version=$(VERSION) -X $(VERSION_PKG).Commit=$(COMMIT) -X $(VERSION_PKG).Date=$(BUILD_DATE)"

test:
	go test ./...

build:
	go build $(LDFLAGS) -o bin/ar ./cmd/ar

# Cross-build release archives. The managed runtime is macOS Apple Silicon
# (Metal) only today; add more GOOS/GOARCH pairs here as platforms are
# supported, and mirror them in the Homebrew formula + release matrix.
PLATFORMS=darwin/arm64

dist: clean
	@mkdir -p dist
	@for p in $(PLATFORMS); do \
		os=$${p%/*}; arch=$${p#*/}; \
		echo "building $$os/$$arch..."; \
		GOOS=$$os GOARCH=$$arch go build $(LDFLAGS) -o dist/ar ./cmd/ar; \
		tar -C dist -czf dist/autoresearch-$(VERSION)-$$os-$$arch.tar.gz ar; \
		rm dist/ar; \
	done
	@ls -lh dist/

clean:
	@rm -rf bin dist

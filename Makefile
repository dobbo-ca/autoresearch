.PHONY: test build
test:
	go test ./...
build:
	go build -o bin/ar ./cmd/ar

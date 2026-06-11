package brain

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func canned(content string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}}},
		})
	}))
}

func TestSubprocessProposeParsesResponse(t *testing.T) {
	srv := canned(`{"hypothesis":"set to 5","target_file":"value.txt","new_content":"5"}`)
	defer srv.Close()
	b := NewSubprocess(srv.URL, "test-model", 0.7)
	p, err := b.Propose(context.Background(), ProposeInput{
		Instructions: "approach 0", Asset: map[string]string{"value.txt": "10"}, Direction: "min",
	})
	if err != nil || p.TargetFile != "value.txt" || p.NewContent != "5" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

func TestSubprocessStripsThinkAndExtracts(t *testing.T) {
	srv := canned("<think>hmm</think>\n{\"hypothesis\":\"x\",\"target_file\":\"a\",\"new_content\":\"b\"}")
	defer srv.Close()
	b := NewSubprocess(srv.URL, "m", 0.0)
	p, err := b.Propose(context.Background(), ProposeInput{Direction: "min"})
	if err != nil || p.TargetFile != "a" {
		t.Fatalf("p %+v err %v", p, err)
	}
}

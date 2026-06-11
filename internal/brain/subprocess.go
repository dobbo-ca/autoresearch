package brain

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Subprocess is a Brain backed by an OpenAI-compatible HTTP server
// (a managed or external llama-server running the model on the GPU).
type Subprocess struct {
	endpoint    string
	model       string
	temperature float64
	client      *http.Client
}

// NewSubprocess builds a Brain that calls {endpoint}/v1/chat/completions.
func NewSubprocess(endpoint, model string, temperature float64) *Subprocess {
	return &Subprocess{
		endpoint:    strings.TrimRight(endpoint, "/"),
		model:       model,
		temperature: temperature,
		client:      &http.Client{Timeout: 10 * time.Minute},
	}
}

func (s *Subprocess) Close() error { return nil }

func (s *Subprocess) Propose(ctx context.Context, in ProposeInput) (Proposal, error) {
	system, user := BuildMessages(in)
	body := map[string]any{
		"model":       s.model,
		"temperature": s.temperature,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"response_format": map[string]string{"type": "json_object"},
		"grammar":         Grammar(), // llama-server enforces; ignored elsewhere
	}
	buf, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.endpoint+"/v1/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return Proposal{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return Proposal{}, fmt.Errorf("call model: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Proposal{}, fmt.Errorf("model returned status %d", resp.StatusCode)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return Proposal{}, fmt.Errorf("decode response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return Proposal{}, fmt.Errorf("model returned no choices")
	}
	return ParseProposal(parsed.Choices[0].Message.Content)
}

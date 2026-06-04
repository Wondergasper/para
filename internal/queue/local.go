package queue

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"apg/internal/proto"
)

// LocalQueue runs the Python pipeline directly as a subprocess per job.
// No NATS, no Docker required — just Python on PATH and the workers/ package.
type LocalQueue struct {
	mu         sync.Mutex
	handler    func(*proto.JobResponse)
	onStart    func(id string) // called when job starts running
	python     string
	repoRoot   string
	maxTimeout time.Duration
}

// NewLocalQueue creates a LocalQueue and validates that Python is reachable.
func NewLocalQueue() (*LocalQueue, error) {
	py, err := resolvePython()
	if err != nil {
		return nil, err
	}

	root, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("get working dir: %w", err)
	}

	log.Printf("[local-queue] Using Python: %s", py)
	log.Printf("[local-queue] Repo root: %s", root)
	return &LocalQueue{
		python:     py,
		repoRoot:   root,
		maxTimeout: 3 * time.Minute, // generous; Ollama can be slow
	}, nil
}

// SetOnStart registers a callback fired when a job transitions to "running".
func (q *LocalQueue) SetOnStart(fn func(id string)) {
	q.mu.Lock()
	q.onStart = fn
	q.mu.Unlock()
}

// SubscribeResults registers the result callback.
func (q *LocalQueue) SubscribeResults(_ context.Context, handler func(*proto.JobResponse)) error {
	q.mu.Lock()
	q.handler = handler
	q.mu.Unlock()
	return nil
}

// PublishJob spawns a goroutine to run the pipeline and calls the result handler.
func (q *LocalQueue) PublishJob(_ context.Context, req *proto.JobRequest, _ bool) error {
	q.mu.Lock()
	handler := q.handler
	onStart := q.onStart
	q.mu.Unlock()

	go func() {
		// Notify that the job is now running
		if onStart != nil {
			onStart(req.Id)
		}
		log.Printf("[local-queue][%s] starting pipeline | func=%s", req.Id, req.FuncName)
		resp := q.runPipeline(req)
		log.Printf("[local-queue][%s] pipeline done | success=%v score=%.2f", req.Id, resp.Success, resp.Score)
		if handler != nil {
			handler(resp)
		}
	}()
	return nil
}

// Close is a no-op for LocalQueue.
func (q *LocalQueue) Close() {}

// runPipeline calls: python scripts/run_pipeline.py --json
// and parses the JSON result back into a proto.JobResponse.
func (q *LocalQueue) runPipeline(req *proto.JobRequest) *proto.JobResponse {
	jobID := req.Id

	// Build the input payload as JSON passed on stdin
	input, err := json.Marshal(map[string]string{
		"id":            jobID,
		"source":        req.Source,
		"func_name":     req.FuncName,
		"model_version": req.ModelVersion,
		"file_path":     req.FilePath,
	})
	if err != nil {
		return errResp(jobID, fmt.Sprintf("failed to marshal job payload: %v", err))
	}

	scriptPath := filepath.Join(q.repoRoot, "scripts", "run_pipeline.py")

	// Verify script exists before launching
	if _, err := os.Stat(scriptPath); err != nil {
		return errResp(jobID, fmt.Sprintf("run_pipeline.py not found at %s: %v", scriptPath, err))
	}

	ctx, cancel := context.WithTimeout(context.Background(), q.maxTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, q.python, scriptPath, "--json", "--max-candidates", "2", "--max-rounds", "2")
	cmd.Stdin = bytes.NewReader(input)
	cmd.Dir = q.repoRoot
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("APG_JOB_ID=%s", jobID),
		"PYTHONUNBUFFERED=1", // ensures Python doesn't buffer stderr
		"PYTHONIOENCODING=utf-8",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	log.Printf("[local-queue][%s] launching: %s %s --json", jobID, q.python, scriptPath)

	runErr := cmd.Run()

	// Always log stderr (Python pipeline logs go there)
	if stderr.Len() > 0 {
		for _, line := range splitLines(stderr.String()) {
			if line != "" {
				log.Printf("[local-queue][%s] py| %s", jobID, line)
			}
		}
	}

	if runErr != nil {
		msg := fmt.Sprintf("pipeline subprocess error: %v", runErr)
		if stdout.Len() > 0 {
			msg += fmt.Sprintf("\nstdout: %s", stdout.String())
		}
		log.Printf("[local-queue][%s] ERROR: %s", jobID, msg)
		return errResp(jobID, msg)
	}

	outBytes := stdout.Bytes()
	if len(outBytes) == 0 {
		return errResp(jobID, "pipeline produced no output — check that Ollama is running and the model is loaded")
	}

	// Find the line that represents the JSON result (starts with '{' and ends with '}')
	var jsonBytes []byte
	for _, line := range splitLines(string(outBytes)) {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
			jsonBytes = []byte(trimmed)
			break
		}
	}

	if len(jsonBytes) == 0 {
		// Fallback to old heuristic if no single line matches
		jsonStart := bytes.LastIndex(outBytes, []byte("{"))
		if jsonStart >= 0 {
			jsonBytes = outBytes[jsonStart:]
		} else {
			jsonBytes = outBytes
		}
	}

	var result struct {
		Success       bool    `json:"success"`
		Score         float64 `json:"score"`
		BestCandidate string  `json:"best_candidate"`
		AnnotatedIR   any     `json:"annotated_ir"`
		Rounds        int     `json:"rounds"`
		Error         string  `json:"error"`
		Attempts      []struct {
			Candidate string  `json:"candidate"`
			Score     float64 `json:"score"`
			Gate      string  `json:"gate"`
			Error     string  `json:"error"`
		} `json:"attempts"`
	}
	if err := json.Unmarshal(jsonBytes, &result); err != nil {
		log.Printf("[local-queue][%s] JSON parse error: %v\nraw stdout: %s", jobID, err, string(stdout.Bytes()))
		return errResp(jobID, fmt.Sprintf("failed to parse pipeline output: %v\nraw: %.500s", err, string(stdout.Bytes())))
	}

	irJSON := ""
	if result.AnnotatedIR != nil {
		if b, err := json.Marshal(result.AnnotatedIR); err == nil {
			irJSON = string(b)
		}
	}

	resp := &proto.JobResponse{
		Id:              jobID,
		Success:         result.Success,
		Score:           result.Score,
		BestCandidate:   result.BestCandidate,
		AnnotatedIrJson: irJSON,
		Rounds:          int32(result.Rounds),
		Error:           result.Error,
	}
	for _, a := range result.Attempts {
		resp.Attempts = append(resp.Attempts, &proto.Attempt{
			Candidate: a.Candidate,
			Score:     a.Score,
			Gate:      a.Gate,
			Error:     a.Error,
		})
	}
	return resp
}

func errResp(id, msg string) *proto.JobResponse {
	return &proto.JobResponse{Id: id, Success: false, Score: 0, Error: msg}
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i, c := range s {
		if c == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

// resolvePython finds a suitable Python 3 executable.
func resolvePython() (string, error) {
	candidates := []string{"python", "python3", "py"}
	if runtime.GOOS == "windows" {
		candidates = []string{"python", "py", "python3"}
	}
	for _, name := range candidates {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("python not found on PATH — install Python 3.11+")
}

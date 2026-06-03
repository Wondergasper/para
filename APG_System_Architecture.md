# AUTONOMOUS PARALLEL CODE GENERATION
## with Formal Correctness Guarantees
### Full System Architecture, Development Phases & ML Strategy
#### Go + Python Implementation Plan
*Based on survey: Autonomous Parallel Code Generation with Formal Correctness Guarantees*
*2025 — CAMLDS / Wonder*
---
## 1. System Overview
This document defines the full architecture, phased development plan, and ML strategy for building an Autonomous Parallel Code Generation system with Formal Correctness Guarantees.
The system takes sequential source code (C/C++) as input and produces a verified parallel implementation — with no human in the loop.
The architecture has five layers, each independently buildable and testable:
| Layer | Responsibility | Technology |
| :--- | :--- | :--- |
| **Layer 1 — Analyse** | Parse source, classify regions, build dependency graph | Go + Python (tree-sitter, ISL) |
| **Layer 2 — Generate** | Produce candidate parallel implementations | Python (Ollama local LLM) |
| **Layer 3 — Verify** | Compile, correctness-test, race-check all candidates | Go + GCC + CBMC + ThreadSanitizer |
| **Layer 4 — Retry Loop** | Feed failures back to generator with critique | Go state machine + Python LLM |
| **Layer 5 — Learn (future)** | Collect proof corpus, emit RL reward signal | Python (PyTorch — Phase 4 only) |
### 1.1 Language Division of Responsibilities
* **Go** owns all orchestration: the HTTP/gRPC API, job routing, the Layer 4 retry state machine, the NATS task queue, observability, and the proof corpus store. It is chosen for its native concurrency model (goroutines, channels), which maps cleanly to a pipeline that runs multiple verification jobs in parallel.
* **Python** owns all ML and analysis: LLM calls via Ollama, tree-sitter parsing, ISL polyhedral bindings, subprocess calls to Lean 4 and Verus, and — in Phase 4 — RL training loops. It is chosen because the entire ML ecosystem (PyTorch, Hugging Face, ISL bindings) is Python-native.
### 1.2 LLM Strategy (Zero Budget)
#### Primary: Ollama (local, free)
* **Details:** Run DeepSeek-Coder 6.7B or Qwen2.5-Coder 7B on your local machine. Requires ~4–8 GB RAM.
* **Install:** `curl -fsSL https://ollama.com/install.sh | sh`
* **Pull model:** `ollama pull deepseek-coder:6.7b`
* **API endpoint:** `http://localhost:11434/v1/chat/completions` (OpenAI-compatible)
#### Fallback: Free cloud tiers (no credit card needed for these quotas)
* **Groq:** Free tier, very fast Llama 3.1 70B inference. Best free cloud option.
* **Google Gemini API:** 60 requests/min free tier, strong at code tasks.
* **Hugging Face Inference API:** Free for Qwen2.5-Coder-7B-Instruct.
* *Note: All expose an OpenAI-compatible endpoint — same client code works for all three.*
### 1.3 Repository Layout
Monorepo structure — one repo, two language subtrees:
```text
apg/                              ← project root
├── cmd/
│   └── orchestrator/             ← Go binary (API + pipeline)
├── internal/
│   ├── gateway/                  ← HTTP/gRPC handlers
│   ├── pipeline/                 ← layer state machine
│   ├── queue/                    ← NATS producer/consumer
│   └── store/                    ← SQLite / Postgres + S3
├── proto/                        ← shared .proto definitions
├── workers/
│   ├── code_understanding/       ← Python gRPC worker
│   ├── gen_openmp/               ← Python LLM gen worker
│   ├── spec_generator/           ← Python CSL spec + verifier bridge
│   └── training/                 ← Python RL (Phase 4 only)
├── scripts/                      ← dev tooling
└── docker-compose.yml            ← local dev stack
```
---
## 2. Architecture — Five Layers in Detail
### Layer 1 — Code Understanding
Layer 1 accepts a sequential C/C++ source file and produces an `AnnotatedIR` — a JSON structure that classifies every function and loop into one of three categories: polyhedral (affine loop bounds, amenable to formal transformation), irregular (data-dependent access, needs LLM-guided parallelisation), or sequential (inherently serial, do not touch).
#### Components
| Component | Details |
| --- | --- |
| **tree-sitter parser (Python)** | Parses C/C++ into AST. Extracts: loop nests, function signatures, call graph, pointer dereferences. |
| **Dataflow / dependency graph (Go)** | Builds a DAG from the AST. Nodes = statements. Edges = read-after-write, write-after-read. Identifies parallelisable loop bodies. |
| **Polyhedral classifier (Python)** | Uses ISL (Integer Set Library) via islpy to test if loop bounds and array accesses are affine. Outputs: affine / non-affine per loop. |
| **LLM semantic annotator (Python)** | For irregular regions, prompts local LLM to add a natural-language summary of what the region computes. Feeds Layer 2 prompt context. |
#### Output — AnnotatedIR schema
```json
{ 
  "file": "matmul.c",
  "regions": [
    { 
      "id": "fn_matmul_loop_0",
      "type": "polyhedral",
      "lines": [14, 28],
      "deps": ["A", "B"],
      "writes": ["C"],
      "affine": true,
      "race_risk": "low" 
    },
    { 
      "id": "fn_graph_bfs",
      "type": "irregular",
      "lines": [45, 90],
      "summary": "BFS over adjacency list — data-dependent neighbour access"
    }
  ]
}
```
### Layer 2 — Parallel Code Generation
Layer 2 receives the `AnnotatedIR` and generates 3–5 candidate parallel implementations per region. It uses the local LLM (Ollama) with structured prompts that include the region type, dependency graph, and any prior failure traces from Layer 4.
#### Generation strategy per region type
| Region type | Approach |
| --- | --- |
| **Polyhedral (affine loops)** | Apply CTT (Code the Transforms): generate a formal loop interchange / tiling rewrite rule. Correctness guaranteed by construction for this class. |
| **Irregular (data-dependent)** | Multi-shot LLM prompting: ask model to generate OpenMP pragma annotations, provide dependency graph as context. Generate 3 variants at different temperature settings. |
| **Sequential (inherently serial)** | Skip. Return original code unchanged. Mark as non-parallelisable in output. |
#### Prompt structure (Layer 2 → LLM)
```text
SYSTEM: You are an expert HPC programmer. Given a C function and its
dependency graph, produce an OpenMP-parallelised version.
Output ONLY valid C code. No explanations.
USER:   Function: [source code]
        Dependencies: [JSON dep graph]
        Race risk regions: [list]
        Previous attempt failed with: [error if retry]
        Generate parallel version:
```
### Layer 3 — Formal Correctness Verification
Layer 3 is the verification pipeline. It runs three sequential gates on each candidate. A candidate must pass all three to be accepted. The gates are ordered by speed: fast gates first to avoid wasting time on candidates that fail compilation.
| Gate | Tool | What it proves |
| --- | --- | --- |
| **Gate 1: Compile** | `gcc -fopenmp -Wall -Werror` | Candidate is syntactically and type-correct. Runs in < 1 second. |
| **Gate 2: Correctness** | Run parallel vs sequential on 10 test inputs, diff outputs | Functional equivalence on the test distribution. Not a formal proof but catches wrong transformations. |
| **Gate 3: Race freedom** | `gcc -fsanitize=thread` (ThreadSanitizer) | Dynamic detection of data races during execution. Free, built into GCC/Clang. |
If all three gates pass, the candidate is promoted as verified output. If any gate fails, the failure message (compiler error, diff output, or TSAN report) is packaged as a `FailureTrace` and sent to Layer 4.
*Future extension (Phase 3+): add CBMC model checking for race freedom on bounded inputs, and Lean 4 / Verus for machine-checked functional equivalence proofs on affine regions.*
### Layer 4 — Self-Verification Loop
Layer 4 is a Go state machine that manages the retry budget. When Layer 3 returns a `FailureTrace`, Layer 4 classifies the failure, constructs a critique prompt, and re-invokes Layer 2 with the additional context. The loop runs up to `MaxRetries` (default: 3) before giving up and returning the best unverified candidate.
| Failure type | Critique strategy |
| --- | --- |
| **Compile error** | Append full compiler error to prompt. Ask LLM to fix the specific error only. |
| **Output mismatch** | Show differing inputs and outputs. Ask LLM to identify what transformation was wrong. |
| **Race condition (TSAN)** | Show TSAN report (thread IDs, memory address, stack traces). Ask LLM to add missing synchronisation. |
| **Budget exhausted** | Return best candidate with test-based correctness score. Flag as unverified in output. |
### Layer 5 — Training Signal (Phase 4 only)
Layer 5 is not required for a functioning system. It becomes relevant in Phase 4 when you have accumulated enough verified candidates to fine-tune the generation model. See Section 5 (ML Strategy) for full details.
### Shared Infrastructure
| Component | Phase 1 implementation → Phase 3 upgrade |
| --- | --- |
| **Task queue** | Phase 1: Go channels (in-process). Phase 2: NATS JetStream (persistent, distributed). |
| **Storage** | Phase 1: SQLite (single file). Phase 3: PostgreSQL + S3-compatible object store (MinIO locally). |
| **Observability** | Phase 2: structured logging (zerolog). Phase 3: OpenTelemetry traces + Prometheus metrics. |
| **Verifier processes** | Lean 4 binary, Verus binary, CBMC — all managed as Go subprocesses with timeout + kill. |
---
## 3. Development Phases
The system is built in four phases. Each phase produces a working, demonstrable system — not a prototype that needs to be thrown away. Each phase adds a layer of correctness guarantee on top of the previous one.
### Phase 1: Foundation Pipeline (Weeks 1–4)
* **Goal:** End-to-end working system.
* **Overview:** Phase 1 builds the simplest possible version of the full pipeline. Every layer is real but minimal. By the end of Phase 1 you can submit a C file and receive an OpenMP-annotated parallel version that compiles and produces correct output on test inputs.
#### Phase 1 — What you build
| Component | Implementation |
| --- | --- |
| **Go HTTP server** | Single endpoint: `POST /submit` (accepts C source), `GET /result/:id`. Uses `net/http` standard library. No external dependencies. |
| **Layer 1 — Parser** | Python script: tree-sitter parses the C file, extracts all for-loops and their line ranges. Outputs AnnotatedIR JSON. No polyhedral analysis yet. |
| **Layer 2 — Generator** | Python script: calls Ollama API with a structured prompt. Generates one OpenMP candidate. Saves to disk. |
| **Layer 3 — Verifier** | Bash pipeline wrapped in Go subprocess: `gcc -fopenmp` compile → run both versions → diff outputs → `gcc -fsanitize=thread`. Returns pass/fail + message. |
| **Layer 4 — Retry** | Simple Go for-loop: if Layer 3 fails, append error to prompt, re-run Layer 2. Max 3 retries. |
| **Storage** | SQLite via `go-sqlite3`. One table: `jobs(id, source, status, result, created_at)`. |
#### Phase 1 — Development approach
1. Set up monorepo. `go mod init apg`. Create `workers/` Python package. Write `docker-compose.yml` with Ollama service.
2. Build Go HTTP server with two routes. Hard-code the pipeline as sequential function calls. No queue yet.
3. Write tree-sitter parser in Python. Test on 5 simple C loop programs. Validate `AnnotatedIR` JSON schema.
4. Write Ollama client in Python (`httpx`). Write the Layer 2 prompt template. Test generation quality manually.
5. Write the bash verification pipeline. Wrap in Go `exec.Command` with timeout. Test on known-correct and known-buggy parallel code.
6. Wire all layers together in the Go HTTP handler. Test the full pipeline on 10 C programs.
#### Phase 1 — Acceptance criteria
* System accepts a C file via HTTP POST and returns a job ID.
* For a simple loop (matrix multiply, array sum), system produces a correct OpenMP version within 3 retries.
* Verification pipeline correctly rejects a hand-crafted race condition (test case).
* End-to-end latency < 60 seconds on a laptop with Ollama running locally.
#### Phase 1 — Tech stack
| Tool | Install command |
| --- | --- |
| **Go 1.22+** | https://go.dev/dl |
| **Python 3.11+** | system or pyenv |
| **Ollama + DeepSeek-Coder 6.7B** | `curl -fsSL https://ollama.com/install.sh \│ sh && ollama pull deepseek-coder:6.7b` |
| **tree-sitter (Python)** | `pip install tree-sitter tree-sitter-c tree-sitter-cpp` |
| **httpx (Python HTTP)** | `pip install httpx` |
| **go-sqlite3** | `go get github.com/mattn/go-sqlite3` |
| **GCC with OpenMP + TSAN** | `apt install gcc` (Linux) / `brew install gcc` (Mac) |
---
### Phase 2: Polyhedral Analysis + Multi-Candidate Generation (Weeks 5–9)
* **Goal:** Formal region classification + parallel candidate pool.
* **Overview:** Phase 2 adds the polyhedral analyser (Layer 1 upgrade) and multi-candidate generation (Layer 2 upgrade). The system now classifies loop regions formally and generates 3–5 candidates per region, picking the first that passes all verification gates.
#### Phase 2 — What you build
| Component | Implementation |
| --- | --- |
| **ISL polyhedral classifier** | Python: `islpy` bindings. For each loop in AnnotatedIR, test if bounds and index expressions are affine. Tag: polyhedral / irregular. |
| **CTT transform engine** | Python: for polyhedral loops, apply correct-by-construction rewrite rules (loop interchange, tiling, distribution). No LLM needed — pure algorithmic. |
| **Multi-candidate pool** | Layer 2 now generates 3 candidates: one CTT (polyhedral), two LLM (temperature 0.2 and 0.7). Layer 3 runs gates in parallel using Go goroutines. |
| **NATS task queue** | Replace Go channels with NATS JetStream. Workers become independent processes. Go orchestrator publishes tasks; Python workers subscribe. |
| **gRPC contracts** | Define `.proto` files for `AnnotatedIR`, `CandidateSet`, `FailureTrace`, `ProofResult`. Generate Go + Python stubs. This is the Go–Python boundary. |
| **Structured logging** | `zerolog` in Go. Python logging with JSON formatter. Correlation IDs propagated through the pipeline. |
#### Phase 2 — Development approach
1. Install `islpy`. Write polyhedral classifier. Test on affine loops from LOOPerSet dataset (public, free). Validate classification accuracy.
2. Implement CTT rewrite rules for: loop interchange, loop tiling (fixed tile size), loop distribution. Test each rule independently with unit tests.
3. Define `.proto` files. Run `protoc` to generate Go and Python stubs. Write integration test: Go sends `AnnotatedIR` proto → Python returns `CandidateSet` proto.
4. Set up NATS JetStream in `docker-compose.yml`. Convert Go pipeline to publish/subscribe model. Convert Python Layer 2 to a NATS subscriber.
5. Upgrade Layer 3 to run Gate 2 and Gate 3 in parallel Go goroutines per candidate. Collect results via channel with timeout.
6. Run full Phase 2 pipeline on 20 C programs. Measure: correctness rate, race detection rate, retry rate.
#### Phase 2 — Additional dependencies
| Tool | Notes |
| --- | --- |
| **islpy** | `pip install islpy` (ISL Python bindings for polyhedral analysis) |
| **NATS server** | `docker run nats:latest` or `brew install nats-server` |
| **nats.go** | `go get github.com/nats-io/nats.go` |
| **nats.py** | `pip install nats-py` |
| **grpcio + protobuf** | `pip install grpcio grpcio-tools protobuf` |
| **google.golang.org/grpc** | `go get google.golang.org/grpc` |
| **zerolog** | `go get github.com/rs/zerolog` |
---
### Phase 3: Formal Verification Bridge (Gap 3) (Weeks 10–16)
* **Goal:** Machine-checked proofs for affine regions.
* **Overview:** Phase 3 addresses Gap 3 from the survey — the hub gap whose solution cascades to every other gap. It adds Layer 3 Gate 4: a formal proof generated by the LLM and checked by Lean 4 or Verus. This is the hardest phase and the one with the most research risk. It is scoped conservatively: formal proofs are attempted only for affine (polyhedral) regions where the theory is well-understood.
#### Phase 3 — What you build
| Component | Implementation |
| --- | --- |
| **Lean 4 subprocess manager (Go)** | Go: `exec.Command` to launch Lean 4 binary. Stream stdout/stderr. Kill on timeout. Parse output for proof success / failure state. |
| **Verus subprocess manager (Go)** | Same pattern as Lean 4. Verus targets Rust; requires the C candidate to be transpiled to Rust first (use `c2rust` tool). |
| **CBMC model checker (Go)** | Easier entry point than Lean 4. CBMC can check race freedom and bounded functional correctness for C directly. Add as Gate 3b before Lean 4. |
| **CSL Spec Generator (Python)** | Python: given the sequential reference + parallel candidate + AnnotatedIR, prompt local LLM to generate a Lean 4 / Verus spec. This is the most prompt-engineering-intensive component. |
| **Proof feedback loop** | If Lean 4 rejects the spec, parse the error message and re-prompt LLM to fix the specific proof obligation. Up to 5 rounds. |
| **Proof corpus store** | PostgreSQL table: `proofs(id, region_id, spec, proof, verifier, success, created_at)`. All verified proofs are stored for Layer 5. |
#### Phase 3 — CSL Spec Generator prompt structure
```text
SYSTEM: You are a formal verification expert. Given a sequential C function
        and its OpenMP-parallelised version, write a Lean 4 theorem that
        states: the parallel version computes the same result as the
        sequential version for all valid inputs, and no data races occur.
        Output ONLY valid Lean 4 code.
USER:   Sequential:  [source]
        Parallel:    [source]
        Region type: polyhedral (affine loop bounds)
        Dependencies: [dep graph JSON]
        Write the Lean 4 correctness theorem:
```
#### Phase 3 — Risk management
* **Research risk:** LLM-generated Lean 4 proofs are hard. VERINA benchmark (2025) shows: 61% code correctness but only 3.6% proof success with best models. With a 6B local model, proof success rate will be lower — possibly < 1% initially.
* **Mitigation:** Start with CBMC (deterministic, no LLM needed) as Gate 3b. CBMC handles bounded functional correctness and some race freedom checks for C programs without a proof assistant. Gate 4 (Lean 4) is an optional enhancement, not a blocker for Phase 3 acceptance.
#### Phase 3 — Additional dependencies
| Tool | Install |
| --- | --- |
| **Lean 4** | `curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \│ sh` |
| **Verus** | `git clone https://github.com/verus-lang/verus && cargo build --release` |
| **CBMC** | `apt install cbmc` or `brew install cbmc` |
| **c2rust** (C→Rust transpiler) | `cargo install c2rust` |
| **PostgreSQL** | `docker run postgres:16` or `apt install postgresql` |
| **pgx/v5** (Go Postgres driver) | `go get github.com/jackc/pgx/v5` |
---
### Phase 4: Training Signal & Fine-Tuning (Layer 5) (Weeks 17+)
* **Goal:** Improve generation model from verified corpus.
* **Overview:** Phase 4 activates Layer 5: the training signal loop. By this point the system has accumulated a corpus of verified (source, parallel, proof) triples. Phase 4 uses this corpus to fine-tune the generation model, progressively improving the quality and verifiability of generated code. See Section 5 (ML Strategy) for full details.
#### Phase 4 — What you build
| Component | Implementation |
| --- | --- |
| **Proof corpus exporter** | Python script: queries PostgreSQL for verified proofs, formats as JSONL training dataset. |
| **RL reward emitter** | Python: for each pipeline run, emit a reward event (1.0 = all gates pass, 0.0 = failure, partial scores for partial passes). |
| **Fine-tune scheduler** | Python: when corpus reaches N examples (e.g. 500), trigger a fine-tuning run on the generation model using HuggingFace PEFT / LoRA. |
| **Model registry** | Simple filesystem structure: `models/{version}/adapter_weights/`. Go orchestrator reads current model version from config. |
| **A/B evaluation** | Route 20% of new jobs to the fine-tuned model. Compare pass rate vs base model. Promote if better. |
---
## 4. Phase Timeline Summary
| Phase | Duration | Key deliverable |
| --- | --- | --- |
| **Phase 1: Foundation** | 4 weeks | End-to-end pipeline: C in → OpenMP out, compile + diff + TSAN verified |
| **Phase 2: Polyhedral + Multi-candidate** | 5 weeks | Formal region classification, CTT engine, NATS queue, gRPC boundary |
| **Phase 3: Formal Verification Bridge** | 7 weeks | CBMC gate + Lean 4 / Verus proof loop for affine regions |
| **Phase 4: ML Fine-tuning** | Ongoing | Proof corpus collection, LoRA fine-tuning, A/B evaluation |
### Minimum Viable Research Contribution (Phase 1 + 2)
A working autonomous pipeline that takes sequential C code and produces OpenMP parallel code with three levels of correctness checking (compile, functional equivalence, ThreadSanitizer) and a retry loop that feeds failure traces back to the generator — with zero training and zero API cost. **This alone addresses Gaps 1, 2, and partially 3 from your survey and is a publishable system.**
---
## 5. ML Strategy — Separated
This section covers all machine learning aspects of the system independently. It is separated here so the ML roadmap can be read and planned independently of the engineering phases. Three questions are addressed: Do you need ML What kind When
### 5.1 Do You Need to Train an ML Model
**Short answer:** No — not in Phases 1, 2, or 3. Here is the full breakdown:
| Component | ML required | What you actually do |
| --- | --- | --- |
| **LLM for code generation** | No training | Call Ollama API (local). Prompt engineering only. |
| **LLM for critique / repair** | No training | Same Ollama API. Different prompt template. |
| **LLM for CSL spec generation** | No training | Same Ollama API. Structured output prompt. |
| **Polyhedral classifier** | No ML at all | ISL library — pure mathematical decision procedure. |
| **CTT transform engine** | No ML at all | Rule-based loop transformation — deterministic algorithm. |
| **Dependency graph builder** | No ML at all | Graph algorithm on AST — O(n) dataflow analysis. |
| **Formal verifiers (Lean/Verus/CBMC)** | No ML at all | External binaries. Subprocess calls. |
| **LOOPer polyhedral scheduler** | Download checkpoint | Pre-trained model from paper repo. No training needed. |
| **RL fine-tuning (Layer 5)** | Yes — Phase 4 only | Fine-tune DeepSeek-Coder on verified corpus. Needs GPU. |
| **Process reward model** | Optional — Phase 4 | Use LLM-as-judge or download open-source PRM. |
### 5.2 The Only LLM Usage: Prompt Engineering
In Phases 1–3, 100% of the LLM usage is prompt engineering — crafting structured inputs that guide a pre-trained model (DeepSeek-Coder 6.7B) to produce useful outputs. No gradients, no training loops, no GPU required. The model runs on CPU via Ollama.
The system uses four distinct prompt templates:
* **T1 — Region Annotation:** Layer 1: given a C function, summarise what it computes and identify parallelism opportunities.
* **T2 — Parallel Code Gen:** Layer 2: given source + dep graph, generate OpenMP-parallelised version. Include pragma placement reasoning.
* **T3 — Failure Critique:** Layer 4: given failed candidate + error message, identify the root cause and suggest a specific fix.
* **T4 — CSL Spec Gen:** Layer 3 (Phase 3): given sequential + parallel + dep graph, write Lean 4 / Verus correctness theorem.
### 5.3 Local LLM Setup (Ollama)
Step-by-step setup for the local LLM:
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
# 2. Pull models (pick one based on your RAM)
ollama pull deepseek-coder:1.3b    # 800MB — minimal RAM
ollama pull deepseek-coder:6.7b    # 4GB  — recommended
ollama pull qwen2.5-coder:7b       # 4GB  — alternative, strong at reasoning
# 3. Verify the API is running
curl http://localhost:11434/v1/chat/completions 
  -H 'Content-Type: application/json' 
  -d '{"model": "deepseek-coder:6.7b", "messages": [{"role": "user", "content": "hello"}]}'
# 4. Python client (same code works for Groq / Gemini by changing base_url)
pip install openai   # openai package works with Ollama's compatible API
```
#### Python LLM client (model-agnostic)
```python
from openai import OpenAI
def get_client(provider='ollama'):
    if provider == 'ollama':
        return OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    elif provider == 'groq':
        return OpenAI(base_url='https://api.groq.com/openai/v1', api_key=GROQ_KEY)
    elif provider == 'gemini':
        return OpenAI(base_url='https://generativelanguage.googleapis.com/v1beta/openai/', api_key=GEMINI_KEY)
def generate_parallel(source: str, dep_graph: dict, error: str = '') -> str:
    client = get_client()  # swap provider string to switch LLM
    resp = client.chat.completions.create(
        model='deepseek-coder:6.7b',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': build_prompt(source, dep_graph, error)},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
```
### 5.4 Phase 4 ML — Fine-Tuning Details
Phase 4 fine-tuning is triggered when the proof corpus contains at least 500 verified triples. It uses LoRA (Low-Rank Adaptation) — a parameter-efficient technique that fine-tunes a small adapter on top of the frozen base model. LoRA runs on a single consumer GPU (16GB VRAM) and is free to use via the HuggingFace PEFT library.
#### Fine-tuning dataset format
```json
{
  "messages": [
    {"role": "system", "content": "Generate a correct OpenMP parallel version..."},
    {"role": "user",   "content": "Source: [C code]\nDep graph: [JSON]"},
    {"role": "assistant", "content": "[verified parallel C code]"}
  ]
}
```
#### LoRA fine-tuning script (minimal)
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct')
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct')
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05, bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=load_corpus_from_postgres(),
    args=TrainingArguments(
        output_dir='./adapter', num_train_epochs=3,
        per_device_train_batch_size=4, fp16=True
    ),
)
trainer.train()
model.save_pretrained('./adapter')
```
#### Phase 4 ML dependencies
| Package | Purpose |
| --- | --- |
| **transformers** | `pip install transformers` — load DeepSeek-Coder base model |
| **peft** | `pip install peft` — LoRA adapter training |
| **trl** | `pip install trl` — SFTTrainer for supervised fine-tuning |
| **datasets** | `pip install datasets` — HuggingFace dataset format for corpus |
| **bitsandbytes** | `pip install bitsandbytes` — 4-bit quantisation for low-VRAM GPUs |
| **accelerate** | `pip install accelerate` — multi-GPU / mixed-precision training |
| **torch** | `pip install torch` — PyTorch base (CUDA build for GPU) |
### 5.5 RL with Verifier Reward (Advanced)
The verifier-as-reward paradigm (Rao et al., 2025) is the most powerful ML approach available for this system. Instead of learning from human-labelled data, the model learns from the verifier's binary pass/fail signal — which is perfectly reliable and infinitely scalable.
The reward function for a generated candidate C given reference R:
```python
def reward(candidate, reference, test_inputs):
    r = 0.0
    if compiles(candidate):                       r += 0.2
    if outputs_match(candidate, reference, test_inputs): r += 0.4
    if no_race_conditions(candidate):             r += 0.2
    if formal_proof_exists(candidate, reference): r += 0.2
    return r  # 0.0 to 1.0
```
This reward function directly maps to the three-layer correctness model in your survey (Gap 2). The formal proof component (0.2) becomes available only after Phase 3 is complete. RL training using this reward is implemented via GRPO or PPO in the `trl` library.
### 5.6 ML Risk Register
| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| **Local LLM quality too low for proof generation** | High | Start with CBMC (deterministic). Lean 4 is optional enhancement, not blocker. |
| **Proof corpus too small to fine-tune** | Medium | Run pipeline on LOOPerSet (28M programs) to generate synthetic examples. |
| **LoRA fine-tuning makes model worse (catastrophic forgetting)** | Low | Keep adapter separate from base. A/B test every fine-tuned version before promoting. |
| **RL reward hacking (proof gaming)** | Medium | Use holdout test suite not seen during RL. Add specification coverage metric. |
| **GPU not available for Phase 4** | Medium | Use Google Colab free tier (T4 GPU) for fine-tuning runs. Export adapter weights locally. |
---
## 6. Research Gaps Addressed per Phase
The seven gaps from the survey are addressed progressively across the four development phases.
| Gap (from survey) | Addressed in | How |
| --- | --- | --- |
| **G1: Concurrency bug detection** | Phase 1 | ThreadSanitizer integration in Layer 3 Gate 3. |
| **G2: Three-layer correctness** | Phase 1–2 | Compile → functional equiv → race freedom gates in sequence. |
| **G3: Seq→parallel + formal proof** | Phase 3 | CBMC + Lean 4 / Verus bridge for affine regions. |
| **G4: Cross-vendor portability** | Phase 3+ | Verus targets Rust (via c2rust), portable across vendors. |
| **G5: HPC training data scarcity** | Phase 4 | Bootstrap corpus from pipeline runs on LOOPerSet. |
| **G6: Static correctness predictor** | Phase 2 | Polyhedral classifier predicts parallelisability before generation. |
| **G7: Scalability as correctness** | Phase 3+ | CBMC bounded model checking at multiple thread counts. |
---
## 7. Quick-Start Checklist (Phase 1)
Everything needed to have a running system within a day:
1. **Install Go 1.22+** → https://go.dev/dl
2. **Install Python 3.11+** → system package manager or pyenv
3. **Install Ollama** → `curl -fsSL https://ollama.com/install.sh | sh`
4. **Pull DeepSeek-Coder** → `ollama pull deepseek-coder:6.7b`
5. **Install GCC with OpenMP and ThreadSanitizer** → `apt install gcc` (Linux)
6. **Install tree-sitter** → `pip install tree-sitter tree-sitter-c tree-sitter-cpp`
7. **Install httpx and openai** → `pip install httpx openai`
8. **Install go-sqlite3** → `go get github.com/mattn/go-sqlite3`
9. **Clone repo and run docker-compose up** → starts Ollama + SQLite
10. **Submit test C file** → `curl -X POST http://localhost:8080/submit -F 'file=@matmul.c'`
### Minimum Hardware Requirements
* **Phase 1–3:** 8GB RAM, any modern CPU (no GPU needed). Ollama runs DeepSeek-Coder 6.7B in ~4GB RAM.
* **Phase 4 fine-tuning:** 16GB VRAM GPU recommended. Google Colab free tier (T4, 16GB) works for small runs.
* **Storage:** 20GB free disk space for model weights, generated code, and proof corpus.
---
*— End of Document —*

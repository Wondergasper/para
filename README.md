# APG Phase 1 Backend

This folder supports **Phase 1: Foundation Pipeline** from
`APG_System_Architecture.docx`.

Phase 1 target:

```text
C source in -> analyse -> generate OpenMP candidate -> verify -> retry -> return result
```

There is no frontend in this build. The entry point is the Go orchestrator:

```powershell
go run ./cmd/orchestrator
```

The server exposes the Phase 1 API from the architecture document:

- `POST /submit` accepts C source code as the raw request body or multipart `file`
- `GET /result/{id}` returns job status, score, result code, and errors

## Setup

Install Phase 1 Python dependencies:

```powershell
pip install -r requirements/phase1.txt
```

Install and run Ollama, then pull a local coding model:

```powershell
ollama pull deepseek-coder:6.7b
```

## Submit A Job

Start the orchestrator:

```powershell
go run ./cmd/orchestrator
```

Submit a sample C file:

```powershell
curl.exe -X POST http://localhost:8080/submit --data-binary "@examples/vector_scale.c"
```

Then fetch the job by the returned `id`:

```powershell
curl.exe http://localhost:8080/result/<job_id>
```

## Current Phase 1 Status

Implemented:

- Go HTTP orchestrator
- `POST /submit`
- `GET /result/{id}`
- persistent job tracking in `data/jobs.json`
- Python pipeline CLI bridge via `scripts/run_pipeline.py`
- architecture-aligned Python worker packages under `workers/`
- generated C output-comparison harness calls for simple Phase 1 array functions

Still remaining from `APG_System_Architecture.docx` Phase 1:

- install GCC/OpenMP locally so Gate 1 and Gate 2 can run end-to-end
- improve Layer 1 with tree-sitter instead of LLM-only analysis
- complete race checking on Linux/macOS with ThreadSanitizer

## Current Phase 2 Status

Started from **Phase 2: Polyhedral Analysis + Multi-Candidate Generation** in
`APG_System_Architecture.docx`.

Implemented:

- AST-based polyhedral classifier using `pycparser` (replaces initial regex heuristic)
- robust affine expression detection for loop bounds and array indices
- indirect-index detection for irregular regions
- deterministic CTT candidate pool generation:
  - standard outer-loop parallelization
  - loop collapse (`collapse(2)`) for multi-dimensional nests
  - AST-based loop interchange
- pipeline fallback to the local classifier when LLM analysis is unavailable
- distributed task queuing with NATS JetStream
- gRPC/Protobuf contracts for cross-language job communication
- asynchronous results listener in the orchestrator

Still remaining from Phase 2:

- replace the manual AST classifier with `islpy` / ISL polyhedral checks (pending Windows build fix)
- add more CTT transforms such as tiling, interchange, and distribution
- run verification over candidates in parallel
- add structured JSON logging and correlation IDs

## Current Phase 3 Status

Started from **Phase 3: Formal Verification Bridge** in
`APG_System_Architecture.docx`.

Implemented:

- CBMC bounded-equivalence harness generation for simple array functions
- optional `gate_cbmc_model_check` verifier gate
- `PipelineConfig(enable_cbmc=True, cbmc_path="cbmc")`
- `scripts/run_pipeline.py --enable-cbmc --cbmc-path cbmc`
- clean missing-tool errors for GCC and CBMC
- local JSONL proof corpus store at `data/proofs.jsonl`
- `scripts/run_pipeline.py --proof-corpus data/proofs.jsonl`

Still remaining from Phase 3:

- install CBMC and validate the generated harnesses end-to-end
- add CBMC race-freedom checks, not only output-equivalence assertions
- add a Lean 4 subprocess manager with richer proof error parsing
- add a Verus subprocess path for Rust-transpiled candidates
- replace the local JSONL proof corpus with PostgreSQL

## Current Phase 4 Status

Started from **Phase 4: Training Signal & Fine-Tuning** in
`APG_System_Architecture.docx`.

Implemented:

- export local Phase 3 `data/proofs.jsonl` records to SFT JSONL
- `python -m workers.training.export_corpus --local-proof-corpus data/proofs.jsonl --output corpus.jsonl`
- verifier reward helpers matching the document weights:
  - compile: `+0.2`
  - output match: `+0.4`
  - race freedom: `+0.2`
  - formal proof: `+0.2`
- local JSONL reward event store at `data/rewards.jsonl`
- `scripts/run_pipeline.py --reward-events data/rewards.jsonl`
- model registry version selection with runtime routing based on version tag
- orchestrator A/B testing flag to route a percentage of traffic to fine-tuned adapters

Still remaining from Phase 4:

- collect at least 500 verified proof records
- run LoRA fine-tuning against the exported corpus
- run A/B evaluation with a real fine-tuned adapter

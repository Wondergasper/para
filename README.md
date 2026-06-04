# APG Phase 2 — Autonomous Parallel Code Generation
> A multi-layered pipeline for transforming sequential C/C++ source code into formally verified OpenMP parallel implementations with zero human intervention.

[![Phase 1: Complete](https://img.shields.io/badge/Phase%201-Complete-success?style=flat-square)](#)
[![Phase 2: Complete](https://img.shields.io/badge/Phase%202-Complete-success?style=flat-square)](#)
[![Phase 3: In Progress](https://img.shields.io/badge/Phase%203-In%20Progress-blue?style=flat-square)](#)
[![Phase 4: In Progress](https://img.shields.io/badge/Phase%204-In%20Progress-blue?style=flat-square)](#)

---

## 🛠️ System Overview

The system is split into two specialized sub-systems:
* **Go Orchestrator:** Manages the API endpoints, SQLite-based job store, and job routing over NATS JetStream.
* **Python Workers:** Houses the AST-based loop classifier, CTT transform engine, LLM code generation via Ollama, verification gates (GCC, TSAN, CBMC), and reinforcement learning feedback logic.

For a deep-dive architectural analysis, see [analysis_results.md](file:///C:/Users/USER/.gemini/antigravity/brain/c00a59fb-98f8-4ec1-ad32-5628e517d134/analysis_results.md).

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have the following installed on your host system:
* **Go 1.22+**
* **Python 3.11+**
* **GCC / Clang** (with OpenMP & ThreadSanitizer support)
* **NATS Server** (for queue-based worker execution)
* **Ollama** (for local LLM inference)

### 2. Environment Setup
Install the Python dependencies:
```powershell
pip install -r requirements/phase1.txt
```

Download and start the local LLM model:
```powershell
ollama pull deepseek-coder:6.7b
```

Ensure NATS Server is running (usually via Docker or service manager):
```powershell
docker run -d --name nats -p 4222:4222 -p 8222:8222 nats:latest -js
```

---

## 🏃 Execution Guide

### 1. Start the Python Worker
The worker listens to NATS and processes code parallelization jobs:
```powershell
python -m workers.nats_worker
```

### 2. Start the Go Orchestrator
In a separate terminal, start the HTTP API:
```powershell
go run ./cmd/orchestrator
```

### 3. Submit a Job
Submit a sequential C file to the pipeline:
```powershell
curl.exe -X POST http://localhost:8080/submit --data-binary "@examples/vector_scale.c"
```
Response will return a JSON body containing a unique `"id"`.

### 4. Fetch Results
Retrieve the job status, score, parallelized C code, and verification details:
```powershell
curl.exe http://localhost:8080/result/<job_id>
```

> [!TIP]
> You can also run the pipeline directly on a file without NATS:
> `python scripts/run_pipeline.py --source examples/vector_scale.c`

---

## 📊 Phase-by-Phase Development Status

### Phase 1: Foundation Pipeline (Complete)
* **Go HTTP Orchestrator:** standard endpoints `/submit` and `/result/{id}` are live.
* **Metadata Persistence:** Stores job metadata and configurations.
* **Verification Harness:** Generates dynamic comparison test files and evaluates them via GCC compilation and differential float analysis.

### Phase 2: Polyhedral Analysis & Candidate Pools (Complete)
* **AST Classifier:** Custom AST visitor using `pycparser` classifying loops into `polyhedral`, `irregular`, and `sequential`.
* **CTT Transforms:** Deterministic transformations for Outer-loop Parallelization, Loop Interchange, and Loop Collapse (`collapse(2)`).
* **Parallel Verification:** Verification gates compile and run candidate checks in parallel using Python thread pools.
* **Queued Execution:** Decoupled task queue using NATS JetStream and serialized with gRPC/Protobuf contracts.

### Phase 3: Formal Verification Bridge (In Progress)
* **CBMC Model Checking:** Generates bounded equivalence assertions checking candidate code correctness.
* **Proof Corpus Store:** Stores verified execution/harness details in `data/proofs.jsonl`.
* **Lean 4 Spec Gen:** Synthesizes Lean 4 equivalence theorems.
* *Remaining:* Realizing native execution runtime for Lean 4/Verus proof checks.

### Phase 4: SFT Signal & A/B Routing (In Progress)
* **Reward Engine:** Calculates reward feedback score (weights: compile `+0.2`, output match `+0.4`, race `+0.2`, proof `+0.2`).
* **Adapter Routing:** A/B testing framework routes configurable traffic percentages to fine-tuned adapters.
* *Remaining:* Accumulating 500 verified proof triples and executing LoRA fine-tuning scripts.

<div align="center">

# ⚡ APG — Autonomous Parallel Code Generation

**Submit serial code. Get verified parallel code back. Automatically.**

[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-22c55e?style=flat-square&logo=checkmarx)](#)
[![Phase 2](https://img.shields.io/badge/Phase%202-Complete-22c55e?style=flat-square&logo=checkmarx)](#)
[![Phase 3](https://img.shields.io/badge/Phase%203-Active-3b82f6?style=flat-square&logo=buffer)](#)
[![Phase 4](https://img.shields.io/badge/Phase%204-Active-3b82f6?style=flat-square&logo=buffer)](#)
[![Languages](https://img.shields.io/badge/Languages-C%20%7C%20Fortran%20%7C%20Python%20%7C%20Rust-a855f7?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](#)

<br/>

> APG is a self-improving, locally-running AI system that reads your sequential code,
> generates a correct parallel version using OpenMP / Rayon / Numba, formally verifies
> it compiles and produces matching output, and returns it ready to use —
> with **zero cloud dependency and zero human labelling**.

</div>

---

## 📑 Table of Contents

- [What It Does](#-what-it-does)
- [Architecture](#-architecture)
- [Supported Languages](#-supported-languages)
- [Quick Start](#-quick-start)
- [REST API](#-rest-api)
- [SDK Usage](#-sdk--developer-integration)
- [CLI Tools](#-cli-tools)
- [How the Pipeline Works](#-how-the-pipeline-works)
- [Self-Improvement (Phase 4)](#-self-improvement--phase-4)
- [Project Status](#-project-status)
- [What Is Missing / Roadmap](#-roadmap--whats-next)
- [Contributing](#-contributing)

---

## 🎯 What It Does

You give APG a slow, serial function. It:

1. **Classifies** it — is it safe to parallelise? (polyhedral, reduction, irregular, or sequential/unsafe)
2. **Generates** multiple parallel candidate versions using a local LLM + deterministic transforms
3. **Verifies** each candidate through a 4-gate pipeline (compile → output match → race detection → formal model check)
4. **Returns** the best verified version with a confidence score and speedup measurement
5. **Learns** — every verified result is saved to a corpus that eventually fine-tunes the underlying model

### Example

**Input** (sequential C):
```c
void scale_vector(float* A, float s, int N) {
    for (int i = 0; i < N; i++) A[i] *= s;
}
```

**Output** (verified parallel C):
```c
void scale_vector(float* A, float s, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) A[i] *= s;
}
// score: 0.8  gate: output_match  speedup: 3.7x (best at 4 threads)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User / SDK / CI                         │
│              HTTP POST /submit  ←→  GET /result/{id}        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Go Orchestrator  (cmd/orchestrator)            │
│  • REST API  • SQLite job store  • NATS JetStream queue     │
└────────────────────────┬────────────────────────────────────┘
                         │ NATS message
┌────────────────────────▼────────────────────────────────────┐
│              Python Worker  (workers/pipeline.py)           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  T1      │  │  T2      │  │  T3      │  │ Verifier  │  │
│  │ Classify │→ │ Generate │→ │ Critique │→ │ 4 Gates   │  │
│  │ (LLM)   │  │ (LLM+CTT)│  │ (LLM)   │  │           │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────┬─────┘  │
│                                                   │        │
│  ┌────────────────────────────────────────────────▼──────┐ │
│  │  Language Driver  (workers/lang/)                     │ │
│  │  C  │  Fortran  │  Python/Numba  │  Rust/Rayon        │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ verified result
┌────────────────────────▼────────────────────────────────────┐
│              Proof Corpus  (data/proofs.jsonl)              │
│              → Phase 4 LoRA fine-tuning trigger             │
└─────────────────────────────────────────────────────────────┘
```

**Two sub-systems, cleanly separated:**

| Sub-system | Language | Responsibility |
|---|---|---|
| **Orchestrator** | Go | HTTP API, job queue, SQLite persistence, NATS routing |
| **Workers** | Python | Classification, generation, verification, training |

---

## 🌐 Supported Languages

| Language | Parallelism | Gates | Thread Sweep | Status |
|---|---|---|---|---|
| **C** | OpenMP `#pragma omp` | compile → output → TSAN → CBMC | ✅ 1/2/4/8 threads | ✅ Full |
| **Fortran** | OpenMP `!$omp` | compile → output → TSAN | ✅ 1/2/4/8 threads | ✅ Full |
| **Python** | Numba `@jit(parallel=True, prange)` | compile → output | ✅ NUMBA_NUM_THREADS sweep | ✅ Full |
| **Rust** | Rayon `par_iter` | compile → output | ✅ RAYON_NUM_THREADS sweep | ✅ Full |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
go 1.22+
python 3.11+
gcc (with OpenMP)        # sudo apt install gcc  /  choco install mingw
ollama                   # https://ollama.com

# Optional but recommended
nats-server              # docker run -d -p 4222:4222 nats:latest -js
cbmc                     # sudo apt install cbmc
gfortran                 # sudo apt install gfortran
rustc + cargo            # https://rustup.rs
```

### 1-minute setup

```bash
# Clone
git clone https://github.com/yourname/apg && cd apg

# Python deps
pip install -r requirements/phase1.txt

# Pull the LLM model
ollama pull deepseek-coder:6.7b

# Start NATS (Docker)
docker run -d --name nats -p 4222:4222 nats:latest -js

# Check everything is ready
python scripts/check_env.py
```

### Run APG

```bash
# Terminal 1 — Python worker
python -m workers.nats_worker

# Terminal 2 — Go orchestrator
go run ./cmd/orchestrator

# Terminal 3 — submit a job
curl -X POST http://localhost:8080/submit \
  -H "Content-Type: application/json" \
  -d '{"source": "void scale(float* A, float s, int N) { for(int i=0;i<N;i++) A[i]*=s; }", "language": "c"}'

# {"id": "abc-123", "status": "pending"}

curl http://localhost:8080/result/abc-123
# {"status":"done","score":0.8,"parallel_code":"...","speedup":3.7}
```

> **Tip:** Run without NATS: `python scripts/run_pipeline.py --source examples/vector_scale.c`

---

## 🔌 REST API

Base URL: `http://localhost:8080`

### `POST /submit`

Submit a function for parallelisation.

**Request body:**
```json
{
  "source":    "void my_func(float* A, int N) { ... }",
  "language":  "c",
  "func_name": "my_func"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `source` | string | ✅ | The source code of a **single function** (no `main()`) |
| `language` | string | ✅ | `"c"`, `"fortran"`, `"python"`, `"rust"` |
| `func_name` | string | ❌ | Auto-detected if omitted |

**Response:**
```json
{ "id": "550e8400-e29b-41d4-a716-446655440000", "status": "pending" }
```

**Error codes:**
| Code | Meaning |
|---|---|
| `422` | Submitted a full program (has `main()`), or source > 10,000 chars |
| `400` | Missing or empty source |
| `500` | Internal pipeline error |

---

### `GET /result/{id}`

Poll for job result.

**Response (pending):**
```json
{ "id": "...", "status": "pending" }
```

**Response (done):**
```json
{
  "id": "550e8400...",
  "status": "done",
  "score": 0.8,
  "gate_passed": "output",
  "parallel_code": "void my_func(...) { #pragma omp parallel for\n  for(...) ... }",
  "speedup": 3.7,
  "best_threads": 4,
  "annotated_ir": {
    "type": "polyhedral",
    "reduction_variables": [],
    "warnings": []
  },
  "details": {
    "autotuning": {
      "runs": [
        {"threads": 1, "speedup": 1.0},
        {"threads": 2, "speedup": 1.9},
        {"threads": 4, "speedup": 3.7},
        {"threads": 8, "speedup": 3.5}
      ],
      "best_threads": 4,
      "best_speedup": 3.7
    }
  }
}
```

| Field | Meaning |
|---|---|
| `score` | 0.0 – 1.0. Gate weights: compile=0.2, output=0.6, race=0.8, proof=1.0 |
| `gate_passed` | Highest verification gate cleared: `compile`, `output`, `race`, `cbmc`, `proof` |
| `speedup` | Best measured speedup vs serial baseline |
| `best_threads` | Thread count that gave best speedup |

---

### `GET /health`

Returns `200 OK` when the orchestrator is running.

---

## 🧰 SDK & Developer Integration

APG is designed to be used as a service SDK. There is **no language-specific SDK package yet** — but it works cleanly with any HTTP client. Here is how to integrate it into your tools:

### Python SDK wrapper (copy-paste ready)

```python
# apg_client.py  — drop this into your project
import time
import requests

class APGClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base = base_url

    def parallelise(self, source: str, language: str = "c",
                    func_name: str = None, timeout: int = 120) -> dict:
        """
        Submit a function and block until result is ready.
        Returns the full result dict, or raises on failure.
        """
        payload = {"source": source, "language": language}
        if func_name:
            payload["func_name"] = func_name

        resp = requests.post(f"{self.base}/submit", json=payload, timeout=10)
        resp.raise_for_status()
        job_id = resp.json()["id"]

        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(f"{self.base}/result/{job_id}", timeout=10)
            r.raise_for_status()
            body = r.json()
            if body["status"] != "pending":
                return body
            time.sleep(3)

        raise TimeoutError(f"APG job {job_id} did not complete within {timeout}s")
```

**Usage:**
```python
from apg_client import APGClient

apg = APGClient("http://localhost:8080")

result = apg.parallelise("""
void dot_product(float* A, float* B, float* out, int N) {
    float s = 0;
    for (int i = 0; i < N; i++) s += A[i] * B[i];
    *out = s;
}
""", language="c")

if result["score"] >= 0.8:
    print("✓ Parallelised successfully!")
    print(f"  Speedup: {result['speedup']:.1f}x on {result['best_threads']} threads")
    print(result["parallel_code"])
else:
    print(f"✗ Could not parallelise — gate reached: {result['gate_passed']}")
```

---

### JavaScript / Node.js

```javascript
const APG = {
  async parallelise(source, language = 'c', baseUrl = 'http://localhost:8080') {
    const { id } = await fetch(`${baseUrl}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, language })
    }).then(r => r.json());

    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 3000));
      const result = await fetch(`${baseUrl}/result/${id}`).then(r => r.json());
      if (result.status !== 'pending') return result;
    }
    throw new Error('APG timeout');
  }
};

// Usage
const result = await APG.parallelise(`
  void scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) A[i] *= s;
  }
`);
console.log(result.parallel_code);
```

---

### Shell / CI pipeline

```bash
#!/bin/bash
# apg_submit.sh — use in CI pipelines

APG_URL="${APG_URL:-http://localhost:8080}"
SOURCE="$1"
LANG="${2:-c}"

ID=$(curl -sf -X POST "$APG_URL/submit" \
  -H "Content-Type: application/json" \
  -d "{\"source\": $(jq -Rs . < "$SOURCE"), \"language\": \"$LANG\"}" \
  | jq -r .id)

echo "Job: $ID"

for i in $(seq 1 40); do
  sleep 3
  RESULT=$(curl -sf "$APG_URL/result/$ID")
  STATUS=$(echo "$RESULT" | jq -r .status)
  if [ "$STATUS" != "pending" ]; then
    echo "$RESULT" | jq .
    exit 0
  fi
done

echo "Timeout" && exit 1
```

```bash
# In your CI/CD
./apg_submit.sh src/kernels.c c
```

---

### Docker integration (run APG as a sidecar)

```yaml
# docker-compose.yml
services:
  apg-orchestrator:
    build: .
    ports:
      - "8080:8080"
    depends_on: [nats, apg-worker]

  apg-worker:
    build: .
    command: python -m workers.nats_worker
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on: [nats, ollama]

  nats:
    image: nats:latest
    command: -js
    ports: ["4222:4222"]

  ollama:
    image: ollama/ollama
    volumes: ["ollama_data:/root/.ollama"]
    ports: ["11434:11434"]
```

---

## 🛠️ CLI Tools

All scripts live in [`scripts/`](file:///c:/Users/USER/Desktop/para/scripts/).

| Script | Purpose | Example |
|---|---|---|
| `run_pipeline.py` | Run pipeline on a local file (no NATS) | `python scripts/run_pipeline.py --source examples/vector_scale.c` |
| `batch_submit.py` | Submit all files in a directory | `python scripts/batch_submit.py --dir src/kernels/` |
| `corpus_report.py` | Show corpus health & Phase 4 readiness | `python scripts/corpus_report.py` |
| `clean_corpus.py` | Remove failed/duplicate corpus entries | `python scripts/clean_corpus.py --in-place` |
| `bootstrap_corpus.py` | Download PolybenchC and submit all 30 kernels | `python scripts/bootstrap_corpus.py --batch-size 4` |
| `generate_synthetic.py` | Generate 200 synthetic C functions and submit | `python scripts/generate_synthetic.py --count 200` |
| `auto_finetune.py` | Trigger LoRA fine-tuning when corpus is ready | `python scripts/auto_finetune.py --watch` |
| `check_env.py` | Check all dependencies are installed | `python scripts/check_env.py` |

---

## ⚙️ How the Pipeline Works

Each submitted job passes through up to 7 stages:

```
Stage 1 — INPUT GUARD
  Rejects: main() programs, files > 10K chars

Stage 2 — T1: CLASSIFY  (LLM)
  Outputs: polyhedral / reduction / irregular / sequential
  If sequential → returned immediately with explanation

Stage 3 — T2: GENERATE  (LLM + CTT deterministic transforms)
  Produces: pool of 3–5 parallel candidate versions
  Temperature escalates per retry round (0.2 → 0.35 → 0.5)

Stage 4 — VERIFICATION GATES  (per language driver)
  Gate 1: Compile check (gcc / gfortran / rustc / python ast)
  Gate 2: Output match (run both, compare float outputs < 1e-4 diff)
            + Thread autotuning sweep [1, 2, 4, 8 threads]
  Gate 3: Race detection (ThreadSanitizer on Linux, CBMC on Windows)
  Gate 4: Formal proof (Lean 4 / CBMC bounded model check)

Stage 5 — T3: CRITIQUE  (LLM)
  If all candidates fail gates, LLM critiques the error and retries
  Up to 3 rounds of critique-retry

Stage 6 — SCORE & RETURN
  Best candidate by score is returned
  Score: compile=0.2, output=0.6, race=0.8, proof=1.0

Stage 7 — CORPUS SAVE
  Result saved to data/proofs.jsonl
  When 500+ verified entries exist → auto_finetune.py fires Phase 4
```

---

## 🧠 Self-Improvement — Phase 4

APG gets smarter over time without any human involvement:

```
More code submitted
        ↓
Corpus grows (data/proofs.jsonl)
        ↓
500 verified (serial → parallel) pairs reached
        ↓
LoRA fine-tuning runs on the local model
        ↓
A/B test: new adapter vs baseline
        ↓
If new model wins → promoted as active model
        ↓
Future parallelisations are better quality
```

**Check your corpus status at any time:**
```bash
python scripts/corpus_report.py
```
```
Phase 4 LoRA threshold: 500 verified entries
Current verified:        7
STATUS: Need 493 more entries before LoRA trigger.
        Run: python scripts/bootstrap_corpus.py
```

---

## 📊 Project Status

| Phase | What | Status |
|---|---|---|
| **Phase 1** | Go API, SQLite, NATS queue, basic C pipeline | ✅ Complete |
| **Phase 2** | Polyhedral classifier, CTT transforms, multi-candidate pool, gRPC | ✅ Complete |
| **Phase 3** | CBMC gate wired, Fortran/Python/Rust drivers, TSAN, thread autotuning | ✅ Active |
| **Phase 4** | Corpus bootstrapper, auto-finetune trigger, LoRA/RL pipeline | ✅ Active (needs corpus) |

**Language support matrix:**

| | C | Fortran | Python | Rust |
|---|---|---|---|---|
| Classify | ✅ | ✅ | ✅ | ✅ |
| Generate OpenMP/parallel | ✅ | ✅ | ✅ (Numba) | ✅ (Rayon) |
| Compile gate | ✅ | ✅ | ✅ | ✅ |
| Output match gate | ✅ | ✅ | ✅ | ✅ |
| Thread sweep | ✅ | ✅ | ✅ | ✅ |
| Race detection | ✅ TSAN + CBMC | ✅ TSAN (Linux) | ⚠️ Soft check | ⚠️ Soft check |
| Formal proof gate | ⚡ CBMC | ❌ | ❌ | ❌ |

---

## 🗺️ Roadmap — What's Next

These items are scoped and designed, just not built yet:

- [ ] **Lean 4 formal proof runtime** — needs Lake + Mathlib project setup (A2)
- [ ] **Fortran polyhedral classifier** — local fallback without LLM call (C2)
- [ ] **Python TSAN equivalent** — stronger race check via deterministic output comparison (C4)
- [ ] **RL loop wiring** — GRPO reward training using `trl` library (B3)
- [ ] **Official Python SDK package** — `pip install apg-client`
- [ ] **VS Code extension** — right-click → "Parallelise this function"
- [ ] **GitHub Action** — `apg-action@v1` for CI parallelisation gate
- [ ] **Web dashboard** — live job monitor (scaffolded in `dashboard/`)
- [ ] **CUDA backend** — GPU kernel generation (T2 CUDA prompt already exists)

---

## 📁 Project Structure

```
apg/
├── cmd/orchestrator/       # Go HTTP server entrypoint
├── internal/apg/           # Go API handlers, job store, NATS client
├── workers/
│   ├── pipeline.py         # Main pipeline orchestrator
│   ├── lang/               # Language drivers (C, Fortran, Python, Rust)
│   ├── code_understanding/ # Polyhedral classifier, loop safety checker
│   ├── gen_openmp/         # T2 generation, CTT transforms
│   ├── spec_generator/     # T1 classify, T3 critique, verifier, reward
│   └── training/           # LoRA train, export corpus, A/B test
├── scripts/                # CLI tools (corpus, bootstrap, finetune)
├── examples/               # C, Fortran, Python, Rust example functions
├── data/                   # proofs.jsonl, benchmark_suite.jsonl
├── tests/python/           # pytest test suite
├── proto/                  # gRPC/Protobuf contracts
└── requirements/           # Python dependencies by phase
```

---

## 🤝 Contributing

APG is structured to make adding new language drivers easy:

1. Create `workers/lang/yourlang_driver.py` extending `LanguageDriver`
2. Implement: `detect_func_name`, `extract_functions`, `analyze_loops`, `verify_candidate`, `get_t1_system_prompt`, `get_t2_system_prompt`, `get_t2_prompt_hint`
3. Register in `workers/lang/__init__.py`
4. Add examples in `examples/yourlang/`
5. Add test class in `tests/python/test_multi_language.py`

---

## 📄 License

MIT — use freely, commercially or otherwise.

---

<div align="center">
<sub>Built with Go · Python · Ollama · OpenMP · CBMC · Rayon · Numba</sub>
</div>

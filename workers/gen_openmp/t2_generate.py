"""
prompts/t2_generate.py
----------------------
Prompt Template T2 — Parallel Code Generation (Layer 2)

What it does:
    Takes a C function + its AnnotatedIR from T1.
    Generates an OpenMP-parallelized version of the function.
    On retry, accepts an error trace so the model can fix the previous attempt.

Key rules baked into the system prompt:
    - Output ONLY valid C code — no explanations, no markdown
    - Every write to shared memory must be protected (atomic/critical/reduction)
    - Assume MAY_ALIAS on all pointer arguments unless explicitly proven otherwise
    - No dynamic allocation inside parallel regions (no malloc/new inside #pragma omp)
    - For reductions, use the OpenMP reduction() clause instead of manual atomics
    - Private loop variables must be declared private or be declared inside the region
"""

import json

from workers.common.llm_client import generate, generate_batch
from workers.gen_openmp.ctt_transform import generate_ctt_candidates


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_T2 = """You are an expert HPC programmer specialised in OpenMP parallelisation of C code.
Given a C function and its dependency graph, produce a correct OpenMP-parallelised version.

OUTPUT RULES (follow exactly):
1. Output ONLY valid C code. No explanations. No markdown fences. No comments outside the code.
2. The output must be a complete, compilable C function.
3. Include necessary #include directives if the original has them.

PARALLELISATION RULES (strict):
- Every write to a shared array must be atomic, inside a critical section, or use a reduction clause.
- Assume MAY_ALIAS for all pointer arguments: do NOT remove restrict unless it was in the original.
- Do NOT use malloc, calloc, or new inside a parallel region.
- For reduction operations (sum, product, min, max), use: #pragma omp parallel for reduction(op:var)
- Loop variables declared outside the parallel region must be listed as private(var).
- Loop variables declared inside the parallel region are implicitly private — no need to list them.
- For polyhedral loops with no dependencies: use #pragma omp parallel for
- For polyhedral loops with reduction only: use #pragma omp parallel for reduction(+:sum)
- For irregular (indirect index) loops: use #pragma omp atomic or critical as needed

COMMON MISTAKES TO AVOID:
- Do NOT parallelise a loop that has a RAW dependency on the loop variable (e.g. A[i] depends on A[i-1])
- Do NOT assume array arguments are non-aliasing unless the original has __restrict__ or restrict
- Do NOT add omp_set_num_threads() — let the caller control thread count via OMP_NUM_THREADS

EXAMPLE INPUT / OUTPUT:
Input function:
    void vscale(float* A, float scalar, int N) {
        for (int i = 0; i < N; i++) A[i] *= scalar;
    }
Input dep graph: {"RAW":[],"WAR":[],"WAW":[]}

Output:
    void vscale(float* A, float scalar, int N) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) A[i] *= scalar;
    }"""


SYSTEM_PROMPT_CUDA = """You are an expert HPC programmer specialised in CUDA parallelisation of C/C++ code.
Given a C function and its dependency graph, produce a correct CUDA-parallelised version.

OUTPUT RULES (follow exactly):
1. Output ONLY valid C/C++ CUDA code. No explanations. No markdown fences. No comments outside the code.
2. The output must contain:
   - A `__global__` CUDA kernel function that performs the actual computation.
   - A host wrapper function that matches the original function signature exactly.
3. The host wrapper must handle:
   - Allocating device memory via `cudaMalloc`.
   - Copying inputs to device via `cudaMemcpy`.
   - Launching the CUDA kernel with appropriate grid and block dimensions.
   - Checking for launch errors or synchronizing via `cudaDeviceSynchronize()`.
   - Copying results back to host via `cudaMemcpy`.
   - Freeing device memory via `cudaFree`.

EXAMPLE INPUT / OUTPUT:
Input function:
    void vscale(float* A, float scalar, int N) {
        for (int i = 0; i < N; i++) A[i] *= scalar;
    }
Input dep graph: {"RAW":[],"WAR":[],"WAW":[]}

Output:
    __global__ void vscale_kernel(float* A, float scalar, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            A[i] *= scalar;
        }
    }

    void vscale(float* A, float scalar, int N) {
        float* d_A;
        cudaMalloc(&d_A, N * sizeof(float));
        cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, scalar, N);
        cudaDeviceSynchronize();
        cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_A);
    }"""


SYSTEM_PROMPT_OPENMP_TARGET = """You are an expert HPC programmer specialised in OpenMP target offloading for C code.
Given a C function and its dependency graph, produce a correct OpenMP-target parallelised version.

OUTPUT RULES (follow exactly):
1. Output ONLY valid C code. No explanations. No markdown fences. No comments outside the code.
2. The output must be a complete, compilable C function.
3. Use standard OpenMP target offload pragmas (e.g. `#pragma omp target teams distribute parallel for map(...)`).
4. Make sure to specify the `map` clause with correct map types:
   - `map(to: ...)` for read-only variables/arrays.
   - `map(from: ...)` for write-only variables/arrays.
   - `map(tofrom: ...)` for read-write variables/arrays.

EXAMPLE INPUT / OUTPUT:
Input function:
    void vscale(float* A, float scalar, int N) {
        for (int i = 0; i < N; i++) A[i] *= scalar;
    }
Input dep graph: {"RAW":[],"WAR":[],"WAW":[]}

Output:
    void vscale(float* A, float scalar, int N) {
        #pragma omp target teams distribute parallel for map(tofrom: A[0:N])
        for (int i = 0; i < N; i++) {
            A[i] *= scalar;
        }
    }"""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_t2_prompt(
    source_code:  str,
    dep_graph:    dict,
    error_trace:  str = "",
    annotated_ir: dict = None,
) -> str:
    """
    Build the T2 prompt from source code and dependency info.

    Args:
        source_code:  The original C function.
        dep_graph:    Dependency graph dict from T1 (keys: RAW, WAR, WAW).
        error_trace:  Error from the previous attempt (empty on first try).
        annotated_ir: Full AnnotatedIR from T1 (optional, for extra context).

    Returns:
        Formatted prompt string.
    """
    lines = [f"Function to parallelise:\n{source_code.strip()}"]
    lines.append(f"\nDependency graph:\n{json.dumps(dep_graph, indent=2)}")

    # Include reduction variables hint if present
    if annotated_ir and annotated_ir.get("reduction_variables"):
        rvars = annotated_ir["reduction_variables"]
        lines.append(f"\nReduction variables detected: {rvars}")
        lines.append("Use the OpenMP reduction() clause for these variables.")

    # Include parallelism type hint
    if annotated_ir and annotated_ir.get("type") == "irregular":
        lines.append(
            "\nNOTE: This function has irregular (indirect) array accesses. "
            "Use #pragma omp atomic or critical for shared writes."
        )

    # Include previous error for retry
    if error_trace:
        lines.append(
            f"\nPREVIOUS ATTEMPT FAILED. Error was:\n{error_trace.strip()}\n"
            "Fix the issue described above. Do NOT repeat the same mistake."
        )

    lines.append("\nGenerate the parallelised C function:")
    return "\n".join(lines)


# ── Main generation function ───────────────────────────────────────────────────

def generate_parallel(
    source_code:  str,
    dep_graph:    dict,
    annotated_ir: dict = None,
    error_trace:  str  = "",
    provider:     str  = "ollama",
    model:        str  = None,
    max_retries:  int  = 3,
    target:       str  = "openmp",
) -> list[str]:
    """
    Generate OpenMP or GPU parallelised C/CUDA code using batch LLM inference.
    """
    prompt = build_t2_prompt(
        source_code, dep_graph,
        error_trace=error_trace,
        annotated_ir=annotated_ir,
    )

    if target == "cuda":
        system_prompt = SYSTEM_PROMPT_CUDA
    elif target == "openmp-target":
        system_prompt = SYSTEM_PROMPT_OPENMP_TARGET
    else:
        system_prompt = SYSTEM_PROMPT_T2

    # Call generate_batch once to query multiple candidates simultaneously
    raw_responses = generate_batch(
        prompt=prompt,
        system=system_prompt,
        provider=provider,
        model=model,
        temp=0.2,
        n=max_retries,
    )

    candidates = []
    for raw in raw_responses:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            cleaned = "\n".join(lines[1:end_idx])

        # Basic sanity check: does the output look like C or CUDA code?
        if any(keyword in cleaned for keyword in ("for", "while", "pragma", "kernel", "cuda", "{")):
            candidates.append(cleaned)

    if not candidates and raw_responses:
        # Fallback to returning the raw response if none passed sanity checks
        candidates.append(raw_responses[0])

    return candidates


def generate_candidate_pool(
    source_code: str,
    dep_graph: dict,
    annotated_ir: dict = None,
    error_trace: str = "",
    provider: str = "ollama",
    model: str = None,
    max_llm_candidates: int = 3,
    llm_generator=generate_parallel,
    target: str = "openmp",
) -> list[str]:
    """
    Phase 2 candidate pool: deterministic CTT candidates first, LLM candidates next.
    """
    candidates = []
    if target == "openmp":
        ctt_candidates = generate_ctt_candidates(source_code, annotated_ir or {})
        for cand in ctt_candidates:
            if cand not in candidates:
                candidates.append(cand)

    llm_candidates = []
    if max_llm_candidates > 0:
        llm_candidates = llm_generator(
            source_code=source_code,
            dep_graph=dep_graph,
            annotated_ir=annotated_ir,
            error_trace=error_trace,
            provider=provider,
            model=model,
            max_retries=max_llm_candidates,
            target=target,
        )
    for candidate in llm_candidates:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simple polyhedral case — vector addition
    source = """void vector_add(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}"""

    dep_graph = {"RAW": [], "WAR": [], "WAW": []}
    annotated_ir = {
        "type": "polyhedral",
        "summary": "Element-wise addition of two vectors.",
        "affine_dimensions": ["i"],
        "dependencies": dep_graph,
        "reduction_variables": [],
    }

    print("Input:")
    print(source)
    print("\nGenerating parallel version...")

    candidates = generate_parallel(
        source_code=source,
        dep_graph=dep_graph,
        annotated_ir=annotated_ir,
        max_retries=1,
    )

    print(f"\nGot {len(candidates)} candidate(s):")
    for i, c in enumerate(candidates, 1):
        print(f"\n--- Candidate {i} ---")
        print(c)

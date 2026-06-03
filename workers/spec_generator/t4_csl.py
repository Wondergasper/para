"""
prompts/t4_csl.py
-----------------
Prompt Template T4 — CSL Spec / Formal Proof Generation (Phase 3, Layer 3 Gate 4)

What it does:
    Takes sequential C code + parallel C code + dependency graph.
    Generates a Lean 4 theorem asserting:
        1. Functional equivalence (parallel output == sequential output for all inputs)
        2. Race freedom (no two threads write the same memory location simultaneously)

When it is called:
    Only in Phase 3, after Gates 1–3 have already passed (compile, output match,
    ThreadSanitizer clean). Gate 4 is the optional formal proof gate.

Note on practical use:
    Lean 4 proof generation by a 6–7B model is aspirational — treat it as a
    best-effort spec that a human verifier or CBMC can validate. The system
    still works without a passing Gate 4; it just scores 0.8 instead of 1.0.
"""

import json

from workers.common.llm_client import generate


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_T4 = """You are a formal verification expert specialised in Lean 4 and parallel program correctness.
Given a sequential C function, its parallel OpenMP equivalent, and the dependency graph,
write the Lean 4 proof segments to verify functional equivalence and race freedom.

OUTPUT RULES (follow exactly):
1. Output ONLY valid JSON. No markdown. No explanations.
2. The JSON must match this schema:
   {
     "seq_model_body": "<Lean 4 sequential model body expression, e.g. List.zipWith (· + ·) A B>",
     "par_model_body": "<Lean 4 parallel model body expression, e.g. seqVectorAdd A B N>",
     "equiv_proof": "<Lean 4 equivalence proof tactic block, e.g. rfl or simp>",
     "race_free_predicate": "<Lean 4 race free predicate, e.g. ¬ (i < N ∧ j < N ∧ i = j)>",
     "race_free_proof": "<Lean 4 race free proof tactic block, e.g. intro h; exact ...>"
   }

Ensure all Lean 4 expressions and tactic steps are syntactically valid in Lean 4."""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_t4_prompt(
    sequential_code: str,
    parallel_code:   str,
    dep_graph:       str,
    skeleton_spec:   str,
) -> str:
    """
    Build the T4 formal proof prompt.
    """
    lines = [
        "=== SEQUENTIAL C FUNCTION ===",
        sequential_code.strip(),
        "",
        "=== PARALLEL OPENMP FUNCTION ===",
        parallel_code.strip(),
        "",
        "=== SKELETON SPEC TEMPLATE ===",
        skeleton_spec.strip(),
        "",
        "Output a JSON object containing the values to fill the skeleton placeholders."
    ]
    return "\n".join(lines)


# ── Main CSL generation function ───────────────────────────────────────────────

def generate_proof(
    sequential_code: str,
    parallel_code:   str,
    dep_graph:       dict,
    annotated_ir:    dict = None,
    provider:        str  = "ollama",
    model:           str  = None,
) -> str:
    """
    Generate a Lean 4 formal proof for the parallel transformation.
    """
    from workers.spec_generator.spec_template import generate_lean_skeleton, render_lean_spec
    
    func_name = annotated_ir.get("func_name", "func") if annotated_ir else "func"
    skeleton, defaults = generate_lean_skeleton(func_name, annotated_ir or {})
    
    prompt = build_t4_prompt(sequential_code, parallel_code, json.dumps(dep_graph), skeleton)
    raw = generate(prompt, SYSTEM_PROMPT_T4, provider, model, temp=0.15, max_tokens=2048)

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end_idx])

    try:
        results = json.loads(cleaned)
        # Verify required keys
        for key in defaults.keys():
            if key not in results or not results[key].strip():
                results[key] = defaults[key]
    except Exception:
        # Fallback to default placeholders if LLM fails or doesn't return valid JSON
        results = defaults

    return render_lean_spec(skeleton, results)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seq = """void vector_add(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}"""

    par = """void vector_add(float* A, float* B, float* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}"""

    dep_graph    = {"RAW": [], "WAR": [], "WAW": []}
    annotated_ir = {
        "type": "polyhedral",
        "summary": "Element-wise addition of two vectors.",
        "affine_dimensions": ["i"],
        "dependencies": dep_graph,
        "reduction_variables": [],
    }

    print("Generating Lean 4 proof...")
    lean_code = generate_proof(seq, par, dep_graph, annotated_ir)
    print("\nLean 4 output:")
    print(lean_code)

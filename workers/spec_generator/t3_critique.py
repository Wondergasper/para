"""
prompts/t3_critique.py
----------------------
Prompt Template T3 — Failure Critique (Layer 4)

What it does:
    Called when a candidate from T2 fails one of the verifier gates.
    Takes: original code + failing candidate + error message.
    Returns: root cause diagnosis + a surgical fix (max 3 lines changed).

Error types handled:
    - TSAN_RACE       → ThreadSanitizer detected a data race → missing sync
    - REDUCTION_BUG   → race on scalar accumulator (sum +=) → add reduction() clause
    - OUTPUT_MISMATCH → parallel output differs from sequential reference
    - COMPILE_ERROR   → missing private/firstprivate clause, bad syntax
    - DEADLOCK        → incorrect use of critical sections
"""

import json

from workers.common.llm_client import generate


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_T3 = """You are a debugging oracle for parallel C/OpenMP code.
Given an original sequential function, a failing parallel candidate, and the error,
identify the root cause and provide a surgical fix.

OUTPUT RULES (follow exactly):
1. Output ONLY valid JSON. No markdown. No explanation outside the JSON.
2. The JSON must match this schema:
   {
     "error_type": "<TSAN_RACE|REDUCTION_BUG|OUTPUT_MISMATCH|COMPILE_ERROR|DEADLOCK|UNKNOWN>",
     "root_cause": "<one sentence explaining exactly why it fails>",
     "fix_description": "<one sentence describing the fix>",
     "patch": "<unified diff patch to apply to the candidate>"
   }

ERROR DIAGNOSIS RULES:
- TSAN_RACE       → error contains "data race", "ThreadSanitizer", "TSAN"
- REDUCTION_BUG   → error contains "mismatch" AND the original code has a scalar accumulator
                    (sum +=, total +=, count ++) that was NOT protected with reduction() clause.
                    Recognise this pattern: a scalar variable written with += inside a parallel loop.
- OUTPUT_MISMATCH → error contains "mismatch", "wrong output", "expected", "got"
- COMPILE_ERROR   → error contains "error:", "undefined", "undeclared", "expected ';'"
- DEADLOCK        → error contains "deadlock", "hang", "timeout"
- UNKNOWN         → none of the above match

FIX RULES (strict):
- The fix MUST change at most 3 lines from the candidate.
- Generate a standard unified diff patch (using ---, +++, @@ headers) to correct the candidate.
- For TSAN_RACE: add #pragma omp atomic OR change to reduction() clause.
- For REDUCTION_BUG: add the reduction() clause to the #pragma omp parallel for directive.
    Example: change "#pragma omp parallel for" to "#pragma omp parallel for reduction(+:sum)"
    Use reduction(*:var) for multiplication, reduction(-:var) for subtraction.
    The scalar variable must be declared and initialised OUTSIDE the parallel region.
    Do NOT declare the accumulator inside the parallel region.
- For OUTPUT_MISMATCH: check if a RAW dependency prevents parallelism — if so, remove the parallel pragma.
- For COMPILE_ERROR: add the missing clause (private, firstprivate, shared) or fix the syntax.
- For DEADLOCK: restructure critical section nesting.

The "patch" field must contain the unified diff format patch representation, e.g.:
--- candidate
+++ fixed
@@ -line,count +line,count @@
-old line
+new line"""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_t3_prompt(
    original_code:   str,
    candidate_code:  str,
    error_message:   str,
    gate_failed:     str = "",
) -> str:
    """
    Build the T3 debugging prompt.

    Args:
        original_code:  The original sequential C function.
        candidate_code: The failing parallel candidate from T2.
        error_message:  The error output from the verifier gate.
        gate_failed:    Which gate failed: "compile" | "output" | "race" | "proof"

    Returns:
        Formatted prompt string.
    """
    lines = [
        "=== ORIGINAL SEQUENTIAL FUNCTION ===",
        original_code.strip(),
        "",
        "=== FAILING PARALLEL CANDIDATE ===",
        candidate_code.strip(),
        "",
        "=== ERROR MESSAGE ===",
        error_message.strip(),
    ]

    if gate_failed:
        gate_hints = {
            "compile": "The code failed to compile. Look for missing private clauses or syntax errors.",
            "output":  "The output does not match the sequential reference. A data dependency may have been violated.",
            "race":    "ThreadSanitizer detected a data race. A shared variable is written without synchronisation.",
            "proof":   "The formal proof could not be established. The transformation may not be semantically equivalent.",
        }
        hint = gate_hints.get(gate_failed, "")
        if hint:
            lines += ["", f"=== GATE FAILED: {gate_failed.upper()} ===", hint]

    lines += ["", "Diagnose the root cause and output the fix JSON:"]
    return "\n".join(lines)


# ── Main critique function ─────────────────────────────────────────────────────

def critique(
    original_code:  str,
    candidate_code: str,
    error_message:  str,
    gate_failed:    str = "",
    provider:       str = "ollama",
    model:          str = None,
) -> dict:
    """
    Run T3 critique on a failing candidate.

    Args:
        original_code:  Original sequential C function.
        candidate_code: The failing parallel candidate.
        error_message:  Error from the verifier gate.
        gate_failed:    Which gate failed ("compile"|"output"|"race"|"proof").
        provider:       LLM provider.
        model:          Model override.

    Returns:
        Dict with keys: error_type, root_cause, fix_description, fixed_code.

    Raises:
        ValueError: If the LLM response is not valid JSON.
    """
    prompt = build_t3_prompt(original_code, candidate_code, error_message, gate_failed)
    raw    = generate(prompt, SYSTEM_PROMPT_T3, provider, model, temp=0.1)

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end_idx])

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"T3 returned invalid JSON.\nRaw response:\n{raw}\nJSON error: {e}"
        )

    required = {"error_type", "root_cause", "fix_description", "patch"}
    missing  = required - set(result.keys())
    if missing:
        raise ValueError(f"T3 response missing required keys: {missing}")

    return result


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    original = """float sum_array(float* A, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += A[i];
    return sum;
}"""

    # A broken candidate that writes to sum without a reduction clause
    broken_candidate = """float sum_array(float* A, int N) {
    float sum = 0.0f;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) sum += A[i];
    return sum;
}"""

    error_msg = (
        "ThreadSanitizer: data race on variable 'sum' "
        "at sum_array.c:4. Write of size 4 by thread T2. "
        "Previous write by thread T1."
    )

    print("Running T3 critique...")
    result = critique(
        original_code=original,
        candidate_code=broken_candidate,
        error_message=error_msg,
        gate_failed="race",
    )

    print("\nT3 output:")
    print(json.dumps(result, indent=2))
    print("\nPatch generated:")
    print(result["patch"])
    
    from workers.common.patch_applier import apply_patch
    try:
        fixed = apply_patch(broken_candidate, result["patch"])
        print("\nPatched code successfully applied:")
        print(fixed)
    except Exception as e:
        print(f"\nFailed to apply patch: {e}")

"""
prompts/t1_analysis.py
----------------------
Prompt Template T1 — Region Annotation (Layer 1)

What it does:
    Takes a raw C function as input.
    Returns a structured JSON object (AnnotatedIR) that classifies
    the function's parallelism type and summarises its computation.

Output schema (AnnotatedIR):
    {
        "type": "polyhedral" | "irregular" | "sequential",
        "summary": "brief description of what the function computes",
        "affine_dimensions": ["i", "j"],     // only if type == "polyhedral"
        "dependencies": {
            "RAW": [["A[i]", "A[i-1]"]],    // Read-After-Write
            "WAR": [],                        // Write-After-Read
            "WAW": []                         // Write-After-Write
        },
        "reduction_variables": ["sum"]        // if any reductions are present
    }

Classification rules (baked into the system prompt):
    - If any array index is indirect (e.g. A[B[i]]) → type = "irregular"
    - If loop bounds are non-affine (runtime-computed) → type = "irregular"
    - If all bounds and indices are compile-time affine → type = "polyhedral"
    - If no loops exist or loop count == 1 with no parallelism → type = "sequential"
"""

import json

from workers.common.llm_client import generate


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_T1 = """You are a polyhedral analysis engine specialised in C code.
Given a C function, classify its parallelism type and summarise its computation.

OUTPUT RULES (follow exactly):
1. Output ONLY valid JSON. No explanation, no markdown, no code fences.
2. The JSON must match this schema exactly:
   {
     "type": "<polyhedral|irregular|sequential>",
     "summary": "<one sentence describing what the function computes>",
     "affine_dimensions": ["<loop var>", ...],
     "dependencies": {
       "RAW": [["<written>", "<read>"], ...],
       "WAR": [["<read>",    "<written>"], ...],
       "WAW": [["<w1>",      "<w2>"], ...]
     },
     "reduction_variables": ["<var>", ...]
   }

CLASSIFICATION RULES (strict):
- If ANY array index is indirect (e.g. A[B[i]], C[idx[j]]) → type MUST be "irregular"
- If ANY loop bound is a non-affine expression (pointer arithmetic, runtime value) → "irregular"
- If ALL loop bounds and ALL array indices are affine in the loop variables → "polyhedral"
- If there are no loops or loops cannot be parallelised → "sequential"
- "affine_dimensions" is ONLY included when type == "polyhedral"
- "reduction_variables" is an empty list if no reductions are present
- "dependencies" lists data dependencies that prevent naive parallelisation

EXAMPLES:
Input:  for(int i=1; i<N; i++) A[i] = A[i-1] + 1;
Output: {"type":"polyhedral","summary":"Increments each element by the previous.","affine_dimensions":["i"],"dependencies":{"RAW":[["A[i]","A[i-1]"]],"WAR":[],"WAW":[]},"reduction_variables":[]}

Input:  for(int i=0; i<N; i++) B[C[i]] += 1;
Output: {"type":"irregular","summary":"Histogram update with indirect index.","affine_dimensions":[],"dependencies":{"RAW":[],"WAR":[],"WAW":[["B[C[i]]","B[C[i]]"]]},"reduction_variables":[]}"""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_t1_prompt(source_code: str) -> str:
    """
    Wrap the source code in a clean prompt for T1.

    Args:
        source_code: The full C function as a string.

    Returns:
        The formatted prompt string to send to the LLM.
    """
    return f"Analyse this C function:\n\n{source_code.strip()}\n\nOutput AnnotatedIR JSON:"


# ── Main analysis function ─────────────────────────────────────────────────────

def analyse(
    source_code: str,
    provider: str = "ollama",
    model: str = None,
) -> dict:
    """
    Run T1 analysis on a C function.

    Args:
        source_code: The C function to analyse.
        provider:    LLM provider ("ollama" | "groq" | "gemini").
        model:       Model override (uses provider default if None).

    Returns:
        Parsed AnnotatedIR dict.

    Raises:
        ValueError:   If the LLM response is not valid JSON.
        RuntimeError: If the LLM call fails.
    """
    from workers.code_understanding.clang_analyzer import is_clang_available, analyse_with_clang
    if is_clang_available():
        try:
            return analyse_with_clang(source_code)
        except Exception:
            pass

    prompt   = build_t1_prompt(source_code)
    raw      = generate(prompt, SYSTEM_PROMPT_T1, provider, model, temp=0.1)

    # Strip markdown fences if the model wraps output despite instructions
    cleaned  = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"T1 returned invalid JSON.\n"
            f"Raw response:\n{raw}\n"
            f"JSON error: {e}"
        )

    # Validate required keys
    required = {"type", "summary", "dependencies", "reduction_variables"}
    missing  = required - set(result.keys())
    if missing:
        raise ValueError(f"T1 response missing required keys: {missing}")

    # Ensure type is one of the valid values
    valid_types = {"polyhedral", "irregular", "sequential"}
    if result["type"] not in valid_types:
        raise ValueError(
            f"T1 returned unknown type '{result['type']}'. "
            f"Expected one of {valid_types}"
        )

    # Add empty affine_dimensions if not present (non-polyhedral case)
    result.setdefault("affine_dimensions", [])

    return result


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # Polyhedral — simple loop
        """void vector_add(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}""",
        # Irregular — indirect indexing
        """void scatter(int* dst, int* src, int* idx, int N) {
    for (int i = 0; i < N; i++) {
        dst[idx[i]] = src[i];
    }
}""",
        # Polyhedral with reduction
        """float dot_product(float* A, float* B, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}""",
    ]

    for i, code in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test case {i}:")
        print(code)
        print(f"\nT1 output:")
        result = analyse(code)
        print(json.dumps(result, indent=2))

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


# ── Specialist Agent Prompts ───────────────────────────────────────────────────

SYSTEM_PROMPT_COMPILER_CRITIC = """You are a Compiler Specialist Critic for C/OpenMP code.
Your role is to analyze compilation failures and warnings.
Identify issues like:
- Missing private, firstprivate, or shared variable clauses.
- Loop variables declared in wrong scopes.
- Incorrect syntax or unsupported compiler options.
Provide a concise, technical explanation of what is wrong and how to fix it."""

SYSTEM_PROMPT_RACE_CRITIC = """You are a Memory Safety & TSAN Critic for parallel C/OpenMP code.
Your role is to analyze ThreadSanitizer (TSAN) reports and concurrency data races.
Identify issues like:
- Shared memory access conflicts (writes from multiple threads without synchronization).
- Missing atomic operations or critical sections.
- Variables that should be declared under an OpenMP reduction clause.
Provide a concise, technical explanation of the race condition and how to resolve it."""

SYSTEM_PROMPT_CORRECTNESS_CRITIC = """You are a Functional Correctness Critic for parallel C/OpenMP code.
Your role is to analyze logic bugs where the parallel program output differs from the sequential reference.
Identify issues like:
- Loop-carried data dependencies (RAW, WAR, WAW) that prevent safe parallelization.
- Incorrect loop boundaries, induction variables, or mapping in tiled/collapsed loop nests.
- Wrong reduction operators or initialization values.
Provide a concise, technical explanation of the logical discrepancy and how to resolve it."""

SYSTEM_PROMPT_SYNTHESIS = """You are a Patch Synthesizer.
Given the debate transcript containing a primary critic analysis and peer reviews,
synthesize the final consensus and surgical fix.

OUTPUT RULES (follow exactly):
1. Output ONLY valid JSON. No markdown. No explanation outside the JSON.
2. The JSON must match this schema:
   {
     "error_type": "<TSAN_RACE|REDUCTION_BUG|OUTPUT_MISMATCH|COMPILE_ERROR|DEADLOCK|UNKNOWN>",
     "root_cause": "<one sentence explaining exactly why it fails>",
     "fix_description": "<one sentence describing the fix>",
     "patch": "<unified diff patch to apply to the candidate>"
   }

FIX RULES (strict):
- The fix MUST change at most 3 lines from the candidate.
- Generate a standard unified diff patch (using ---, +++, @@ headers) to correct the candidate.
- The "patch" field must contain the unified diff format patch representation, e.g.:
--- candidate
+++ fixed
@@ -line,count +line,count @@
-old line
+new line"""

# Backwards compatibility reference
SYSTEM_PROMPT_T3 = SYSTEM_PROMPT_SYNTHESIS


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
    Run multi-agent critique debate on a failing candidate.
    """
    # 1. Determine roles based on failed gate
    if gate_failed == "compile":
        primary_critic = "Compiler Critic"
        primary_system = SYSTEM_PROMPT_COMPILER_CRITIC
        peers = [
            ("Memory Safety Critic", SYSTEM_PROMPT_RACE_CRITIC),
            ("Correctness Critic", SYSTEM_PROMPT_CORRECTNESS_CRITIC)
        ]
    elif gate_failed == "race":
        primary_critic = "Memory Safety Critic"
        primary_system = SYSTEM_PROMPT_RACE_CRITIC
        peers = [
            ("Compiler Critic", SYSTEM_PROMPT_COMPILER_CRITIC),
            ("Correctness Critic", SYSTEM_PROMPT_CORRECTNESS_CRITIC)
        ]
    else:  # output or proof
        primary_critic = "Correctness Critic"
        primary_system = SYSTEM_PROMPT_CORRECTNESS_CRITIC
        peers = [
            ("Compiler Critic", SYSTEM_PROMPT_COMPILER_CRITIC),
            ("Memory Safety Critic", SYSTEM_PROMPT_RACE_CRITIC)
        ]

    # 2. Query the Primary Critic
    primary_prompt = (
        f"=== ORIGINAL SEQUENTIAL FUNCTION ===\n{original_code.strip()}\n\n"
        f"=== FAILING PARALLEL CANDIDATE ===\n{candidate_code.strip()}\n\n"
        f"=== ERROR MESSAGE ===\n{error_message.strip()}\n\n"
        f"Please analyze the failure above as the {primary_critic}."
    )
    primary_analysis = generate(primary_prompt, primary_system, provider, model, temp=0.2)

    # 3. Query peer reviewers in parallel to keep latency low
    peer_reviews = {}

    def run_peer_review(name, peer_system):
        peer_prompt = (
            f"=== ORIGINAL SEQUENTIAL FUNCTION ===\n{original_code.strip()}\n\n"
            f"=== FAILING PARALLEL CANDIDATE ===\n{candidate_code.strip()}\n\n"
            f"=== ERROR MESSAGE ===\n{error_message.strip()}\n\n"
            f"=== PRIMARY CRITIC ({primary_critic}) ANALYSIS ===\n{primary_analysis.strip()}\n\n"
            f"As the {name}, review the primary critic's analysis. "
            f"Do you agree? Are there additional syntax, safety, or logic issues the primary critic missed?"
        )
        return generate(peer_prompt, peer_system, provider, model, temp=0.2)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_peer_review, name, peer_system): name
            for name, peer_system in peers
        }
        for fut in futures:
            name = futures[fut]
            try:
                peer_reviews[name] = fut.result()
            except Exception as e:
                peer_reviews[name] = f"Error during peer review call: {e}"

    # 4. Assemble the debate transcript
    debate_transcript = [
        f"=== PRIMARY CRITIC ({primary_critic}) ANALYSIS ===",
        primary_analysis.strip(),
        ""
    ]
    for name, review in peer_reviews.items():
        debate_transcript.extend([
            f"=== PEER REVIEW ({name}) ===",
            review.strip(),
            ""
        ])

    # 5. Synthesize consensus patch using the synthesis agent
    synthesis_prompt = (
        f"=== ORIGINAL SEQUENTIAL FUNCTION ===\n{original_code.strip()}\n\n"
        f"=== FAILING PARALLEL CANDIDATE ===\n{candidate_code.strip()}\n\n"
        f"=== ERROR MESSAGE ===\n{error_message.strip()}\n\n"
        f"=== DEBATE TRANSCRIPT ===\n" + "\n".join(debate_transcript) + "\n\n"
        f"Synthesize the consensus from the debate transcript and generate the patch JSON."
    )
    raw = generate(synthesis_prompt, SYSTEM_PROMPT_SYNTHESIS, provider, model, temp=0.1)

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
            f"T3 Synthesis returned invalid JSON.\nRaw response:\n{raw}\nJSON error: {e}"
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

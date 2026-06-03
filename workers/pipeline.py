"""
pipeline.py
-----------
Main APG Pipeline Orchestrator

Connects all layers in order:
    Layer 1 (T1) → Classify + extract dependency graph
    Layer 2 (T2) → Generate parallel candidates
    Layer 3 (Verifier) → Score each candidate through 4 gates
    Layer 4 (T3) → Critique and fix failing candidates

Entry point:
    from pipeline import run

    result = run(source_code="void foo(...) { ... }")
    print(result["best_candidate"])
    print(result["score"])
    print(result["annotated_ir"])

Pipeline config (all tuneable via PipelineConfig):
    - max_t2_candidates: how many parallel versions T2 generates per round
    - max_critique_rounds: how many T3 fix loops to attempt
    - enable_proof: whether to run Gate 4 (Lean 4, Phase 3+)
    - provider / model: which LLM backend to use
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from workers.code_understanding.polyhedral import classify_region
from workers.code_understanding.t1_analysis import analyse
from workers.gen_openmp.t2_generate import generate_candidate_pool
from workers.spec_generator.proof_corpus import ProofCorpusStore, ProofRecord
from workers.spec_generator.reward import reward_from_score
from workers.spec_generator.reward_events import RewardEvent, RewardEventStore
from workers.spec_generator.t3_critique import critique
from workers.spec_generator.t4_csl import generate_proof
from workers.spec_generator.verifier import verify_candidate, VerificationResult


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    provider:            str   = "ollama"
    model:               str   = None         # None = use provider default
    model_version:       str   = "base"       # Model registry version
    max_t2_candidates:   int   = 3            # T2 generation attempts per round
    max_critique_rounds: int   = 3            # T3 fix loop iterations
    enable_proof:        bool  = False        # Gate 4 (Phase 3+ only)
    enable_tsan:         bool  = True         # Gate 3 (set False on Windows)
    enable_cbmc:         bool  = False        # Phase 3 CBMC bounded model check
    cbmc_path:           str   = "cbmc"       # CBMC binary path
    score_threshold:     float = 0.8          # Minimum score to accept a result
    verbose:             bool  = True         # Print progress to stdout
    use_local_classifier: bool = False        # Phase 2 local classifier fallback
    skip_verification:   bool  = False        # Unit-test candidate construction without GCC
    proof_corpus_path:   str   = ""           # Phase 3 proof corpus JSONL path
    reward_events_path:  str   = ""           # Phase 4 reward events JSONL path
    job_id:              str   = ""           # Optional job identifier for log prefixing


# ── Pipeline result ─────────────────────────────────────────────────────────────

@dataclass
class AttemptRecord:
    candidate: str
    score:     float
    gate:      str
    error:     str = ""

@dataclass
class PipelineResult:
    success:        bool  = False
    score:          float = 0.0
    best_candidate: str   = ""
    annotated_ir:   dict  = field(default_factory=dict)
    attempts:       list[AttemptRecord] = field(default_factory=list)
    rounds:         int   = 0
    elapsed_sec:    float = 0.0
    error:          str   = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

_record_lock = threading.Lock()


def _log(msg: str, verbose: bool, job_id: str = ""):
    if verbose:
        prefix = f"[APG:{job_id}]" if job_id else "[APG]"
        print(f"{prefix} {msg}")


def _verify_candidate_task(
    index: int,
    candidate: str,
    source_code: str,
    func_name: str,
    dep_graph: dict,
    annotated_ir: dict,
    test_inputs: list,
    cfg: PipelineConfig,
    active_model: str,
) -> tuple[int, str, VerificationResult]:
    """
    Task to verify a single candidate, optionally generating a proof first.
    Returns (index, candidate, result).
    """
    # Optional: generate Lean 4 proof for Gate 4
    lean_code = None
    if cfg.enable_proof:
        try:
            lean_code = generate_proof(
                source_code, candidate, dep_graph, annotated_ir,
                provider=cfg.provider, model=active_model,
            )
        except RuntimeError:
            lean_code = None

    ver = verify_candidate(
        candidate_code=candidate,
        reference_code=source_code,
        func_name=func_name,
        test_inputs=test_inputs,
        lean_code=lean_code,
        enable_tsan=cfg.enable_tsan,
        enable_cbmc=cfg.enable_cbmc,
        cbmc_path=cfg.cbmc_path,
        enable_proof=cfg.enable_proof,
    )

    with _record_lock:
        maybe_record_proof(
            cfg, source_code, candidate, annotated_ir, dep_graph,
            func_name, ver.gate_passed, ver.score >= cfg.score_threshold,
            ver.score, ver.details.get("cbmc", ver.error),
        )
        maybe_record_reward(
            cfg, source_code, candidate, func_name,
            ver.gate_passed, ver.score >= cfg.score_threshold, ver.score,
        )

    return index, candidate, ver

# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(
    source_code: str,
    func_name:   str            = "func",
    test_inputs: list           = None,
    config:      PipelineConfig = None,
) -> PipelineResult:
    """
    Run the full APG pipeline on a single C function.
    """
    cfg        = config or PipelineConfig()
    
    # Model Registry / A/B Test Routing
    active_model = cfg.model
    if cfg.model_version and cfg.model_version != "base":
        if cfg.provider == "ollama":
            # For Ollama, we expect the fine-tuned adapter to be merged and tagged
            active_model = f"apg-finetuned:{cfg.model_version}"
        else:
            # For cloud APIs, this might map to a specific fine-tuned endpoint
            active_model = f"fine-tuned-{cfg.model_version}"
        _log(f"Model Registry: Routing request to fine-tuned version '{cfg.model_version}' ({active_model})", cfg.verbose, cfg.job_id)

    test_inputs = test_inputs or [{"N": 1024, "seed": 42}, {"N": 512, "seed": 7}]
    result     = PipelineResult()
    start      = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 1 — T1 Analysis
    # ──────────────────────────────────────────────────────────────────────────
    _log("Layer 1: Running T1 analysis...", cfg.verbose, cfg.job_id)
    try:
        if cfg.use_local_classifier:
            annotated_ir = classify_region(source_code)
        else:
            annotated_ir = analyse(source_code, provider=cfg.provider, model=active_model)
    except (ValueError, RuntimeError) as e:
        if cfg.use_local_classifier:
            result.error = f"T1 analysis failed: {e}"
            _log(f"✗ {result.error}", cfg.verbose, cfg.job_id)
            result.elapsed_sec = time.time() - start
            return result
        annotated_ir = classify_region(source_code)
        _log("  T1 LLM analysis failed; used Phase 2 local classifier.", cfg.verbose, cfg.job_id)

    result.annotated_ir = annotated_ir
    _log(f"  Type: {annotated_ir['type']} | Summary: {annotated_ir['summary']}", cfg.verbose, cfg.job_id)

    # If the function is sequential with no parallelism opportunity, bail early
    if annotated_ir["type"] == "sequential":
        _log("  Function classified as sequential — no parallelism to apply.", cfg.verbose, cfg.job_id)
        result.error = "Function is sequential — no parallel transformation applicable."
        result.elapsed_sec = time.time() - start
        return result

    dep_graph = annotated_ir["dependencies"]

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 2 + 3 + 4 — Generate → Verify → Critique loop
    # ──────────────────────────────────────────────────────────────────────────
    best_score     = 0.0
    best_candidate = ""
    best_ver_result: Optional[VerificationResult] = None
    error_trace    = ""

    for round_num in range(1, cfg.max_critique_rounds + 1):
        result.rounds = round_num
        _log(f"\nLayer 2 [Round {round_num}]: Generating {cfg.max_t2_candidates} candidate(s)...", cfg.verbose, cfg.job_id)

        # ── T2: Generate candidates ────────────────────────────────────────────
        try:
            candidates = generate_candidate_pool(
                source_code=source_code,
                dep_graph=dep_graph,
                annotated_ir=annotated_ir,
                error_trace=error_trace,
                provider=cfg.provider,
                model=active_model,
                max_llm_candidates=cfg.max_t2_candidates,
            )
        except RuntimeError as e:
            result.error = f"T2 generation failed: {e}"
            _log(f"✗ {result.error}", cfg.verbose, cfg.job_id)
            break

        _log(f"  Generated {len(candidates)} candidate(s).", cfg.verbose, cfg.job_id)

        if cfg.skip_verification:
            best_candidate = candidates[0] if candidates else ""
            best_score = 0.0
            maybe_record_proof(
                cfg, source_code, best_candidate, annotated_ir, dep_graph,
                func_name, "skipped", False, best_score, "verification skipped",
            )
            maybe_record_reward(
                cfg, source_code, best_candidate, func_name,
                "skipped", False, best_score,
            )
            break

        # ── Layer 3: Verify each candidate ────────────────────────────────────
        round_best_score     = 0.0
        round_best_candidate = ""
        round_best_ver       = None

        if cfg.skip_verification:
            best_candidate = candidates[0] if candidates else ""
            best_score = 0.0
            maybe_record_proof(
                cfg, source_code, best_candidate, annotated_ir, dep_graph,
                func_name, "skipped", False, best_score, "verification skipped",
            )
            maybe_record_reward(
                cfg, source_code, best_candidate, func_name,
                "skipped", False, best_score,
            )
            break

        _log(f"  Verifying {len(candidates)} candidate(s) in parallel...", cfg.verbose, cfg.job_id)

        with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as executor:
            futures = [
                executor.submit(
                    _verify_candidate_task, i, candidate, source_code,
                    func_name, dep_graph, annotated_ir, test_inputs, cfg,
                    active_model
                )
                for i, candidate in enumerate(candidates)
            ]

            for future in futures:
                i, candidate, ver = future.result()
                result.attempts.append(AttemptRecord(
                    candidate=candidate,
                    score=ver.score,
                    gate=ver.gate_passed,
                    error=ver.error
                ))
                _log(f"    Candidate {i+1} Score: {ver.score:.1f} | Gate: {ver.gate_passed} | Error: {ver.error[:60] if ver.error else 'none'}", cfg.verbose, cfg.job_id)

                if ver.score > round_best_score:
                    round_best_score     = ver.score
                    round_best_candidate = candidate
                    round_best_ver       = ver

        # Track overall best
        if round_best_score > best_score:
            best_score      = round_best_score
            best_candidate  = round_best_candidate
            best_ver_result = round_best_ver

        # ── Check if we've met the threshold ──────────────────────────────────
        if best_score >= cfg.score_threshold:
            _log(f"\n✓ Score {best_score:.1f} meets threshold {cfg.score_threshold:.1f}. Done.", cfg.verbose, cfg.job_id)
            break

        # ── Layer 4: T3 Critique — prepare error for next round ───────────────
        if best_ver_result and best_ver_result.error:
            _log(f"\nLayer 4 [Round {round_num}]: Running T3 critique...", cfg.verbose, cfg.job_id)
            try:
                fix = critique(
                    original_code=source_code,
                    candidate_code=best_candidate,
                    error_message=best_ver_result.error,
                    gate_failed=best_ver_result.gate_reached,
                    provider=cfg.provider,
                    model=active_model,
                )
                _log(f"  T3 diagnosis: {fix['error_type']} — {fix['root_cause']}", cfg.verbose, cfg.job_id)
                # Apply the patch to the best candidate for the next round's context
                from workers.common.patch_applier import apply_patch
                best_candidate = apply_patch(best_candidate, fix["patch"])
                error_trace    = f"{fix['error_type']}: {fix['root_cause']}\nPrevious fix attempt:\n{fix['fix_description']}"
            except Exception as e:
                _log(f"  T3 critique or patch application failed: {e}", cfg.verbose, cfg.job_id)
                error_trace = best_ver_result.error

    # ── Assemble final result ──────────────────────────────────────────────────
    result.score          = best_score
    result.best_candidate = best_candidate
    result.success        = best_score >= cfg.score_threshold
    result.elapsed_sec    = time.time() - start

    _log(f"\n{'='*50}", cfg.verbose, cfg.job_id)
    _log(f"Pipeline complete in {result.elapsed_sec:.1f}s", cfg.verbose, cfg.job_id)
    _log(f"Final score: {result.score:.1f} | Rounds: {result.rounds} | Success: {result.success}", cfg.verbose, cfg.job_id)

    return result


def maybe_record_proof(
    cfg: PipelineConfig,
    source_code: str,
    candidate_code: str,
    annotated_ir: dict,
    dep_graph: dict,
    func_name: str,
    verifier: str,
    success: bool,
    score: float,
    proof: str,
) -> None:
    if not cfg.proof_corpus_path or not candidate_code:
        return
    ProofCorpusStore(cfg.proof_corpus_path).append(
        ProofRecord(
            source_code=source_code,
            parallel_code=candidate_code,
            verifier=verifier,
            success=success,
            score=score,
            proof=proof,
            annotated_ir=annotated_ir,
            dep_graph=dep_graph,
            func_name=func_name,
        )
    )


def maybe_record_reward(
    cfg: PipelineConfig,
    source_code: str,
    candidate_code: str,
    func_name: str,
    verifier: str,
    success: bool,
    score: float,
) -> None:
    if not cfg.reward_events_path or not candidate_code:
        return
    RewardEventStore(cfg.reward_events_path).append(
        RewardEvent(
            source_code=source_code,
            candidate_code=candidate_code,
            func_name=func_name,
            verifier=verifier,
            reward=reward_from_score(score),
            score=score,
            success=success,
        )
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Example: run from command line with a hardcoded test function
    source = """void matrix_scale(float* M, float s, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i * cols + j] *= s;
        }
    }
}"""

    print("Running APG Pipeline on matrix_scale...")
    print(f"Source:\n{source}\n")

    cfg = PipelineConfig(
        provider="ollama",
        max_t2_candidates=2,
        max_critique_rounds=2,
        enable_proof=False,
        verbose=True,
    )

    result = run(
        source_code=source,
        func_name="matrix_scale",
        test_inputs=[{"N": 256, "seed": 42}],
        config=cfg,
    )

    print(f"\n{'='*50}")
    print(f"Success:      {result.success}")
    print(f"Score:        {result.score:.2f}")
    print(f"Rounds:       {result.rounds}")
    print(f"Elapsed:      {result.elapsed_sec:.1f}s")
    print(f"\nBest candidate:\n{result.best_candidate}")
    print(f"\nAnnotated IR:\n{json.dumps(result.annotated_ir, indent=2)}")

"""
test_cbmc_integration.py
------------------------
Integration test for Gate 3b: CBMC bounded model checking.

Skips automatically if CBMC is not installed.
"""
import shutil
import pytest
from workers.spec_generator.verifier import gate_cbmc_model_check, verify_candidate


pytestmark = pytest.mark.skipif(
    not shutil.which("cbmc"),
    reason="CBMC not installed — skipping CBMC integration tests"
)


VECTOR_SCALE_SEQ = """
void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) A[i] *= s;
}
"""

VECTOR_SCALE_PAR = """
void vector_scale(float* A, float s, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) A[i] *= s;
}
"""

VECTOR_SCALE_RACE = """
void vector_scale(float* A, float s, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 1; i++) {
        A[i] = A[i+1] * s;  /* WAR dependency — may race */
    }
}
"""


def test_cbmc_correct_candidate_passes():
    """A correct parallel candidate should pass CBMC."""
    ok, msg = gate_cbmc_model_check(
        candidate_code=VECTOR_SCALE_PAR,
        reference_code=VECTOR_SCALE_SEQ,
        func_name="vector_scale",
        array_size=4,
    )
    assert ok, f"Expected CBMC pass, got failure: {msg}"


def test_cbmc_result_stored_in_details():
    """verify_candidate with enable_cbmc=True stores CBMC output in details."""
    result = verify_candidate(
        candidate_code=VECTOR_SCALE_PAR,
        reference_code=VECTOR_SCALE_SEQ,
        func_name="vector_scale",
        test_inputs=[{"N": 64, "seed": 42}],
        enable_tsan=False,
        enable_cbmc=True,
    )
    assert "cbmc" in result.details, "details should contain 'cbmc' key"
    assert result.score >= 0.6, f"Score should be >= 0.6, got {result.score}"

import unittest

from workers.pipeline import PipelineConfig
from workers.spec_generator.verifier import build_cbmc_harness, gate_cbmc_model_check


class Phase3CBMCTests(unittest.TestCase):
    def test_cbmc_harness_compares_reference_and_candidate_outputs(self):
        reference = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) A[i] *= s;
}"""
        candidate = """void vector_scale(float* A, float s, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) A[i] *= s;
}"""

        harness = build_cbmc_harness(candidate, reference, "vector_scale", array_size=8)

        self.assertIn("void apg_ref_vector_scale", harness)
        self.assertIn("void apg_candidate_vector_scale", harness)
        self.assertIn("apg_ref_vector_scale(ref_A, 2.5f, N);", harness)
        self.assertIn("apg_candidate_vector_scale(par_A, 2.5f, N);", harness)
        self.assertIn("float diff_A = ref_A[i] - par_A[i];", harness)

    def test_cbmc_gate_skips_when_cbmc_is_unavailable(self):
        ok, message = gate_cbmc_model_check(
            candidate_code="void f(void) {}",
            reference_code="void f(void) {}",
            func_name="f",
            cbmc_path="definitely_missing_cbmc_binary",
        )

        self.assertTrue(ok)
        self.assertIn("skipped", message.lower())

    def test_pipeline_config_accepts_cbmc_options(self):
        config = PipelineConfig(enable_cbmc=True, cbmc_path="cbmc")

        self.assertTrue(config.enable_cbmc)
        self.assertEqual(config.cbmc_path, "cbmc")


if __name__ == "__main__":
    unittest.main()

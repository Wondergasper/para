import unittest

from workers.spec_generator.verifier import build_output_harness, build_tsan_harness


class OutputHarnessTests(unittest.TestCase):
    def test_harness_calls_reference_and_candidate_functions(self):
        reference = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) A[i] *= s;
}"""
        candidate = """void vector_scale(float* A, float s, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) A[i] *= s;
}"""

        harness = build_output_harness(candidate, reference, "vector_scale", 128, 42)

        self.assertIn("void apg_ref_vector_scale", harness)
        self.assertIn("void apg_candidate_vector_scale", harness)
        self.assertIn("apg_ref_vector_scale(ref_A, 2.5f, N);", harness)
        self.assertIn("apg_candidate_vector_scale(par_A, 2.5f, N);", harness)
        self.assertIn("double diff_A = fabs((double)ref_A[i] - (double)par_A[i]);", harness)

    def test_harness_routes_output_pointer_for_vector_add(self):
        reference = """void vector_add(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}"""
        candidate = """void vector_add(float* A, float* B, float* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}"""

        harness = build_output_harness(candidate, reference, "vector_add", 128, 42)

        self.assertIn("apg_ref_vector_add(ref_A, ref_B, ref_C, N);", harness)
        self.assertIn("apg_candidate_vector_add(par_A, par_B, par_C, N);", harness)

    def test_tsan_harness_calls_candidate_function(self):
        candidate = """void vector_add(float* A, float* B, float* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}"""

        harness = build_tsan_harness(candidate, "vector_add")

        self.assertIn("void apg_tsan_vector_add", harness)
        self.assertIn("apg_tsan_vector_add(A, B, C, N);", harness)
        self.assertIn("float *C = (float*)malloc(N * sizeof(float));", harness)


    def test_is_docker_functional(self):
        from workers.spec_generator.verifier import is_docker_functional
        res = is_docker_functional()
        self.assertIn(res, (True, False))


if __name__ == "__main__":
    unittest.main()

import unittest
import unittest.mock

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

    @unittest.mock.patch("workers.spec_generator.verifier.is_docker_functional", return_value=False)
    @unittest.mock.patch("workers.spec_generator.verifier.find_gcc", return_value="/usr/bin/gcc")
    @unittest.mock.patch("workers.spec_generator.verifier.subprocess.run")
    def test_verify_candidate_autotuning_success(self, mock_run, mock_find_gcc, mock_is_docker):
        from unittest.mock import MagicMock
        
        def make_completed_process(returncode, stdout, stderr=""):
            m = MagicMock()
            m.returncode = returncode
            m.stdout = stdout
            m.stderr = stderr
            return m

        mock_run.side_effect = [
            make_completed_process(0, ""), # Gate 1 compile
            make_completed_process(0, ""), # Gate 2 compile
            make_completed_process(0, "PASS max_diff=0.00e+00 time_ref=0.100000 time_par=0.100000"), # Thread 1
            make_completed_process(0, "PASS max_diff=0.00e+00 time_ref=0.100000 time_par=0.060000"), # Thread 2
            make_completed_process(0, "PASS max_diff=0.00e+00 time_ref=0.100000 time_par=0.030000"), # Thread 4
            make_completed_process(0, "PASS max_diff=0.00e+00 time_ref=0.100000 time_par=0.040000"), # Thread 8
        ]

        from workers.spec_generator.verifier import verify_candidate
        
        candidate = "void dummy(int N) { #pragma omp parallel for\nfor(int i=0; i<N; i++); }"
        reference = "void dummy(int N) { for(int i=0; i<N; i++); }"
        
        res = verify_candidate(candidate, reference, func_name="dummy", enable_tsan=False)
        
        self.assertTrue(res.compile_ok)
        self.assertTrue(res.output_ok)
        self.assertIn("autotuning", res.details)
        
        tuning = res.details["autotuning"]
        self.assertEqual(tuning["best_threads"], 4)
        self.assertAlmostEqual(tuning["best_speedup"], 10.0 / 3.0) # 0.1 / 0.03
        self.assertEqual(len(tuning["runs"]), 4)
        self.assertEqual(tuning["runs"][2]["threads"], 4)

    @unittest.mock.patch("workers.spec_generator.t3_critique.generate")
    def test_multi_agent_critique_debate(self, mock_generate):
        mock_generate.side_effect = [
            "Primary Critic: Accumulator needs reduction.",
            "Peer 1: I agree, reduction is missing.",
            "Peer 2: I also agree, atomic would work too.",
            '{"error_type": "REDUCTION_BUG", "root_cause": "missing reduction", "fix_description": "add reduction", "patch": "--- candidate\\n+++ fixed\\n@@ -1,2 +1,2 @@\\n-#pragma omp parallel for\\n+#pragma omp parallel for reduction(+:sum)"}'
        ]
        
        from workers.spec_generator.t3_critique import critique
        
        original = "void f() { int sum = 0; for (int i=0; i<10; i++) sum += i; }"
        candidate = "void f() { int sum = 0; #pragma omp parallel for\nfor (int i=0; i<10; i++) sum += i; }"
        error_msg = "output mismatch at sum"
        
        res = critique(original, candidate, error_msg, gate_failed="output")
        
        self.assertEqual(res["error_type"], "REDUCTION_BUG")
        self.assertEqual(res["root_cause"], "missing reduction")
        self.assertEqual(res["fix_description"], "add reduction")
        self.assertIn("reduction(+:sum)", res["patch"])
        self.assertEqual(mock_generate.call_count, 4)

    def test_gate_formal_proof_success(self):
        from workers.spec_generator.verifier import find_lean, gate_formal_proof
        lean_exe = find_lean()
        if not lean_exe:
            self.skipTest("Lean 4 compiler not available on this system")
            
        valid_lean = """
def dummy : Nat := 1
theorem dummy_eq : dummy = 1 := rfl
"""
        ok, err = gate_formal_proof(valid_lean)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_gate_formal_proof_failure(self):
        from workers.spec_generator.verifier import find_lean, gate_formal_proof
        lean_exe = find_lean()
        if not lean_exe:
            self.skipTest("Lean 4 compiler not available on this system")
            
        invalid_lean = """
def dummy : Nat := 1
theorem dummy_eq : dummy = 2 := rfl
"""
        ok, err = gate_formal_proof(invalid_lean)
        self.assertFalse(ok)
        self.assertTrue("type mismatch" in err.lower() or "error" in err.lower())

    @unittest.mock.patch("workers.spec_generator.verifier.find_nvcc", return_value="/usr/local/cuda/bin/nvcc")
    @unittest.mock.patch("workers.spec_generator.verifier.subprocess.run")
    def test_verify_candidate_cuda_success(self, mock_run, mock_find_nvcc):
        from unittest.mock import MagicMock
        
        def make_completed_process(returncode, stdout, stderr=""):
            m = MagicMock()
            m.returncode = returncode
            m.stdout = stdout
            m.stderr = stderr
            return m

        mock_run.side_effect = [
            make_completed_process(0, ""), # Gate 1 compile (nvcc)
            make_completed_process(0, ""), # Gate 2 compile (nvcc)
            make_completed_process(0, "PASS max_diff=0.00e+00 time_ref=0.100000 time_par=0.020000"), # CUDA run
        ]

        from workers.spec_generator.verifier import verify_candidate
        
        candidate = "__global__ void dummy_kernel() {} void dummy(int N) { dummy_kernel<<<1, 1>>>(); }"
        reference = "void dummy(int N) { for(int i=0; i<N; i++); }"
        
        res = verify_candidate(candidate, reference, func_name="dummy", enable_tsan=True, target="cuda")
        
        self.assertTrue(res.compile_ok)
        self.assertTrue(res.output_ok)
        self.assertFalse(res.race_ok)  # TSAN is bypassed, so race_ok remains False
        self.assertIn("autotuning", res.details)
        
        tuning = res.details["autotuning"]
        self.assertEqual(tuning["best_threads"], 1)
        self.assertAlmostEqual(tuning["best_speedup"], 5.0) # 0.1 / 0.02
        self.assertEqual(len(tuning["runs"]), 1)
        
        # Verify mock_run called nvcc and .cu files
        args_gate1 = mock_run.call_args_list[0][0][0]
        self.assertIn("/usr/local/cuda/bin/nvcc", args_gate1)
        self.assertTrue(any(arg.endswith(".cu") for arg in args_gate1))

        args_gate2 = mock_run.call_args_list[1][0][0]
        self.assertIn("/usr/local/cuda/bin/nvcc", args_gate2)
        self.assertTrue(any(arg.endswith(".cu") for arg in args_gate2))

    @unittest.mock.patch("workers.gen_openmp.t2_generate.generate_batch")
    def test_generate_candidates_cuda_prompt(self, mock_generate_batch):
        mock_generate_batch.return_value = ["void mock_cuda() {}"]
        
        from workers.gen_openmp.t2_generate import generate_parallel
        
        source = "void dummy() {}"
        dep_graph = {"RAW": [], "WAR": [], "WAW": []}
        
        res = generate_parallel(source, dep_graph, target="cuda")
        
        self.assertEqual(res, ["void mock_cuda() {}"])
        mock_generate_batch.assert_called_once()
        kwargs = mock_generate_batch.call_args[1]
        self.assertIn("system", kwargs)
        self.assertIn("CUDA", kwargs["system"])
        self.assertIn("__global__", kwargs["system"])


if __name__ == "__main__":
    unittest.main()

import os
import unittest
import tempfile
import json
from workers.pipeline import PipelineConfig, run
from workers.spec_generator.proof_corpus import ProofCorpusStore, ProofRecord
from workers.lang.c_driver import CDriver
from workers.lang.fortran_driver import FortranDriver
from workers.lang.python_driver import PythonDriver
from workers.lang.rust_driver import RustDriver

class E2EIntegrationTests(unittest.TestCase):
    def test_proof_corpus_deduplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "proofs.jsonl")
            store = ProofCorpusStore(path)

            record1 = ProofRecord("void f() {}", "void f() { #pragma omp parallel }", "skipped", True, 0.8)
            record2 = ProofRecord("void f() {}", "void f() { #pragma omp parallel }", "skipped", True, 0.8)
            record3 = ProofRecord("void f() {}", "void f() { #pragma omp parallel for }", "skipped", True, 0.8)

            store.append(record1)
            store.append(record2)  # Duplicate, should be ignored
            store.append(record3)  # Distinct, should be saved

            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.assertEqual(len(lines), 2)
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])
            self.assertEqual(data1["parallel_code"], "void f() { #pragma omp parallel }")
            self.assertEqual(data2["parallel_code"], "void f() { #pragma omp parallel for }")

    def test_c_pipeline_produces_unified_diff(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}"""
        cfg = PipelineConfig(
            use_local_classifier=True,
            max_t2_candidates=0,
            max_critique_rounds=1,
            skip_verification=True,
            verbose=False,
        )
        res = run(source_code=source, func_name="vector_scale", config=cfg)
        self.assertTrue(res.best_candidate)
        self.assertTrue(res.diff)
        self.assertIn("--- original/vector_scale", res.diff)
        self.assertIn("+++ parallel/vector_scale", res.diff)
        self.assertIn("+    #pragma omp parallel for", res.diff)

    def test_tsan_warning_on_windows(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) A[i] *= s;
}"""
        cfg = PipelineConfig(
            use_local_classifier=True,
            max_t2_candidates=0,
            max_critique_rounds=1,
            skip_verification=True,
            verbose=False,
            enable_tsan=False,
        )
        res = run(source_code=source, func_name="vector_scale", config=cfg)
        self.assertIn("warnings", res.annotated_ir)
        self.assertIn("ThreadSanitizer disabled on Windows. Race detection bypassed.", res.annotated_ir["warnings"])

    def test_fortran_driver_extraction_and_safety(self):
        driver = FortranDriver()
        source = """
        subroutine mat_scale(A, s, N)
            integer :: N
            real :: A(N)
            real :: s
            integer :: i
            do i = 1, N
                A(i) = A(i) * s
            end do
        end subroutine mat_scale
        """
        funcs = driver.extract_functions(source)
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0]["name"], "mat_scale")
        self.assertIn("do i = 1, N", funcs[0]["body"])

        # Test safety
        safe_body = "do i = 1, N\nA(i) = A(i) * 2\nend do"
        unsafe_exit = "do i = 1, N\nif (A(i) < 0) exit\nend do"
        unsafe_io = "do i = 1, N\nprint *, A(i)\nend do"

        self.assertTrue(driver.analyze_loops(safe_body)["safe_for_openmp"])
        self.assertFalse(driver.analyze_loops(unsafe_exit)["safe_for_openmp"])
        self.assertFalse(driver.analyze_loops(unsafe_io)["safe_for_openmp"])

    def test_python_driver_extraction_and_safety(self):
        driver = PythonDriver()
        source = """
def scale(A, s, N):
    for i in range(N):
        A[i] *= s
"""
        funcs = driver.extract_functions(source)
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0]["name"], "scale")

        # Test safety
        safe_body = "for i in range(N):\n    A[i] = A[i] * s"
        unsafe_return = "for i in range(N):\n    if A[i] < 0:\n        return i"
        unsafe_io = "for i in range(N):\n    print(A[i])"

        self.assertTrue(driver.analyze_loops(safe_body)["safe_for_openmp"])
        self.assertFalse(driver.analyze_loops(unsafe_return)["safe_for_openmp"])
        self.assertFalse(driver.analyze_loops(unsafe_io)["safe_for_openmp"])

    def test_rust_driver_extraction_and_safety(self):
        driver = RustDriver()
        source = """
pub fn scale(A: &mut [f32], s: f32, N: usize) {
    for i in 0..N {
        A[i] *= s;
    }
}
"""
        funcs = driver.extract_functions(source)
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0]["name"], "scale")

        # Test safety
        safe_body = "for i in 0..N {\n    A[i] = A[i] * s;\n}"
        unsafe_exit = "for i in 0..N {\n    if A[i] < 0.0 {\n        break;\n    }\n}"
        unsafe_io = "for i in 0..N {\n    println!(\"{}\", A[i]);\n}"

        self.assertTrue(driver.analyze_loops(safe_body)["safe_for_openmp"])
        self.assertFalse(driver.analyze_loops(unsafe_exit)["safe_for_openmp"])
        self.assertFalse(driver.analyze_loops(unsafe_io)["safe_for_openmp"])

if __name__ == "__main__":
    unittest.main()

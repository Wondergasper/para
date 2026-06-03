import unittest
from workers.gen_openmp.ctt_transform import generate_ctt_candidates

class TestCTTTransforms(unittest.TestCase):
    def test_generate_ctt_candidates_simple(self):
        source = "void f() { for (int i=0; i<10; i++) A[i] = 0; }"
        ir = {"type": "polyhedral", "dependencies": {"RAW": []}}
        candidates = generate_ctt_candidates(source, ir)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertIn("#pragma omp parallel for", candidates[0])

    def test_generate_ctt_candidates_2d_nest(self):
        source = """void matmul(float* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 0.0f;
        }
    }
}"""
        ir = {"type": "polyhedral", "dependencies": {"RAW": []}}
        candidates = generate_ctt_candidates(source, ir)
        
        # Should have at least:
        # 1. Simple parallel for
        # 2. Loop collapse(2)
        # 3. Loop interchange (if implemented correctly)
        self.assertGreaterEqual(len(candidates), 2)
        
        has_collapse = any("collapse(2)" in c for c in candidates)
        self.assertTrue(has_collapse, "Should have a collapse(2) candidate")
        
        # Verify interchange (swapping i and j)
        has_interchange = any("for (int j = 0;" in c and c.find("for (int j = 0;") < c.find("for (int i = 0;") for c in candidates)
        self.assertTrue(has_interchange, "Should have an interchanged candidate")

if __name__ == "__main__":
    unittest.main()

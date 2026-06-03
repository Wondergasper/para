import unittest

from workers.code_understanding.polyhedral import classify_region
from workers.gen_openmp.ctt_transform import generate_ctt_candidate
from workers.gen_openmp.t2_generate import generate_candidate_pool
from workers.pipeline import PipelineConfig, run


class Phase2PolyhedralTests(unittest.TestCase):
    def test_classifies_simple_affine_loop_as_polyhedral(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}"""

        result = classify_region(source)

        self.assertEqual(result["type"], "polyhedral")
        self.assertEqual(result["affine_dimensions"], ["i"])
        self.assertFalse(result["dependencies"]["RAW"])

    def test_classifies_complex_affine_index_as_polyhedral(self):
        source = """void scale(float* A, int N) {
    for (int i = 0; i < N; i++) {
        A[i * 2 + 1] *= 2.0f;
    }
}"""
        result = classify_region(source)
        self.assertEqual(result["type"], "polyhedral")

    def test_classifies_non_affine_index_as_irregular(self):
        source = """void scale(float* A, int N) {
    for (int i = 0; i < N; i++) {
        A[i * i] *= 2.0f;
    }
}"""
        result = classify_region(source)
        self.assertEqual(result["type"], "irregular")

    def test_classifies_indirect_index_as_irregular(self):
        source = """void scatter(float* A, int* idx, int N) {
    for (int i = 0; i < N; i++) {
        A[idx[i]] = 1.0f;
    }
}"""

        result = classify_region(source)

        self.assertEqual(result["type"], "irregular")
        self.assertIn("indirect", result["summary"].lower())


class Phase2CandidatePoolTests(unittest.TestCase):
    def test_ctt_candidate_adds_openmp_pragma_to_polyhedral_loop(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}"""
        annotated_ir = classify_region(source)

        candidate = generate_ctt_candidate(source, annotated_ir)

        self.assertIn("#pragma omp parallel for", candidate)
        self.assertIn("for (int i = 0; i < N; i++)", candidate)

    def test_candidate_pool_uses_ctt_before_llm_candidates(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}"""
        annotated_ir = classify_region(source)

        candidates = generate_candidate_pool(
            source_code=source,
            dep_graph=annotated_ir["dependencies"],
            annotated_ir=annotated_ir,
            llm_generator=lambda **_: ["void llm_candidate(void) {}"],
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertIn("#pragma omp parallel for", candidates[0])
        self.assertEqual(candidates[1], "void llm_candidate(void) {}")

    def test_pipeline_can_fallback_to_local_classifier(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}"""

        result = run(
            source_code=source,
            func_name="vector_scale",
            config=PipelineConfig(
                use_local_classifier=True,
                max_t2_candidates=0,
                max_critique_rounds=1,
                enable_tsan=False,
                skip_verification=True,
                verbose=False,
            ),
        )

        self.assertEqual(result.annotated_ir["type"], "polyhedral")
        self.assertIn("#pragma omp parallel for", result.best_candidate)


    def test_generate_batch_mocked(self):
        from unittest.mock import MagicMock, patch
        from workers.common.llm_client import generate_batch

        mock_choice = MagicMock()
        mock_choice.message.content = "void test_func() {}"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice, mock_choice, mock_choice]

        with patch("workers.common.llm_client.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            results = generate_batch(
                prompt="test",
                system="test",
                provider="ollama",
                n=3
            )
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0], "void test_func() {}")
            mock_client.chat.completions.create.assert_called_once()
            _, kwargs = mock_client.chat.completions.create.call_args
            self.assertEqual(kwargs["n"], 3)

    def test_generate_batch_logs_batch_failure(self):
        from unittest.mock import MagicMock, patch
        from workers.common.llm_client import generate_batch

        mock_choice = MagicMock()
        mock_choice.message.content = "void fallback(void) {}"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("workers.common.llm_client.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                RuntimeError("n unsupported"),
                mock_response,
            ]
            mock_get_client.return_value = mock_client

            with self.assertLogs("apg.llm", level="WARNING") as logs:
                results = generate_batch(
                    prompt="test",
                    system="test",
                    provider="ollama",
                    n=1,
                )

            self.assertEqual(results, ["void fallback(void) {}"])
            self.assertTrue(any("Batch generation failed" in msg for msg in logs.output))

    def test_generate_retries_transient_failures(self):
        from unittest.mock import MagicMock, patch
        from workers.common.llm_client import generate

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("workers.common.llm_client.get_client") as mock_get_client, \
             patch("workers.common.llm_client.time.sleep") as mock_sleep:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                RuntimeError("temporary"),
                mock_response,
            ]
            mock_get_client.return_value = mock_client

            result = generate(
                prompt="test",
                system="test",
                provider="ollama",
                max_retries=2,
                retry_backoff=0,
            )

            self.assertEqual(result, "ok")
            self.assertEqual(mock_client.chat.completions.create.call_count, 2)
            mock_sleep.assert_called_once()


    def test_clang_analyzer_checks_availability(self):
        from workers.code_understanding.clang_analyzer import is_clang_available
        res = is_clang_available()
        self.assertIn(res, (True, False))


if __name__ == "__main__":
    unittest.main()

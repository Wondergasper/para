import json
import os
import tempfile
import unittest

from workers.pipeline import PipelineConfig, run
from workers.spec_generator.proof_corpus import ProofCorpusStore, ProofRecord


class ProofCorpusStoreTests(unittest.TestCase):
    def test_appends_verified_record_as_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "proofs.jsonl")
            store = ProofCorpusStore(path)

            record = ProofRecord(
                source_code="void f(void) {}",
                parallel_code="void f(void) {}",
                verifier="cbmc",
                success=True,
                score=0.8,
                proof="assertions passed",
                annotated_ir={"type": "polyhedral"},
                dep_graph={"RAW": [], "WAR": [], "WAW": []},
                func_name="f",
            )
            store.append(record)

            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verifier"], "cbmc")
        self.assertTrue(rows[0]["success"])
        self.assertEqual(rows[0]["func_name"], "f")

    def test_iter_verified_filters_successful_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "proofs.jsonl")
            store = ProofCorpusStore(path)
            store.append(ProofRecord("src", "par", "cbmc", True, 0.8))
            store.append(ProofRecord("src", "bad", "cbmc", False, 0.2))

            verified = list(store.iter_verified(min_score=0.8))

        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0].parallel_code, "par")

    def test_pipeline_writes_proof_corpus_record_when_configured(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}"""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "proofs.jsonl")
            result = run(
                source_code=source,
                func_name="vector_scale",
                config=PipelineConfig(
                    use_local_classifier=True,
                    max_t2_candidates=0,
                    max_critique_rounds=1,
                    skip_verification=True,
                    proof_corpus_path=path,
                    verbose=False,
                ),
            )
            records = list(ProofCorpusStore(path).iter_records())

        self.assertIn("#pragma omp parallel for", result.best_candidate)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].func_name, "vector_scale")
        self.assertEqual(records[0].verifier, "skipped")


if __name__ == "__main__":
    unittest.main()

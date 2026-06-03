import json
import os
import tempfile
import unittest

from workers.pipeline import PipelineConfig, run
from workers.spec_generator.proof_corpus import ProofCorpusStore, ProofRecord
from workers.spec_generator.reward import reward_from_gates, reward_from_score
from workers.spec_generator.reward_events import RewardEventStore
from workers.training.export_corpus import export_local_corpus


class Phase4ExportTests(unittest.TestCase):
    def test_exports_local_proof_corpus_to_sft_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proofs_path = os.path.join(tmpdir, "proofs.jsonl")
            output_path = os.path.join(tmpdir, "corpus.jsonl")
            store = ProofCorpusStore(proofs_path)
            store.append(
                ProofRecord(
                    source_code="void vector_scale(float* A, float s, int N) { for (int i = 0; i < N; i++) A[i] *= s; }",
                    parallel_code="void vector_scale(float* A, float s, int N) { #pragma omp parallel for\nfor (int i = 0; i < N; i++) A[i] *= s; }",
                    verifier="cbmc",
                    success=True,
                    score=0.8,
                    dep_graph={"RAW": [], "WAR": [], "WAW": []},
                    annotated_ir={"type": "polyhedral", "reduction_variables": []},
                    func_name="vector_scale",
                )
            )
            store.append(
                ProofRecord(
                    source_code="void bad(void) {}",
                    parallel_code="void bad(void) {}",
                    verifier="cbmc",
                    success=False,
                    score=0.2,
                )
            )

            written = export_local_corpus(proofs_path, output_path, min_score=0.8, verbose=False)

            with open(output_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(written, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual([m["role"] for m in rows[0]["messages"]], ["system", "user", "assistant"])
        self.assertIn("vector_scale", rows[0]["messages"][1]["content"])
        self.assertIn("#pragma omp parallel for", rows[0]["messages"][2]["content"])


class Phase4RewardTests(unittest.TestCase):
    def test_reward_from_gates_matches_architecture_weights(self):
        reward = reward_from_gates(
            compile_ok=True,
            output_ok=True,
            race_ok=True,
            proof_ok=False,
        )

        self.assertEqual(reward, 0.8)

    def test_reward_from_score_clamps_to_unit_interval(self):
        self.assertEqual(reward_from_score(1.3), 1.0)
        self.assertEqual(reward_from_score(-0.2), 0.0)

    def test_pipeline_emits_reward_event_when_configured(self):
        source = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}"""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rewards.jsonl")
            run(
                source_code=source,
                func_name="vector_scale",
                config=PipelineConfig(
                    use_local_classifier=True,
                    max_t2_candidates=0,
                    max_critique_rounds=1,
                    skip_verification=True,
                    reward_events_path=path,
                    verbose=False,
                ),
            )
            events = list(RewardEventStore(path).iter_events())

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].func_name, "vector_scale")
        self.assertEqual(events[0].reward, 0.0)
        self.assertEqual(events[0].verifier, "skipped")


if __name__ == "__main__":
    unittest.main()

"""
Command-line bridge for the Go Phase 1 orchestrator.

Reads a C source file, runs the APG Python pipeline, and prints a single JSON
object to stdout. Logs must stay off stdout because the Go runner parses stdout.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.pipeline import PipelineConfig, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the APG Phase 1 pipeline.")
    parser.add_argument("--source", required=True, help="Path to a C source file.")
    parser.add_argument("--func-name", default="func", help="Function name under test.")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "groq", "gemini"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--enable-tsan", action="store_true")
    parser.add_argument("--enable-cbmc", action="store_true")
    parser.add_argument("--cbmc-path", default="cbmc")
    parser.add_argument("--enable-proof", action="store_true")
    parser.add_argument("--proof-corpus", default="", help="Optional JSONL path for proof records.")
    parser.add_argument("--reward-events", default="", help="Optional JSONL path for reward events.")
    args = parser.parse_args()

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            source_code = f.read()
    except OSError as exc:
        print(json.dumps({"success": False, "score": 0.0, "error": str(exc)}))
        return 1

    cfg = PipelineConfig(
        provider=args.provider,
        model=args.model,
        max_t2_candidates=args.max_candidates,
        max_critique_rounds=args.max_rounds,
        enable_proof=args.enable_proof,
        enable_cbmc=args.enable_cbmc,
        cbmc_path=args.cbmc_path,
        enable_tsan=args.enable_tsan and os.name != "nt",
        proof_corpus_path=args.proof_corpus,
        reward_events_path=args.reward_events,
        verbose=False,
    )

    try:
        result = run(source_code=source_code, func_name=args.func_name, config=cfg)
        print(json.dumps(asdict(result)))
        return 0 if result.success else 1
    except Exception as exc:
        print(json.dumps({"success": False, "score": 0.0, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Command-line bridge for the Go orchestrator.

Two modes:
  1. --source FILE   (original) -- reads a C source file
  2. --json          (local-queue mode) -- reads a JSON object from stdin:
       { "id": "...", "source": "...", "func_name": "...", "model_version": "..." }
     Prints a single JSON object to stdout.
     All logs go to stderr so Go can parse stdout cleanly.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.pipeline import PipelineConfig, run

# Redirect all logging to stderr so stdout stays clean JSON
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the APG pipeline.")
    parser.add_argument("--source",         default=None,
                        help="Path to a C source file (file mode).")
    parser.add_argument("--json",           action="store_true",
                        help="Read JSON job from stdin, write JSON result to stdout (local-queue mode).")
    parser.add_argument("--func-name",      default="func")
    parser.add_argument("--provider",       default=os.getenv("APG_PROVIDER", "ollama"),
                        choices=["ollama", "groq", "gemini"])
    parser.add_argument("--model",          default=os.getenv("APG_MODEL", None))
    parser.add_argument("--max-candidates", type=int, default=int(os.getenv("APG_MAX_CANDIDATES", "3")))
    parser.add_argument("--max-rounds",     type=int, default=int(os.getenv("APG_MAX_ROUNDS", "3")))
    parser.add_argument("--enable-tsan",    action="store_true")
    parser.add_argument("--enable-cbmc",    action="store_true")
    parser.add_argument("--cbmc-path",      default="cbmc")
    parser.add_argument("--enable-proof",   action="store_true")
    parser.add_argument("--proof-corpus",   default="data/proofs.jsonl")
    parser.add_argument("--reward-events",  default="data/rewards.jsonl")
    parser.add_argument("--language",       default="c",
                        choices=["c", "fortran", "python", "rust"])
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Mode 1: --json  (local-queue / Go subprocess mode)
    # ------------------------------------------------------------------
    if args.json:
        try:
            payload = json.load(sys.stdin)
        except Exception as exc:
            print(json.dumps({"success": False, "score": 0.0,
                              "error": f"stdin parse error: {exc}"}))
            return 1

        source_code   = payload.get("source", "")
        func_name     = payload.get("func_name", "func") or "func"
        model_version = payload.get("model_version", "base")

        if not source_code.strip():
            print(json.dumps({"success": False, "score": 0.0,
                              "error": "empty source code"}))
            return 1

        # Fallback defaults from CLI arguments
        provider = args.provider
        model = args.model
        max_candidates = args.max_candidates
        max_rounds = args.max_rounds
        use_local_classifier = False
        target = "openmp"
        language = args.language
        enable_cbmc = args.enable_cbmc
        cbmc_path = args.cbmc_path

        # Check if model_version contains JSON configuration
        try:
            configs = json.loads(model_version)
            if isinstance(configs, dict):
                model_version = configs.get("model_version", "base")
                provider = configs.get("provider", provider)
                model = configs.get("model", model)
                max_candidates = configs.get("max_candidates", max_candidates)
                max_rounds = configs.get("max_rounds", max_rounds)
                use_local_classifier = configs.get("use_local_classifier", use_local_classifier)
                target = configs.get("target", target)
                language = configs.get("language", language)
                enable_cbmc = configs.get("enable_cbmc", enable_cbmc)
                cbmc_path = configs.get("cbmc_path", cbmc_path)
        except (json.JSONDecodeError, TypeError):
            pass

        cfg = PipelineConfig(
            provider           = provider,
            model              = model,
            model_version      = model_version,
            max_t2_candidates  = max_candidates,
            max_critique_rounds= max_rounds,
            enable_proof       = args.enable_proof,
            enable_cbmc        = enable_cbmc,
            cbmc_path          = cbmc_path,
            enable_tsan        = (os.name != "nt"),   # auto-disabled on Windows
            proof_corpus_path  = args.proof_corpus,
            reward_events_path = args.reward_events,
            verbose            = True,                # logs go to stderr
            use_local_classifier = use_local_classifier,
            target             = target,
            language           = language,
            job_id             = payload.get("id", ""),
            source_file        = payload.get("file_path", ""),
        )

        try:
            result = run(source_code=source_code, func_name=func_name, config=cfg)
            print(json.dumps(asdict(result)))
            return 0 if result.success else 1
        except Exception as exc:
            print(json.dumps({"success": False, "score": 0.0, "error": str(exc)}))
            return 1

    # ------------------------------------------------------------------
    # Mode 2: --source FILE  (original CLI mode)
    # ------------------------------------------------------------------
    if not args.source:
        parser.error("Either --source or --json is required.")

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            source_code = f.read()
    except OSError as exc:
        print(json.dumps({"success": False, "score": 0.0, "error": str(exc)}))
        return 1

    cfg = PipelineConfig(
        provider           = args.provider,
        model              = args.model,
        max_t2_candidates  = args.max_candidates,
        max_critique_rounds= args.max_rounds,
        enable_proof       = args.enable_proof,
        enable_cbmc        = args.enable_cbmc,
        cbmc_path          = args.cbmc_path,
        enable_tsan        = args.enable_tsan and os.name != "nt",
        proof_corpus_path  = args.proof_corpus,
        reward_events_path = args.reward_events,
        source_file        = args.source,
        language           = args.language,
        verbose            = False,
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

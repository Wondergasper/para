"""
run_rl_loop.py
--------------
Orchestrates the Reinforcement Learning from Verification Feedback (RLVF) loop:
1. Reads successful proofs (score >= 0.8) from data/proofs.jsonl
2. Exports them as SFT dataset to data/corpus.jsonl
3. Spawns LoRA training using workers/training/train_lora.py
4. Generates an Ollama Modelfile linking the base model to the trained adapter
5. Registers the adapter as a new model in Ollama: `apg-finetuned:latest`
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

# Insert project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.training.export_corpus import export_local_corpus

def main():
    parser = argparse.ArgumentParser(description="Run closed-loop RLVF model training.")
    parser.add_argument("--proofs",     default="data/proofs.jsonl", help="Path to verifier proofs.jsonl")
    parser.add_argument("--corpus",     default="data/corpus.jsonl", help="SFT output corpus path")
    parser.add_argument("--output",     default="data/apg-adapter-v1", help="Adapter save directory")
    parser.add_argument("--base-model", default="qwen2.5-coder:1.5b", help="Ollama base model name for Modelfile")
    parser.add_argument("--model-key",  default="deepseek-1.3b",     help="Training model key (deepseek-1.3b | deepseek-6.7b)")
    parser.add_argument("--dry-run",    action="store_true",         help="Validate data and setup without running training")
    parser.add_argument("--min-score",  type=float, default=0.8,     help="Minimum verifier score to export (default 0.8)")
    
    args = parser.parse_args()

    print(f"\n==========================================")
    print(f"APG Closed-Loop RLVF Orchestrator")
    print(f"==========================================")

    # 1. Check if proofs file exists
    if not os.path.exists(args.proofs):
        print(f"[ERROR] Proofs file not found at: {args.proofs}")
        print(f"  Submit and verify some jobs first using the dashboard.")
        return 1

    # 2. Export local corpus
    print(f"\nStep 1: Exporting successful candidates (score >= {args.min_score}) to SFT format...")
    try:
        written = export_local_corpus(args.proofs, args.corpus, min_score=args.min_score, verbose=True)
        print(f"  [OK] Exported {written} records to {args.corpus}")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        return 1

    if written == 0:
        print("[WARN] No successful candidates found in proofs.jsonl. Cannot train yet.")
        return 0

    # 3. LoRA training
    print(f"\nStep 2: Starting LoRA Fine-Tuning...")
    training_cmd = [
        sys.executable,
        "workers/training/train_lora.py",
        "--corpus", args.corpus,
        "--output", args.output,
        "--model", args.model_key,
    ]
    if args.dry_run:
        training_cmd.append("--dry-run")
        print("  Running in DRY-RUN mode.")

    print(f"  Executing: {' '.join(training_cmd)}")
    ret = subprocess.run(training_cmd)
    if ret.returncode != 0:
        if args.dry_run:
            print("[WARN] LoRA training dependency check failed (likely missing packages: transformers, peft, trl, torch, etc.), but continuing dry-run.")
            os.makedirs(args.output, exist_ok=True)
        else:
            print("[ERROR] LoRA training failed.")
            return ret.returncode

    # 4. Generate Modelfile
    print(f"\nStep 3: Generating Ollama Modelfile...")
    modelfile_path = os.path.join(args.output, "Modelfile")
    adapter_path = os.path.abspath(args.output)
    
    if args.dry_run:
        content = f"# APG Fine-Tuned Model via RLVF (DRY RUN ALIAS)\nFROM {args.base_model}\n"
    else:
        content = f"""# APG Fine-Tuned Model via RLVF
FROM {args.base_model}

# Load the trained LoRA adapter
ADAPTER {adapter_path}
"""
    try:
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [OK] Modelfile created at {modelfile_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write Modelfile: {e}")
        return 1

    # 5. Register model in Ollama
    print(f"\nStep 4: Registering model in Ollama...")
    ollama_cmd = [
        "ollama",
        "create",
        "apg-finetuned:latest",
        "-f",
        modelfile_path
    ]
    print(f"  Executing: {' '.join(ollama_cmd)}")
    ret = subprocess.run(ollama_cmd)
    if ret.returncode != 0:
        print("[ERROR] Ollama model creation failed. Make sure Ollama is running and command is on PATH.")
        return ret.returncode

    print(f"\n==========================================")
    print(f"RLVF LOOP COMPLETE!")
    print(f"  Model 'apg-finetuned:latest' is now registered in Ollama.")
    print(f"  Select model version 'latest' in your dashboard to use it!")
    print(f"==========================================\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())

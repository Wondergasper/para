"""
auto_finetune.py
----------------
Automatically triggers Phase 4 LoRA fine-tuning when the APG proof corpus
reaches the required threshold (default: 500 verified entries).

What it does:
    1. Checks the current proof corpus size
    2. If >= threshold: exports corpus to JSONL, triggers LoRA training,
       runs A/B test, promotes model if A/B test passes
    3. Logs result to data/finetune_log.jsonl
    4. Optionally runs on a recurring schedule (--watch mode)

Usage:
    # Run once (manual trigger):
    python scripts/auto_finetune.py

    # Run once, force even if below threshold:
    python scripts/auto_finetune.py --force

    # Watch mode: check every N hours:
    python scripts/auto_finetune.py --watch --interval 6

Requirements:
    - PyTorch + transformers + peft + trl installed
    - GPU recommended (16 GB VRAM) or use --device cpu for small corpus
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

DEFAULT_CORPUS_PATH    = "data/proofs.jsonl"
DEFAULT_EXPORT_PATH    = "data/corpus_sft.jsonl"
DEFAULT_ADAPTERS_DIR   = "data"
DEFAULT_LOG_PATH       = "data/finetune_log.jsonl"
LORA_THRESHOLD          = 500


# ── Corpus helpers ─────────────────────────────────────────────────────────────

def count_verified(corpus_path: str, min_score: float = 0.8) -> int:
    """Count verified records in the local JSONL proof corpus."""
    count = 0
    if not os.path.exists(corpus_path):
        return 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("success") and r.get("score", 0) >= min_score:
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def get_next_adapter_version(adapters_dir: str) -> int:
    """Find the next adapter version number."""
    version = 1
    while os.path.exists(os.path.join(adapters_dir, f"apg-adapter-v{version}")):
        version += 1
    return version


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def run_export(corpus_path: str, export_path: str, min_score: float = 0.8) -> int:
    """Export the local corpus to SFT JSONL. Returns number of exported records."""
    print(f"  Exporting corpus to {export_path}...")
    cmd = [
        sys.executable, "-m", "workers.training.export_corpus",
        "--local-proof-corpus", corpus_path,
        "--output", export_path,
        "--min-score", str(min_score),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Export failed:\n{result.stderr}")

    # Count exported lines
    if os.path.exists(export_path):
        with open(export_path, "r") as f:
            return sum(1 for line in f if line.strip())
    return 0


def run_lora_training(corpus_path: str, adapter_output: str, device: str = "auto") -> str:
    """Run LoRA fine-tuning. Returns path to trained adapter."""
    print(f"  Starting LoRA training → {adapter_output}")
    cmd = [
        sys.executable, "-m", "workers.training.train_lora",
        "--corpus", corpus_path,
        "--output", adapter_output,
        "--device", device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LoRA training failed:\n{result.stderr[-2000:]}")
    print(f"  Training complete: {adapter_output}")
    return adapter_output


def run_ab_test(new_adapter: str, baseline_adapter: str = None) -> dict:
    """Run A/B test between new adapter and baseline. Returns verdict dict."""
    print(f"  Running A/B test...")
    cmd = [
        sys.executable, "-m", "workers.training.ab_test",
        "--new-adapter", new_adapter,
        "--benchmark", "data/benchmark_suite.jsonl",
    ]
    if baseline_adapter and os.path.exists(baseline_adapter):
        cmd += ["--baseline-adapter", baseline_adapter]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    # Parse verdict from output
    if "PROMOTE" in output:
        verdict = "PROMOTE"
    elif "KEEP_BASELINE" in output:
        verdict = "KEEP_BASELINE"
    else:
        verdict = "INCONCLUSIVE"

    return {"verdict": verdict, "output": output[-3000:]}


# ── Log helper ────────────────────────────────────────────────────────────────

def log_event(log_path: str, event: dict) -> None:
    event["timestamp"] = datetime.now(UTC).isoformat()
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ── Main finetune run ─────────────────────────────────────────────────────────

def run_finetune(
    corpus_path: str = DEFAULT_CORPUS_PATH,
    export_path: str = DEFAULT_EXPORT_PATH,
    adapters_dir: str = DEFAULT_ADAPTERS_DIR,
    log_path: str = DEFAULT_LOG_PATH,
    threshold: int = LORA_THRESHOLD,
    device: str = "auto",
    force: bool = False,
) -> bool:
    print(f"\n{'='*60}")
    print(f" APG Auto Fine-Tune Check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Step 1: Check corpus
    verified = count_verified(corpus_path)
    print(f"  Verified corpus entries: {verified} / {threshold} required")

    if verified < threshold and not force:
        print(f"  SKIP: corpus too small ({verified} < {threshold})")
        print(f"        Run: python scripts/bootstrap_corpus.py")
        log_event(log_path, {"action": "skip", "reason": "corpus_too_small", "verified": verified})
        return False

    if force and verified < threshold:
        print(f"  FORCE mode: proceeding despite small corpus ({verified} entries)")

    # Step 2: Export
    exported = run_export(corpus_path, export_path)
    print(f"  Exported {exported} records to {export_path}")

    # Step 3: Train
    version = get_next_adapter_version(adapters_dir)
    adapter_path = os.path.join(adapters_dir, f"apg-adapter-v{version}")
    baseline_path = os.path.join(adapters_dir, f"apg-adapter-v{version - 1}") if version > 1 else None

    try:
        run_lora_training(export_path, adapter_path, device)
    except RuntimeError as e:
        print(f"  ERROR during training: {e}")
        log_event(log_path, {"action": "training_failed", "error": str(e)})
        return False

    # Step 4: A/B test
    ab_result = run_ab_test(adapter_path, baseline_path)
    verdict = ab_result["verdict"]
    print(f"  A/B verdict: {verdict}")

    # Step 5: Record result
    log_entry = {
        "action": "finetune_complete",
        "version": version,
        "adapter_path": adapter_path,
        "verified_entries": verified,
        "exported_entries": exported,
        "ab_verdict": verdict,
    }
    log_event(log_path, log_entry)

    if verdict == "PROMOTE":
        print(f"  \u2713 PROMOTED: adapter v{version} is now the active model.")
        # Write current adapter path to a sentinel file for the pipeline to pick up
        sentinel = os.path.join(adapters_dir, "active_adapter.txt")
        with open(sentinel, "w") as f:
            f.write(adapter_path + "\n")
        print(f"  Active adapter written to: {sentinel}")
    else:
        print(f"  Adapter v{version} NOT promoted ({verdict}). Baseline retained.")

    print(f"{'='*60}\n")
    return verdict == "PROMOTE"


# ── Watch mode ────────────────────────────────────────────────────────────────

def watch_mode(interval_hours: float, **kwargs) -> None:
    interval_secs = interval_hours * 3600
    print(f"Watch mode: checking every {interval_hours}h")
    while True:
        run_finetune(**kwargs)
        print(f"  Next check in {interval_hours}h. Sleeping...")
        time.sleep(interval_secs)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG Auto Fine-Tune Trigger")
    parser.add_argument("--corpus",    default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--export",    default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--adapters",  default=DEFAULT_ADAPTERS_DIR)
    parser.add_argument("--log",       default=DEFAULT_LOG_PATH)
    parser.add_argument("--threshold", type=int,   default=LORA_THRESHOLD)
    parser.add_argument("--device",    default="auto",
                        help="PyTorch device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--force",     action="store_true",
                        help="Run even if corpus is below threshold")
    parser.add_argument("--watch",     action="store_true",
                        help="Run in watch mode (repeat on interval)")
    parser.add_argument("--interval",  type=float, default=6.0,
                        help="Check interval in hours (watch mode only)")
    args = parser.parse_args()

    kwargs = dict(
        corpus_path=args.corpus,
        export_path=args.export,
        adapters_dir=args.adapters,
        log_path=args.log,
        threshold=args.threshold,
        device=args.device,
        force=args.force,
    )
    if args.watch:
        watch_mode(args.interval, **kwargs)
    else:
        success = run_finetune(**kwargs)
        sys.exit(0 if success else 1)

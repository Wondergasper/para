"""
corpus_report.py
----------------
Prints a summary of the APG proof corpus from data/proofs.jsonl.
Shows success rate, verifier distribution, parallelism type breakdown,
unique functions, and how many more entries are needed for Phase 4 LoRA.

Usage:
    python scripts/corpus_report.py
    python scripts/corpus_report.py --corpus data/proofs.jsonl --threshold 500
"""
import argparse
import json
import os
from collections import Counter

LORA_THRESHOLD = 500


def load_corpus(path: str) -> list[dict]:
    records = []
    if not os.path.exists(path):
        print(f"Corpus file not found: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def corpus_report(corpus_path: str = "data/proofs.jsonl", threshold: int = LORA_THRESHOLD):
    records = load_corpus(corpus_path)
    total = len(records)

    print(f"\n{'='*60}")
    print(f" APG Proof Corpus Report")
    print(f"{'='*60}")
    print(f" File:         {corpus_path}")
    print(f" Total entries: {total}")

    if total == 0:
        print(" No records found.")
        return

    # Success / failure split
    successes = [r for r in records if r.get("success") and r.get("score", 0) >= 0.8]
    failures = [r for r in records if not r.get("success") or r.get("score", 0) < 0.8]
    print(f"\n Success (score >= 0.8): {len(successes)} ({len(successes)/total*100:.1f}%)")
    print(f" Failures / low score:  {len(failures)} ({len(failures)/total*100:.1f}%)")

    # Score distribution
    score_buckets = Counter()
    for r in records:
        sc = r.get("score", 0)
        if sc >= 1.0:   score_buckets["1.0"] += 1
        elif sc >= 0.8: score_buckets["0.8-1.0"] += 1
        elif sc >= 0.6: score_buckets["0.6-0.8"] += 1
        elif sc >= 0.2: score_buckets["0.2-0.6"] += 1
        else:           score_buckets["0.0"] += 1
    print(f"\n Score distribution:")
    for bucket, count in sorted(score_buckets.items()):
        print(f"   {bucket:>10}: {count}")

    # Verifier distribution
    verifiers = Counter(r.get("verifier", "unknown") for r in records)
    print(f"\n By verifier:")
    for v, c in verifiers.most_common():
        print(f"   {v:>20}: {c}")

    # Parallelism type distribution
    types = Counter(r.get("annotated_ir", {}).get("type", "unknown") for r in records)
    print(f"\n By parallelism type:")
    for t, c in types.most_common():
        print(f"   {t:>20}: {c}")

    # Unique functions
    func_names = Counter(r.get("func_name", "unknown") for r in successes)
    print(f"\n Unique successful functions: {len(func_names)}")
    print(f" Top 10 functions:")
    for fn, c in func_names.most_common(10):
        print(f"   {fn:>30}: {c} successful run(s)")

    # Phase 4 readiness
    needed = max(0, threshold - len(successes))
    print(f"\n{'='*60}")
    print(f" Phase 4 LoRA threshold: {threshold} verified entries")
    print(f" Current verified:       {len(successes)}")
    if needed == 0:
        print(f" STATUS: \u2713 READY for Phase 4 LoRA fine-tuning!")
    else:
        print(f" STATUS: Need {needed} more verified entries before LoRA trigger.")
        print(f"         Run: python scripts/bootstrap_corpus.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG Corpus Report")
    parser.add_argument("--corpus", default="data/proofs.jsonl")
    parser.add_argument("--threshold", type=int, default=LORA_THRESHOLD)
    args = parser.parse_args()
    corpus_report(args.corpus, args.threshold)

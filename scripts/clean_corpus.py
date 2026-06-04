"""
clean_corpus.py
---------------
Cleans the APG proof corpus by removing failed entries and de-duplicating.

What it removes:
    - Entries with score < 0.5
    - Entries with success=False
    - Duplicates: keeps only the highest-scoring entry per (source_code, func_name)

Usage:
    python scripts/clean_corpus.py
    python scripts/clean_corpus.py --input data/proofs.jsonl --output data/proofs_clean.jsonl
    python scripts/clean_corpus.py --in-place  # overwrites original
"""
import argparse
import json
import os
import shutil


def clean_corpus(
    input_path: str = "data/proofs.jsonl",
    output_path: str = "data/proofs_clean.jsonl",
    min_score: float = 0.5,
    in_place: bool = False,
) -> tuple[int, int, int]:
    """
    Clean the corpus. Returns (total_read, kept, dropped).
    """
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return 0, 0, 0

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total = len(records)

    # Filter: keep only entries with score >= min_score and success=True
    filtered = [
        r for r in records
        if r.get("success", False) and r.get("score", 0.0) >= min_score
    ]

    # De-duplicate: keep highest score per (source_code, func_name)
    best: dict[tuple, dict] = {}
    for r in filtered:
        key = (r.get("source_code", ""), r.get("func_name", ""))
        if key not in best or r.get("score", 0) > best[key].get("score", 0):
            best[key] = r

    kept_records = list(best.values())
    kept = len(kept_records)
    dropped = total - kept

    # Write output
    dest = input_path if in_place else output_path
    if in_place:
        # Backup first
        backup = input_path + ".bak"
        shutil.copy2(input_path, backup)
        print(f" Backup saved to: {backup}")

    with open(dest, "w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    return total, kept, dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean APG proof corpus")
    parser.add_argument("--input",     default="data/proofs.jsonl")
    parser.add_argument("--output",    default="data/proofs_clean.jsonl")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--in-place",  action="store_true",
                        help="Overwrite input file (creates .bak backup first)")
    args = parser.parse_args()

    print(f"Cleaning corpus: {args.input}")
    total, kept, dropped = clean_corpus(
        input_path=args.input,
        output_path=args.output,
        min_score=args.min_score,
        in_place=args.in_place,
    )
    dest = args.input if args.in_place else args.output
    print(f" Total read:  {total}")
    print(f" Kept:        {kept}")
    print(f" Dropped:     {dropped}")
    print(f" Written to:  {dest}")

    if kept < 500:
        print(f"\n WARNING: {kept} clean entries < 500 threshold for LoRA.")
        print( "         Run: python scripts/bootstrap_corpus.py to grow corpus.")
    else:
        print(f"\n READY: {kept} entries >= 500. Ready for Phase 4.")

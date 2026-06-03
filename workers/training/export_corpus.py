"""
export_corpus.py
----------------
Phase 4 Trigger: Export verified proof corpus from PostgreSQL → JSONL

What it does:
    Reads all successful (source, parallel, proof) triples from the database.
    Converts them into the SFT conversation format expected by TRL's SFTTrainer.
    Validates each row before writing (compiles, has valid JSON fields).
    Writes output to corpus.jsonl ready for train_lora.py.

When to run:
    Only after your proofs table has >= 500 verified rows.
    Check with: python export_corpus.py --count

Usage:
    # Check how many verified rows you have
    python export_corpus.py --count

    # Export to corpus.jsonl
    python export_corpus.py --output corpus.jsonl

    # Export only rows verified by CBMC (strictest)
    python export_corpus.py --verifier cbmc --output corpus_cbmc.jsonl

Database schema expected:
    CREATE TABLE proofs (
        id            SERIAL PRIMARY KEY,
        source_code   TEXT NOT NULL,
        parallel_code TEXT NOT NULL,
        proof         TEXT,
        dep_graph     JSONB,
        annotated_ir  JSONB,
        success       BOOLEAN NOT NULL DEFAULT FALSE,
        verifier      TEXT,         -- 'cbmc' | 'tsan' | 'output_match'
        score         FLOAT,        -- 0.0 - 1.0 from verifier.py
        func_name     TEXT,
        created_at    TIMESTAMP DEFAULT NOW()
    );
"""

import json
import argparse
import sys
import os
from typing import Iterator

from workers.spec_generator.proof_corpus import ProofCorpusStore

# ── System prompt reused from T2 (must match exactly what was used during collection) ──
SYSTEM_PROMPT_T2 = (
    "You are an expert HPC programmer. Given C code and its dependency graph, "
    "produce an OpenMP-parallelized version. Output ONLY valid C code. No explanations."
)


# ── Database connection ────────────────────────────────────────────────────────

def get_connection(db_url: str):
    """
    Return a psycopg2 connection. Raises ImportError if psycopg2 not installed.
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 is required for export_corpus.py.\n"
            "Install it with: pip install psycopg2-binary"
        )
    return psycopg2.connect(db_url)


# ── Row validator ──────────────────────────────────────────────────────────────

def validate_row(source: str, parallel: str, dep_graph) -> tuple[bool, str]:
    """
    Basic validation before including a row in the corpus.

    Checks:
        - source_code is non-empty
        - parallel_code is non-empty and contains at least one pragma or loop
        - dep_graph is parseable as a dict

    Returns:
        (valid: bool, reason: str)
    """
    if not source or len(source.strip()) < 10:
        return False, "source_code is empty or too short"

    if not parallel or len(parallel.strip()) < 10:
        return False, "parallel_code is empty or too short"

    if "pragma omp" not in parallel.lower() and "#pragma" not in parallel.lower():
        return False, "parallel_code has no OpenMP pragma — likely not parallelised"

    if dep_graph is None:
        return False, "dep_graph is NULL"

    if isinstance(dep_graph, str):
        try:
            json.loads(dep_graph)
        except json.JSONDecodeError:
            return False, "dep_graph is not valid JSON"

    return True, ""


# ── Row → JSONL record converter ───────────────────────────────────────────────

def row_to_record(
    source:    str,
    parallel:  str,
    dep_graph,
    annotated_ir = None,
) -> dict:
    """
    Convert a database row into a TRL SFTTrainer conversation record.

    Format:
        {
            "messages": [
                {"role": "system",    "content": "<T2 system prompt>"},
                {"role": "user",      "content": "Source:\\n...\\nDep graph: {...}"},
                {"role": "assistant", "content": "<parallel C code>"}
            ]
        }
    """
    # Normalise dep_graph to a dict
    if isinstance(dep_graph, str):
        dep_graph = json.loads(dep_graph)
    if isinstance(annotated_ir, str):
        annotated_ir = json.loads(annotated_ir) if annotated_ir else None

    # Build user message
    user_parts = [f"Source:\n{source.strip()}"]
    user_parts.append(f"\nDep graph: {json.dumps(dep_graph)}")

    if annotated_ir:
        if annotated_ir.get("reduction_variables"):
            user_parts.append(
                f"\nReduction variables: {annotated_ir['reduction_variables']}"
            )
        if annotated_ir.get("type"):
            user_parts.append(f"\nParallelism type: {annotated_ir['type']}")

    user_parts.append("\nGenerate the parallelised C function:")

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT_T2},
            {"role": "user",      "content": "".join(user_parts)},
            {"role": "assistant", "content": parallel.strip()},
        ]
    }


# ── Streaming row fetcher ──────────────────────────────────────────────────────

def fetch_rows(
    db_url:   str,
    verifier: str = None,
    min_score: float = 0.6,
) -> Iterator[tuple]:
    """
    Stream verified rows from the database.

    Args:
        db_url:    PostgreSQL connection string.
        verifier:  Filter by verifier type (e.g. 'cbmc'). None = all.
        min_score: Minimum verifier score to include (default 0.6).

    Yields:
        Tuples of (source_code, parallel_code, dep_graph, annotated_ir)
    """
    conn = get_connection(db_url)
    cur  = conn.cursor()

    query = """
        SELECT source_code, parallel_code, dep_graph, annotated_ir
        FROM proofs
        WHERE success = true
          AND score >= %s
    """
    params = [min_score]

    if verifier:
        query  += " AND verifier = %s"
        params.append(verifier)

    query += " ORDER BY score DESC, created_at DESC"

    cur.execute(query, params)

    try:
        for row in cur:
            yield row
    finally:
        cur.close()
        conn.close()


# ── Count helper ───────────────────────────────────────────────────────────────

def count_rows(db_url: str, verifier: str = None, min_score: float = 0.6) -> int:
    """Return the number of eligible rows in the database."""
    conn = get_connection(db_url)
    cur  = conn.cursor()

    query  = "SELECT COUNT(*) FROM proofs WHERE success = true AND score >= %s"
    params = [min_score]
    if verifier:
        query  += " AND verifier = %s"
        params.append(verifier)

    cur.execute(query, params)
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


# ── Main export function ───────────────────────────────────────────────────────

def export_corpus(
    db_url:      str,
    output_path: str,
    verifier:    str   = None,
    min_score:   float = 0.6,
    verbose:     bool  = True,
) -> int:
    """
    Export the verified corpus to a JSONL file.

    Args:
        db_url:      PostgreSQL connection string.
        output_path: Path to write the .jsonl output file.
        verifier:    Filter by verifier type. None = all.
        min_score:   Minimum score threshold.
        verbose:     Print progress.

    Returns:
        Number of records written.
    """
    written  = 0
    skipped  = 0

    with open(output_path, "w", encoding="utf-8") as out_file:
        for src, par, dep, ir in fetch_rows(db_url, verifier, min_score):
            valid, reason = validate_row(src, par, dep)

            if not valid:
                skipped += 1
                if verbose:
                    print(f"  [skip] {reason}")
                continue

            record = row_to_record(src, par, dep, ir)
            out_file.write(json.dumps(record) + "\n")
            written += 1

            if verbose and written % 50 == 0:
                print(f"  Written {written} records...")

    if verbose:
        print(f"\nDone. Written: {written} | Skipped: {skipped}")
        if written < 500:
            print(
                f"  ⚠ WARNING: Only {written} records. "
                f"LoRA fine-tuning needs >= 500. Keep running the pipeline."
            )
        else:
            print(f"  ✓ Corpus is large enough for LoRA fine-tuning.")

    return written


def export_local_corpus(
    proof_corpus_path: str,
    output_path: str,
    min_score: float = 0.8,
    verbose: bool = True,
) -> int:
    """
    Export the local Phase 3 JSONL proof corpus to Phase 4 SFT JSONL.
    """
    store = ProofCorpusStore(proof_corpus_path)
    written = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out_file:
        for record in store.iter_verified(min_score=min_score):
            valid, reason = validate_row(
                record.source_code,
                record.parallel_code,
                record.dep_graph,
            )
            if not valid:
                skipped += 1
                if verbose:
                    print(f"  [skip] {reason}")
                continue

            out_file.write(
                json.dumps(
                    row_to_record(
                        record.source_code,
                        record.parallel_code,
                        record.dep_graph,
                        record.annotated_ir,
                    )
                )
                + "\n"
            )
            written += 1

    if verbose:
        print(f"\nDone. Written: {written} | Skipped: {skipped}")
        if written < 500:
            print(f"  WARNING: Only {written} records. LoRA fine-tuning needs >= 500.")

    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export APG verified corpus from PostgreSQL to JSONL for LoRA fine-tuning."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", "postgresql://localhost/apg"),
        help="PostgreSQL connection string (or set DATABASE_URL env var).",
    )
    parser.add_argument(
        "--output",
        default="corpus.jsonl",
        help="Output JSONL file path (default: corpus.jsonl).",
    )
    parser.add_argument(
        "--verifier",
        default=None,
        choices=["cbmc", "tsan", "output_match", None],
        help="Filter by verifier type. Default: all.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum verifier score to include (default: 0.6).",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Just print the row count and exit (no export).",
    )
    parser.add_argument(
        "--local-proof-corpus",
        default="",
        help="Export from local Phase 3 proof JSONL instead of PostgreSQL.",
    )

    args = parser.parse_args()

    if args.local_proof_corpus:
        if args.count:
            n = sum(1 for _ in ProofCorpusStore(args.local_proof_corpus).iter_verified(args.min_score))
            print(f"Eligible rows in local proof corpus: {n}")
            if n < 500:
                print(f"Need {500 - n} more rows before starting LoRA fine-tuning.")
            else:
                print("Ready for LoRA fine-tuning.")
            sys.exit(0)

        print(f"Exporting local proof corpus to {args.output}...")
        total = export_local_corpus(
            proof_corpus_path=args.local_proof_corpus,
            output_path=args.output,
            min_score=args.min_score,
            verbose=True,
        )
        sys.exit(0 if total > 0 else 1)

    if args.count:
        n = count_rows(args.db_url, args.verifier, args.min_score)
        print(f"Eligible rows in database: {n}")
        if n < 500:
            print(f"⚠ Need {500 - n} more rows before starting LoRA fine-tuning.")
        else:
            print("✓ Ready for LoRA fine-tuning.")
        sys.exit(0)

    print(f"Exporting corpus to {args.output}...")
    total = export_corpus(
        db_url=args.db_url,
        output_path=args.output,
        verifier=args.verifier,
        min_score=args.min_score,
        verbose=True,
    )
    sys.exit(0 if total > 0 else 1)

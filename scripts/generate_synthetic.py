"""
generate_synthetic.py
---------------------
Generates synthetic C functions and submits them to the APG pipeline to grow
the proof corpus toward the 500-entry LoRA training threshold.

What it generates:
    - Vector operations: scale, add, subtract, multiply element-wise
    - Reduction operations: sum, product, min, max, dot-product
    - Matrix operations: transpose, add, scale
    - Stencil operations: 1D averaging, prefix-sum variants

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --api http://localhost:8080 --count 200
"""
import argparse
import itertools
import json
import time
import urllib.request
from string import ascii_lowercase


# ── Template generators ────────────────────────────────────────────────────────

def gen_vector_scale(dtype: str, N: int, var_suffix: str = "") -> tuple[str, str]:
    fn = f"vector_scale{var_suffix}"
    code = f"""\
void {fn}({dtype}* A, {dtype} s, int N) {{
    for (int i = 0; i < N; i++) {{
        A[i] = A[i] * s;
    }}
}}"""
    return fn, code


def gen_vector_add(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"vector_add{var_suffix}"
    code = f"""\
void {fn}({dtype}* A, {dtype}* B, {dtype}* C, int N) {{
    for (int i = 0; i < N; i++) {{
        C[i] = A[i] + B[i];
    }}
}}"""
    return fn, code


def gen_reduction_sum(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"array_sum{var_suffix}"
    code = f"""\
{dtype} {fn}({dtype}* A, int N) {{
    {dtype} s = 0;
    for (int i = 0; i < N; i++) {{
        s += A[i];
    }}
    return s;
}}"""
    return fn, code


def gen_dot_product(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"dot_product{var_suffix}"
    code = f"""\
{dtype} {fn}({dtype}* A, {dtype}* B, int N) {{
    {dtype} s = 0;
    for (int i = 0; i < N; i++) {{
        s += A[i] * B[i];
    }}
    return s;
}}"""
    return fn, code


def gen_vector_subtract(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"vector_sub{var_suffix}"
    code = f"""\
void {fn}({dtype}* A, {dtype}* B, {dtype}* C, int N) {{
    for (int i = 0; i < N; i++) {{
        C[i] = A[i] - B[i];
    }}
}}"""
    return fn, code


def gen_vector_mul(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"vector_mul{var_suffix}"
    code = f"""\
void {fn}({dtype}* A, {dtype}* B, {dtype}* C, int N) {{
    for (int i = 0; i < N; i++) {{
        C[i] = A[i] * B[i];
    }}
}}"""
    return fn, code


def gen_array_max(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"array_max{var_suffix}"
    code = f"""\
{dtype} {fn}({dtype}* A, int N) {{
    {dtype} m = A[0];
    for (int i = 1; i < N; i++) {{
        if (A[i] > m) m = A[i];
    }}
    return m;
}}"""
    return fn, code


def gen_matrix_scale(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"matrix_scale{var_suffix}"
    code = f"""\
void {fn}({dtype}* A, {dtype} s, int rows, int cols) {{
    for (int i = 0; i < rows; i++) {{
        for (int j = 0; j < cols; j++) {{
            A[i * cols + j] *= s;
        }}
    }}
}}"""
    return fn, code


def gen_matrix_add(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"matrix_add{var_suffix}"
    code = f"""\
void {fn}({dtype}* A, {dtype}* B, {dtype}* C, int rows, int cols) {{
    for (int i = 0; i < rows; i++) {{
        for (int j = 0; j < cols; j++) {{
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
        }}
    }}
}}"""
    return fn, code


def gen_stencil_1d(dtype: str, var_suffix: str = "") -> tuple[str, str]:
    fn = f"stencil_1d{var_suffix}"
    code = f"""\
void {fn}({dtype}* A, {dtype}* B, int N) {{
    for (int i = 1; i < N - 1; i++) {{
        B[i] = 0.333 * (A[i-1] + A[i] + A[i+1]);
    }}
}}"""
    return fn, code


# ── All generators × data types ────────────────────────────────────────────────

GENERATORS = [
    gen_vector_scale,
    gen_vector_add,
    gen_reduction_sum,
    gen_dot_product,
    gen_vector_subtract,
    gen_vector_mul,
    gen_array_max,
    gen_matrix_scale,
    gen_matrix_add,
    gen_stencil_1d,
]

DTYPES = ["float", "double", "int"]


def build_all_variants(count: int) -> list[tuple[str, str]]:
    """Generate up to `count` unique (func_name, source_code) pairs."""
    variants = []
    suffix_iter = itertools.count(1)

    for gen in GENERATORS:
        for dtype in DTYPES:
            suffix = f"_{dtype[0]}{next(suffix_iter)}"
            try:
                fn, code = gen(dtype, var_suffix=suffix)
                variants.append((fn, code))
                if len(variants) >= count:
                    return variants
            except Exception:
                continue

    # If we need more, cycle through generators with more suffixes
    for gen, dtype in itertools.product(GENERATORS, DTYPES):
        if len(variants) >= count:
            break
        suffix = f"_v{next(suffix_iter)}"
        try:
            fn, code = gen(dtype, var_suffix=suffix)
            if code not in {c for _, c in variants}:
                variants.append((fn, code))
        except Exception:
            continue

    return variants[:count]


# ── API helpers ────────────────────────────────────────────────────────────────

def submit_to_apg(api_url: str, source_code: str, func_name: str) -> str | None:
    payload = json.dumps({"source": source_code, "language": "c", "func_name": func_name}).encode()
    try:
        req = urllib.request.Request(
            f"{api_url}/submit", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("id")
    except Exception as e:
        print(f"  [WARN] Submit failed for {func_name}: {e}")
        return None


def poll_result(api_url: str, job_id: str, timeout: int = 180) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{api_url}/result/{job_id}", timeout=10) as resp:
                body = json.loads(resp.read())
                if body.get("status") != "pending":
                    return body
        except Exception:
            pass
        time.sleep(5)
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_synthetic(
    api_url: str = "http://localhost:8080",
    count: int = 200,
    batch_size: int = 4,
    delay_between_batches: float = 2.0,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    variants = build_all_variants(count)
    print(f"APG Synthetic Corpus Generator")
    print(f"  API:       {api_url}")
    print(f"  Variants:  {len(variants)}")
    print(f"  Batch:     {batch_size}")
    print()

    successes = 0
    processed = 0

    for i in range(0, len(variants), batch_size):
        batch = variants[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as ex:
            futures = {
                ex.submit(submit_to_apg, api_url, code, fn): (fn, code)
                for fn, code in batch
            }
            job_ids = {}
            for future in as_completed(futures):
                fn, code = futures[future]
                jid = future.result()
                if jid:
                    job_ids[jid] = fn

        for jid, fn in job_ids.items():
            result = poll_result(api_url, jid)
            processed += 1
            if result and result.get("score", 0) >= 0.8:
                successes += 1
                print(f"  \u2713 {fn:40s} score={result.get('score')}")
            else:
                score = result.get("score", "N/A") if result else "timeout"
                print(f"  \u2717 {fn:40s} score={score}")

        time.sleep(delay_between_batches)

    print(f"\nDone. {successes}/{processed} variants parallelized successfully.")
    print("Run 'python scripts/corpus_report.py' to see updated corpus status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG Synthetic Corpus Generator")
    parser.add_argument("--api",        default="http://localhost:8080")
    parser.add_argument("--count",      type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--delay",      type=float, default=2.0,
                        help="Seconds between batches (avoid overwhelming Ollama)")
    args = parser.parse_args()
    generate_synthetic(args.api, args.count, args.batch_size, args.delay)

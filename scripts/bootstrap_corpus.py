"""
bootstrap_corpus.py
-------------------
Bootstraps the APG proof corpus using PolybenchC kernels downloaded from the web
and run through the local APG pipeline.

What it does:
    1. Downloads PolybenchC 4.2.1 kernel C files from the PolybenchC mirror
    2. Extracts the kernel function from each file (the function after the pragma loops)
    3. Submits each kernel to the APG pipeline via HTTP POST to /submit
    4. Polls /result/{id} until done
    5. Reports how many succeeded and how many are now in proofs.jsonl

Usage:
    # Start the APG orchestrator first, then:
    python scripts/bootstrap_corpus.py
    python scripts/bootstrap_corpus.py --api http://localhost:8080 --batch-size 4
    python scripts/bootstrap_corpus.py --no-download  # if kernels already extracted

Requirements:
    - APG Go orchestrator running (default: http://localhost:8080)
    - Ollama running with DeepSeek-Coder model loaded
    - GCC with OpenMP installed
"""
import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# PolybenchC 4.2 kernel function names and their category
# These are the canonical parallelizable kernels from the suite
POLYBENCH_KERNELS = [
    # Linear Algebra Kernels
    ("2mm",        "2mm",         "linear-algebra/kernels/2mm"),
    ("3mm",        "3mm",         "linear-algebra/kernels/3mm"),
    ("atax",       "atax",        "linear-algebra/kernels/atax"),
    ("bicg",       "bicg",        "linear-algebra/kernels/bicg"),
    ("doitgen",    "doitgen",     "linear-algebra/kernels/doitgen"),
    ("mvt",        "mvt",         "linear-algebra/kernels/mvt"),
    # Linear Algebra Solvers
    ("cholesky",   "cholesky",    "linear-algebra/solvers/cholesky"),
    ("durbin",     "durbin",      "linear-algebra/solvers/durbin"),
    ("gramschmidt","gramschmidt", "linear-algebra/solvers/gramschmidt"),
    ("lu",         "lu",          "linear-algebra/solvers/lu"),
    ("ludcmp",     "ludcmp",      "linear-algebra/solvers/ludcmp"),
    ("trisolv",    "trisolv",     "linear-algebra/solvers/trisolv"),
    # Linear Algebra BLAS
    ("gemm",       "gemm",        "linear-algebra/blas/gemm"),
    ("gemver",     "gemver",      "linear-algebra/blas/gemver"),
    ("gesummv",    "gesummv",     "linear-algebra/blas/gesummv"),
    ("symm",       "symm",        "linear-algebra/blas/symm"),
    ("syr2k",      "syr2k",       "linear-algebra/blas/syr2k"),
    ("syrk",       "syrk",        "linear-algebra/blas/syrk"),
    ("trmm",       "trmm",        "linear-algebra/blas/trmm"),
    # Stencils
    ("adi",        "adi",         "stencils/adi"),
    ("fdtd-2d",    "fdtd_2d",     "stencils/fdtd-2d"),
    ("heat-3d",    "heat_3d",     "stencils/heat-3d"),
    ("jacobi-1d",  "jacobi_1d",   "stencils/jacobi-1d"),
    ("jacobi-2d",  "jacobi_2d",   "stencils/jacobi-2d"),
    ("seidel-2d",  "seidel_2d",   "stencils/seidel-2d"),
    # Data mining
    ("covariance", "covariance",  "datamining/covariance"),
    ("correlation","correlation", "datamining/correlation"),
    # Medley
    ("deriche",    "deriche",     "medley/deriche"),
    ("floyd-warshall", "floyd_warshall", "medley/floyd-warshall"),
    ("nussinov",   "nussinov",    "medley/nussinov"),
]

POLYBENCH_GITHUB_RAW = "https://raw.githubusercontent.com/MatthiasJReisinger/PolyBenchC-4.2.1/master"


def download_kernel(name: str, func_name: str, path: str, kernels_dir: str) -> str | None:
    """
    Download a PolybenchC kernel .c file from GitHub mirror.
    Returns the local file path, or None on failure.
    """
    url = f"{POLYBENCH_GITHUB_RAW}/{path}/{name}.c"
    local_path = os.path.join(kernels_dir, f"{name}.c")

    if os.path.exists(local_path):
        return local_path

    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"  [WARN] Could not download {name}: {e}")
        return None


def extract_kernel_function(c_file: str, func_name: str) -> str | None:
    """
    Extract the kernel function from a PolybenchC .c file.
    PolybenchC kernels always have a function named `kernel_<name>` or the func_name.
    We look for it and extract the full function body.
    """
    with open(c_file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Remove #pragma scop / #pragma endscop markers
    content = re.sub(r'#pragma scop', '', content)
    content = re.sub(r'#pragma endscop', '', content)

    # Try to find the kernel function
    patterns = [
        rf'void\s+kernel_{re.escape(func_name)}\s*\([^)]*\)\s*{{',
        rf'void\s+{re.escape(func_name)}\s*\([^)]*\)\s*{{',
        r'static void\s+kernel_\w+\s*\([^)]*\)\s*{{',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            start = match.start()
            # Find matching closing brace
            brace_count = 0
            end = -1
            for i in range(match.end() - 1, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            if end != -1:
                return content[start:end]

    return None


def submit_to_apg(api_url: str, source_code: str, func_name: str, language: str = "c") -> str | None:
    """Submit a job to the APG API. Returns job ID or None on failure."""
    payload = json.dumps({
        "source": source_code,
        "language": language,
        "func_name": func_name,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{api_url}/submit",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            return body.get("id")
    except Exception as e:
        print(f"  [WARN] Submit failed for {func_name}: {e}")
        return None


def poll_result(api_url: str, job_id: str, timeout: int = 300) -> dict | None:
    """Poll /result/{id} until status is not 'pending', or timeout."""
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


def process_kernel(
    name: str, func_name: str, path: str,
    kernels_dir: str, api_url: str, no_download: bool
) -> dict:
    """Download, extract, submit, and poll one kernel. Returns result summary."""
    c_file = os.path.join(kernels_dir, f"{name}.c")

    if not no_download:
        c_file = download_kernel(name, func_name, path, kernels_dir)
        if not c_file:
            return {"name": name, "status": "download_failed"}

    if not os.path.exists(c_file):
        return {"name": name, "status": "file_not_found"}

    source = extract_kernel_function(c_file, func_name)
    if not source:
        return {"name": name, "status": "extraction_failed"}

    job_id = submit_to_apg(api_url, source, func_name)
    if not job_id:
        return {"name": name, "status": "submit_failed"}

    result = poll_result(api_url, job_id)
    if not result:
        return {"name": name, "status": "timeout"}

    score = result.get("score", 0)
    status = "success" if score >= 0.8 else "low_score"
    return {"name": name, "func_name": func_name, "status": status, "score": score, "job_id": job_id}


def bootstrap(
    api_url: str = "http://localhost:8080",
    batch_size: int = 4,
    kernels_dir: str = "data/polybench_kernels",
    no_download: bool = False,
    max_kernels: int = None,
) -> None:
    os.makedirs(kernels_dir, exist_ok=True)
    kernels = POLYBENCH_KERNELS[:max_kernels] if max_kernels else POLYBENCH_KERNELS

    print(f"APG Corpus Bootstrapper")
    print(f"  API:        {api_url}")
    print(f"  Kernels:    {len(kernels)} PolybenchC kernels")
    print(f"  Batch size: {batch_size}")
    print(f"  Output dir: {kernels_dir}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(
                process_kernel, name, func_name, path,
                kernels_dir, api_url, no_download
            ): name
            for name, func_name, path in kernels
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            icon = "\u2713" if r["status"] == "success" else "\u2717"
            score_str = f" (score={r.get('score', 'N/A')})" if "score" in r else ""
            print(f"  {icon} {r['name']:25s} {r['status']}{score_str}")

    successes = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone. {successes}/{len(results)} kernels parallelized successfully.")
    print(f"Run 'python scripts/corpus_report.py' to check current corpus size.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG Corpus Bootstrapper (PolybenchC)")
    parser.add_argument("--api",         default="http://localhost:8080")
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--kernels-dir", default="data/polybench_kernels")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download (use already-extracted kernels in kernels-dir)")
    parser.add_argument("--max-kernels", type=int, default=None,
                        help="Limit to N kernels (for testing)")
    args = parser.parse_args()
    bootstrap(
        api_url=args.api,
        batch_size=args.batch_size,
        kernels_dir=args.kernels_dir,
        no_download=args.no_download,
        max_kernels=args.max_kernels,
    )

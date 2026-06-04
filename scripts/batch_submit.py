import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.lang import get_driver

API_BASE = os.getenv("APG_API_BASE", "http://localhost:8080")

def submit_job(func_name: str, source_code: str, language: str, provider: str, model: str, target: str, enable_cbmc: bool = False) -> str:
    url = f"{API_BASE}/submit"
    # Format of model_version is used to pass full config in local queue
    cfg = {
        "model_version": "base",
        "provider": provider,
        "model": model,
        "max_candidates": 2,
        "max_rounds": 2,
        "target": target,
        "language": language,
        "use_local_classifier": False,
        "enable_cbmc": enable_cbmc
    }
    payload = {
        "source": source_code,
        "provider": provider,
        "model": model,
        "max_candidates": 2,
        "max_rounds": 2,
        "target": target,
        "use_local_classifier": False,
        "enable_cbmc": enable_cbmc
    }
    # Pass configuration string encoded in model field or using json submit request
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["id"]
    except urllib.error.URLError as e:
        print(f"Error submitting {func_name}: {e}")
        return None

def poll_job(job_id: str, timeout: int = 180) -> dict:
    url = f"{API_BASE}/result/{job_id}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                status = data.get("status")
                if status in ("succeeded", "failed"):
                    return data
        except Exception:
            pass
        time.sleep(2)
    return {"status": "timeout", "score": 0.0, "error": "Job execution timed out"}

def process_function(f, language: str, provider: str, model: str, target: str, enable_cbmc: bool = False) -> dict:
    name = f["name"]
    source = f["full"]
    print(f"Submitting function: {name}...")
    job_id = submit_job(name, source, language, provider, model, target, enable_cbmc)
    if not job_id:
        return {"name": name, "status": "failed", "score": 0.0, "time": 0.0, "speedup": 1.0, "error": "Failed to submit"}
    
    t0 = time.time()
    res = poll_job(job_id)
    elapsed = time.time() - t0
    
    # Retrieve speedup if autotuning details are present
    speedup = 1.0
    annotated = res.get("annotated_ir") or {}
    # Check if attempts or autotuning has details
    attempts = res.get("attempts") or []
    if attempts:
        # Get highest score attempt
        best_att = max(attempts, key=lambda x: x.get("score", 0.0), default={})
        # Check if we can get best candidate speedup
        # Or look at annotated_ir details
    
    return {
        "name": name,
        "status": res.get("status", "failed"),
        "score": res.get("score", 0.0),
        "time": elapsed,
        "speedup": speedup,
        "error": res.get("error", "")
    }

def main():
    parser = argparse.ArgumentParser(description="APG Batch Submission CLI")
    parser.add_argument("--file", required=True, help="Path to source file containing multiple functions")
    parser.add_argument("--language", default="c", choices=["c", "fortran", "python", "rust"], help="Target language (defaults to extension auto-detect)")
    parser.add_argument("--provider", default="gemini", help="LLM provider (default: gemini)")
    parser.add_argument("--model", default=None, help="LLM model override")
    parser.add_argument("--target", default="openmp", help="Target platform/pragmas (default: openmp)")
    parser.add_argument("--cbmc", action="store_true", help="Enable CBMC bounded model check (Gate 3b)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return 1

    # Language detection
    lang = args.language
    if lang == "c":
        ext = os.path.splitext(args.file)[1].lower()
        if ext in (".f90", ".f95", ".f", ".for", ".f03", ".f08"):
            lang = "fortran"
        elif ext == ".py":
            lang = "python"
        elif ext in (".rs", ".rust"):
            lang = "rust"

    try:
        driver = get_driver(lang)
    except Exception as e:
        print(f"Error loading driver: {e}")
        return 1

    with open(args.file, "r", encoding="utf-8") as f:
        source_code = f.read()

    funcs = driver.extract_functions(source_code)
    if not funcs:
        print("No functions extracted from file. Make sure the file format is valid.")
        return 1

    print(f"Extracted {len(funcs)} function(s) from {args.file} ({lang}).")
    
    # Process functions concurrently using thread pool
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_function, f, lang, args.provider, args.model, args.target, args.cbmc): f for f in funcs}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'BATCH SUBMISSION SUMMARY':^80}")
    print("="*80)
    print(f"{'Function Name':<25} | {'Status':<10} | {'Score':<5} | {'Time (s)':<8} | {'Details / Error'}")
    print("-"*80)
    for r in results:
        err_msg = r["error"][:30] if r["error"] else ""
        if r["status"] == "succeeded":
            status_str = "\033[92mPASS\033[0m" if sys.stdout.isatty() else "PASS"
        else:
            status_str = "\033[91mFAIL\033[0m" if sys.stdout.isatty() else "FAIL"
            if not err_msg:
                err_msg = "Low score / Compile error"
        
        print(f"{r['name']:<25} | {status_str:<10} | {r['score']:<5.2f} | {r['time']:<8.1f} | {err_msg}")
    print("="*80)

    # Return status code based on success
    if any(r["status"] != "succeeded" for r in results):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

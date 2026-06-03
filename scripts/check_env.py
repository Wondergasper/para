"""
APG Environment Check
=====================
Checks all required tools and Python packages for the APG system.

Usage:
    python scripts/check_env.py

Exit code 0 if all critical deps are present, 1 otherwise.
"""

import importlib
import os
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _ok(msg):   print(f"  [OK]      {msg}")
def _warn(msg): print(f"  [WARN]    {msg}")
def _miss(msg): print(f"  [MISSING] {msg}")
def _info(msg): print(f"            {msg}")


# ── Binary checker ─────────────────────────────────────────────────────────────

WINDOWS_GCC_PATHS = [
    r"C:\msys64\mingw64\bin\gcc.exe",
    r"C:\msys64\ucrt64\bin\gcc.exe",
    r"C:\mingw64\bin\gcc.exe",
    r"C:\mingw-w64\bin\gcc.exe",
    r"C:\tools\mingw64\bin\gcc.exe",
]


def find_gcc():
    found = shutil.which("gcc")
    if found:
        return found
    if os.name == "nt":
        for p in WINDOWS_GCC_PATHS:
            if os.path.isfile(p):
                return p
    return None


WINDOWS_CLANG_PATHS = [
    r"C:\Program Files\LLVM\bin\clang.exe",
    r"C:\Program Files (x86)\LLVM\bin\clang.exe",
]


def find_clang():
    found = shutil.which("clang")
    if found:
        return found
    if os.name == "nt":
        for p in WINDOWS_CLANG_PATHS:
            if os.path.isfile(p):
                return p
    return None


def check_binaries() -> bool:
    print("\nBinaries")
    print("-" * 50)
    all_ok = True

    # GCC (critical)
    gcc = find_gcc()
    if gcc:
        try:
            ver = subprocess.check_output([gcc, "--version"], text=True, timeout=5).splitlines()[0]
            _ok(f"gcc         {ver}  [{gcc}]")
        except Exception:
            _ok(f"gcc         found at {gcc}")
    else:
        _miss("gcc         (required for Gates 1 & 2)")
        _info("Install: https://www.msys2.org/  then: pacman -S mingw-w64-ucrt-x86_64-gcc")
        _info("         or: https://winlibs.com/")
        all_ok = False

    # Clang/TSAN (optional on Windows)
    clang = find_clang()
    if clang:
        try:
            ver = subprocess.check_output([clang, "--version"], text=True, timeout=5).splitlines()[0]
            _ok(f"clang       {ver}")
        except Exception:
            _ok(f"clang       found at {clang}")
    else:
        if os.name == "nt":
            _warn("clang       not found - TSAN gate will be skipped (expected on Windows)")
        else:
            _miss("clang       (required for Gate 3 ThreadSanitizer)")
            _info("Install: apt install clang  /  brew install llvm")

    # CBMC (optional)
    cbmc = shutil.which("cbmc")
    if cbmc:
        try:
            ver = subprocess.check_output([cbmc, "--version"], text=True, timeout=5).strip()
            _ok(f"cbmc        {ver}")
        except Exception:
            _ok(f"cbmc        found at {cbmc}")
    else:
        _warn("cbmc        not found - Gate 3b (bounded model check) will be skipped")
        _info("Install: https://github.com/diffblue/cbmc/releases")

    # Lean 4 (optional)
    lean = shutil.which("lean")
    if lean:
        try:
            ver = subprocess.check_output([lean, "--version"], text=True, timeout=5).strip()
            _ok(f"lean        {ver}")
        except Exception:
            _ok(f"lean        found at {lean}")
    else:
        _warn("lean        not found - Gate 4 (formal proof) will be skipped")
        _info("Install: https://leanprover.github.io/lean4/doc/setup.html")

    # Ollama
    ollama = shutil.which("ollama")
    if ollama:
        try:
            ver = subprocess.check_output([ollama, "--version"], text=True, timeout=5).strip()
            _ok(f"ollama      {ver}")
        except Exception:
            _ok(f"ollama      found at {ollama}")
    else:
        _miss("ollama      (required for local LLM)")
        _info("Install: https://ollama.com/download")
        all_ok = False

    return all_ok


# ── Python package checker ─────────────────────────────────────────────────────

PYTHON_PACKAGES = [
    # (import_name, pip_name, required)
    ("openai",     "openai>=1.30.0",    True,  "LLM client (Ollama/Groq/Gemini)"),
    ("nats",       "nats-py>=2.7.0",    True,  "NATS JetStream messaging"),
    ("pycparser",  "pycparser>=2.21",   True,  "C AST parsing (polyhedral classifier)"),
    ("google.protobuf", "protobuf>=4.25.0", True, "Protobuf serialization"),
    ("tree_sitter", "tree-sitter>=0.21.0", False, "Robust C AST (Layer 1 upgrade)"),
    ("tree_sitter_c", "tree-sitter-c>=0.21.0", False, "C grammar for tree-sitter"),
    ("islpy",      "islpy",             False, "ISL polyhedral analysis (optional)"),
    ("psycopg2",   "psycopg2-binary",   False, "PostgreSQL (Phase 3+ corpus)"),
    ("torch",      "torch",             False, "PyTorch (Phase 4 LoRA training)"),
    ("transformers", "transformers",    False, "HuggingFace (Phase 4)"),
    ("peft",       "peft",              False, "LoRA adapter (Phase 4)"),
    ("trl",        "trl",               False, "SFT Trainer (Phase 4)"),
]


def check_python_packages() -> bool:
    print("\nPython Packages")
    print("-" * 50)
    all_required_ok = True

    for import_name, pip_name, required, purpose in PYTHON_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            _ok(f"{import_name:<20} {version:<12} {purpose}")
        except ImportError:
            if required:
                _miss(f"{import_name:<20} {'[REQUIRED]':<12} {purpose}")
                _info(f"Install: pip install {pip_name}")
                all_required_ok = False
            else:
                _warn(f"{import_name:<20} {'[optional]':<12} {purpose}")
                _info(f"Install: pip install {pip_name}")

    return all_required_ok


# ── Ollama connectivity check ──────────────────────────────────────────────────

def check_ollama_connectivity() -> bool:
    print("\nOllama Connectivity")
    print("-" * 50)
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            import json
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                _ok(f"Ollama running | models: {', '.join(models)}")
                # Check if a coding model is available
                coding_models = [m for m in models if any(
                    k in m.lower() for k in ("deepseek", "qwen", "code", "starcoder")
                )]
                if coding_models:
                    _ok(f"Coding model available: {coding_models[0]}")
                else:
                    _warn("No coding model found. Recommend:")
                    _info("ollama pull deepseek-r1:8b")
                    _info("ollama pull qwen2.5-coder:7b")
                return True
            else:
                _warn("Ollama running but no models pulled.")
                _info("Run: ollama pull deepseek-r1:8b")
                return True
    except urllib.error.URLError:
        _miss("Ollama not reachable at http://localhost:11434")
        _info("Start Ollama: ollama serve")
        return False
    except Exception as e:
        _warn(f"Ollama check error: {e}")
        return False


# ── NATS connectivity check ────────────────────────────────────────────────────

def check_nats_connectivity() -> bool:
    print("\nNATS Connectivity")
    print("-" * 50)
    import socket
    try:
        s = socket.create_connection(("localhost", 4222), timeout=2)
        s.close()
        _ok("NATS reachable at localhost:4222")
        return True
    except (ConnectionRefusedError, OSError):
        _warn("NATS not running at localhost:4222")
        _info("Start NATS: docker run -p 4222:4222 nats:latest -js")
        _info("      OR:   docker compose up -d  (using docker-compose.yml in project root)")
        return False


# ── Phase readiness ────────────────────────────────────────────────────────────

def print_phase_readiness(gcc_ok, ollama_ok, nats_ok, pkgs_ok):
    print("\nPhase Readiness")
    print("-" * 50)

    phase1 = pkgs_ok and ollama_ok
    phase2 = phase1 and nats_ok
    phase3_gcc = gcc_ok
    phase4 = False  # needs ≥500 proof records

    _ok("Phase 1 (HTTP API + LLM pipeline)") if phase1 else _miss("Phase 1")
    _ok("Phase 2 (NATS worker + multi-candidate)") if phase2 else _miss("Phase 2 - start NATS")
    (lambda: _ok("Phase 3 (GCC verification gates)") if phase3_gcc else _warn("Phase 3 - install GCC for live verification"))()
    _warn("Phase 4 (LoRA fine-tuning) - needs >=500 proof records")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("APG System - Environment Check")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"OS:     {sys.platform} ({'Windows' if os.name == 'nt' else 'Unix-like'})")

    gcc_ok  = check_binaries()
    pkgs_ok = check_python_packages()
    ollama_ok = check_ollama_connectivity()
    nats_ok   = check_nats_connectivity()

    print_phase_readiness(gcc_ok, ollama_ok, nats_ok, pkgs_ok)

    print("\n" + "=" * 60)
    if pkgs_ok and ollama_ok:
        print("System is ready. Run the pipeline:")
        print("  go run ./cmd/orchestrator                  (Go orchestrator)")
        print("  python -m workers.nats_worker              (Python worker)")
        print("  curl -X POST http://localhost:8080/submit --data-binary \"@examples/vector_scale.c\"")
    else:
        print("Fix the MISSING items above, then re-run this check.")
    print("=" * 60)

    return 0 if (pkgs_ok and ollama_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

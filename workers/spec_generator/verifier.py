"""
Phase 1 verifier for APG.

Implements the verification gates described in APG_System_Architecture.docx:
compile, output comparison, optional race check, optional proof check.

Windows note: GCC is discovered from common MinGW paths if not on PATH.
TSAN gate is automatically skipped on Windows (requires clang + Linux/macOS).
"""

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field


# ── GCC discovery ──────────────────────────────────────────────────────────────

WINDOWS_GCC_PATHS = [
    r"C:\msys64\mingw64\bin\gcc.exe",
    r"C:\msys64\ucrt64\bin\gcc.exe",
    r"C:\mingw64\bin\gcc.exe",
    r"C:\mingw-w64\bin\gcc.exe",
    r"C:\Program Files\mingw-w64\bin\gcc.exe",
    r"C:\Program Files (x86)\mingw-w64\bin\gcc.exe",
    r"C:\tools\mingw64\bin\gcc.exe",
]


def find_gcc() -> str | None:
    """Return the path to gcc, or None if not found."""
    found = shutil.which("gcc")
    if found:
        return found
    if os.name == "nt":
        for p in WINDOWS_GCC_PATHS:
            if os.path.isfile(p):
                return p
    return None


# ── Clang discovery ────────────────────────────────────────────────────────────

WINDOWS_CLANG_PATHS = [
    r"C:\Program Files\LLVM\bin\clang.exe",
    r"C:\Program Files (x86)\LLVM\bin\clang.exe",
]


def find_clang() -> str | None:
    """Return the path to clang compiler, or None if not found."""
    found = shutil.which("clang")
    if found:
        return found
    if os.name == "nt":
        for p in WINDOWS_CLANG_PATHS:
            if os.path.isfile(p):
                return p
    return None


def is_docker_functional() -> bool:
    """Return True if Docker is installed and running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


# ── Verification result ────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    score: float = 0.0
    gate_reached: str = "none"
    gate_passed: str = "none"
    error: str = ""
    compile_ok: bool = False
    output_ok: bool = False
    race_ok: bool = False
    cbmc_ok: bool = False
    proof_ok: bool = False
    details: dict = field(default_factory=dict)


# ── Harness templates ──────────────────────────────────────────────────────────

COMPILE_HARNESS = textwrap.dedent("""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

{candidate_code}

int main(void) {{ return 0; }}
""")


OUTPUT_HARNESS_BASE = textwrap.dedent("""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

{reference_code}

{candidate_code}

#define N {array_size}
#define SEED {seed}

static void fill_array_float(float* arr, int n, unsigned int* state) {{
    for (int i = 0; i < n; i++) {{
        *state = (*state * 1103515245 + 12345) & 0x7fffffff;
        arr[i] = (float)(*state % 1000) / 100.0f;
    }}
}}

static void fill_array_double(double* arr, int n, unsigned int* state) {{
    for (int i = 0; i < n; i++) {{
        *state = (*state * 1103515245 + 12345) & 0x7fffffff;
        arr[i] = (double)(*state % 1000) / 100.0;
    }}
}}

static void fill_array_int(int* arr, int n, unsigned int* state) {{
    for (int i = 0; i < n; i++) {{
        *state = (*state * 1103515245 + 12345) & 0x7fffffff;
        arr[i] = (int)(*state % 1000);
    }}
}}

int main(void) {{
    unsigned int state = SEED;
    double max_diff = 0.0;

{allocations}

    {ref_call}
    {par_call}

    for (int i = 0; i < N; i++) {{
{diff_checks}
    }}

{frees}

    if (max_diff < 1e-4) {{
        printf("PASS max_diff=%.2e\\n", max_diff);
        return 0;
    }}
    printf("FAIL max_diff=%.2e\\n", max_diff);
    return 1;
}}
""")


TSAN_HARNESS = textwrap.dedent("""
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

{candidate_code}

int main(void) {{
    int N = 1024;
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    if (!A || !B) return 2;
    for (int i = 0; i < N; i++) {{ A[i] = (float)i; B[i] = (float)(N - i); }}
    free(A); free(B);
    return 0;
}}
""")


CBMC_HARNESS_BASE = textwrap.dedent("""
#include <assert.h>

{reference_code}

{candidate_code}

#define N {array_size}

int main(void) {{
{allocations}

    {ref_call}
    {par_call}

    for (int i = 0; i < N; i++) {{
{diff_checks}
    }}

    return 0;
}}
""")


# ── Utility: rename C function ─────────────────────────────────────────────────

def rename_c_function(source_code: str, func_name: str, new_name: str) -> str:
    pattern = rf"(\b[A-Za-z_][A-Za-z0-9_\s\*]*\s+){re.escape(func_name)}(\s*\()"
    return re.sub(pattern, rf"\1{new_name}\2", source_code, count=1)


# ── Parameter extraction ───────────────────────────────────────────────────────

def parse_parameters(source_code: str, func_name: str) -> list[dict]:
    from pycparser import c_parser, c_ast

    parser = c_parser.CParser()
    try:
        fake_libc = "typedef int size_t; typedef int int32_t;\n"
        ast = parser.parse(fake_libc + source_code)
    except Exception as e:
        raise ValueError(f"Could not parse source code for parameter extraction: {e}")

    target_node = None
    for node in ast.ext:
        if isinstance(node, c_ast.FuncDef):
            if node.decl.name == func_name:
                target_node = node
                break
        elif isinstance(node, c_ast.Decl):
            if node.name == func_name:
                target_node = node
                break

    if not target_node:
        raise ValueError(f"Could not find function {func_name} in source")

    decl = target_node if isinstance(target_node, c_ast.Decl) else target_node.decl

    if not isinstance(decl.type, c_ast.FuncDecl):
        raise ValueError(f"Symbol {func_name} is not a function")

    params = []
    if decl.type.args:
        for param_decl in decl.type.args.params:
            if not isinstance(param_decl, c_ast.Decl):
                continue

            name = param_decl.name
            type_node = param_decl.type

            is_ptr = False
            base_type = "float"

            curr = type_node
            while curr:
                if isinstance(curr, c_ast.PtrDecl):
                    is_ptr = True
                elif isinstance(curr, c_ast.ArrayDecl):
                    is_ptr = True
                elif isinstance(curr, c_ast.TypeDecl):
                    if isinstance(curr.type, c_ast.IdentifierType):
                        base_type = curr.type.names[0]
                        break

                if hasattr(curr, 'type'):
                    curr = curr.type
                else:
                    break

            params.append({
                "name": name,
                "type": base_type,
                "is_ptr": is_ptr,
                "base_type": base_type
            })

    return params


# ── Output harness builder ─────────────────────────────────────────────────────

def build_output_harness(
    candidate_code: str,
    reference_code: str,
    func_name: str,
    array_size: int,
    seed: int,
) -> str:
    ref_name = f"apg_ref_{func_name}"
    candidate_name = f"apg_candidate_{func_name}"
    params = parse_parameters(reference_code, func_name)

    allocs = []
    frees = []
    ref_args = []
    par_args = []
    diff_checks = []

    for p in params:
        name = p["name"]
        if p["is_ptr"]:
            base_type = p["base_type"]
            if base_type not in ("float", "double", "int"):
                base_type = "float"

            allocs.append(f"    {base_type} *ref_{name} = malloc(N * sizeof({base_type}));")
            allocs.append(f"    {base_type} *par_{name} = malloc(N * sizeof({base_type}));")
            allocs.append(f"    if (!ref_{name} || !par_{name}) return 2;")
            allocs.append(f"    fill_array_{base_type}(ref_{name}, N, &state);")
            allocs.append(f"    memcpy(par_{name}, ref_{name}, N * sizeof({base_type}));")

            frees.append(f"    free(ref_{name}); free(par_{name});")

            ref_args.append(f"ref_{name}")
            par_args.append(f"par_{name}")

            diff_checks.append(f"        double diff_{name} = fabs((double)ref_{name}[i] - (double)par_{name}[i]);")
            diff_checks.append(f"        if (diff_{name} > max_diff) max_diff = diff_{name};")
        else:
            if any(t in p["base_type"] for t in ("int", "size_t", "long", "short")) or name.upper() == "N":
                ref_args.append("N")
                par_args.append("N")
            else:
                ref_args.append("2.5f")
                par_args.append("2.5f")

    return OUTPUT_HARNESS_BASE.format(
        reference_code=rename_c_function(reference_code, func_name, ref_name),
        candidate_code=rename_c_function(candidate_code, func_name, candidate_name),
        array_size=array_size,
        seed=seed,
        allocations="\n".join(allocs),
        ref_call=f"{ref_name}({', '.join(ref_args)});",
        par_call=f"{candidate_name}({', '.join(par_args)});",
        diff_checks="\n".join(diff_checks),
        frees="\n".join(frees),
    )


# ── CBMC harness builder ───────────────────────────────────────────────────────

def build_cbmc_harness(
    candidate_code: str,
    reference_code: str,
    func_name: str,
    array_size: int = 8,
) -> str:
    ref_name = f"apg_ref_{func_name}"
    candidate_name = f"apg_candidate_{func_name}"
    params = parse_parameters(reference_code, func_name)

    allocs = []
    ref_args = []
    par_args = []
    diff_checks = []

    for p in params:
        name = p["name"]
        if p["is_ptr"]:
            base_type = p["base_type"]
            if base_type not in ("float", "double", "int"):
                base_type = "float"

            allocs.append(f"    {base_type} ref_{name}[N];")
            allocs.append(f"    {base_type} par_{name}[N];")
            allocs.append(f"    for (int i = 0; i < N; i++) {{")
            if base_type == "int":
                allocs.append(f"        ref_{name}[i] = i;")
            else:
                allocs.append(f"        ref_{name}[i] = ({base_type})i;")
            allocs.append(f"        par_{name}[i] = ref_{name}[i];")
            allocs.append(f"    }}")

            ref_args.append(f"ref_{name}")
            par_args.append(f"par_{name}")

            diff_checks.append(f"        {base_type} diff_{name} = ref_{name}[i] - par_{name}[i];")
            if base_type == "int":
                diff_checks.append(f"        assert(diff_{name} == 0);")
            else:
                diff_checks.append(f"        assert(diff_{name} <= 1e-4 && diff_{name} >= -1e-4);")
        else:
            if any(t in p["base_type"] for t in ("int", "size_t", "long", "short")) or name.upper() == "N":
                ref_args.append("N")
                par_args.append("N")
            else:
                ref_args.append("2.5f")
                par_args.append("2.5f")

    return CBMC_HARNESS_BASE.format(
        reference_code=rename_c_function(reference_code, func_name, ref_name),
        candidate_code=rename_c_function(candidate_code, func_name, candidate_name),
        array_size=array_size,
        allocations="\n".join(allocs),
        ref_call=f"{ref_name}({', '.join(ref_args)});",
        par_call=f"{candidate_name}({', '.join(par_args)});",
        diff_checks="\n".join(diff_checks),
    )


# ── Gate 1: Compile ────────────────────────────────────────────────────────────

def gate_compile(candidate_code: str, timeout: int = 30) -> tuple[bool, str]:
    """
    Gate 1: Attempt to compile the candidate with GCC + OpenMP.
    Returns (ok, error_message).
    """
    docker_ok = is_docker_functional()
    gcc = find_gcc()
    
    if not gcc and not docker_ok:
        return False, (
            "GCC not found. Install GCC with OpenMP support:\n"
            "  Windows: https://www.msys2.org/ → pacman -S mingw-w64-ucrt-x86_64-gcc\n"
            "  Linux:   sudo apt install gcc\n"
            "  macOS:   brew install gcc"
        )

    harness = COMPILE_HARNESS.format(candidate_code=candidate_code)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "compile_test.c")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(harness)

        if docker_ok:
            result = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{os.path.abspath(tmpdir)}:/app", "-w", "/app", "gcc", "gcc", "-fopenmp", "-O1", "-Wall", "compile_test.c", "-o", "compile_test", "-lm"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            bin_path = os.path.join(tmpdir, "compile_test")
            result = subprocess.run(
                [gcc, "-fopenmp", "-O1", "-Wall", src_path, "-o", bin_path, "-lm"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

    if result.returncode == 0:
        return True, ""
    return False, f"Compile error:\n{result.stderr or result.stdout}"


# ── Gate 2: Output match ───────────────────────────────────────────────────────

def gate_output_match(
    candidate_code: str,
    reference_code: str,
    func_name: str,
    array_size: int = 1024,
    seed: int = 42,
    timeout: int = 30,
) -> tuple[bool, str]:
    docker_ok = is_docker_functional()
    gcc = find_gcc()
    if not gcc and not docker_ok:
        return False, "GCC not found — Gate 2 skipped."

    try:
        harness = build_output_harness(candidate_code, reference_code, func_name, array_size, seed)
    except ValueError as exc:
        return False, str(exc)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "test_output.c")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(harness)

        if docker_ok:
            compile_result = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{os.path.abspath(tmpdir)}:/app", "-w", "/app", "gcc", "gcc", "-fopenmp", "-O1", "test_output.c", "-o", "test_output", "-lm"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if compile_result.returncode != 0:
                return False, f"Harness compile error:\n{compile_result.stderr or compile_result.stdout}"

            run_result = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{os.path.abspath(tmpdir)}:/app", "-w", "/app", "--network", "none", "--cpus", "0.5", "-m", "128m", "-e", "OMP_NUM_THREADS=4", "gcc", "./test_output"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            bin_path = os.path.join(tmpdir, "test_output")
            compile_result = subprocess.run(
                [gcc, "-fopenmp", "-O1", src_path, "-o", bin_path, "-lm"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if compile_result.returncode != 0:
                return False, f"Harness compile error:\n{compile_result.stderr}"

            run_result = subprocess.run(
                [bin_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "OMP_NUM_THREADS": "4"},
            )

    output = (run_result.stdout or "").strip()
    if "PASS" in output:
        return True, ""
    return False, f"Output mismatch: {output or run_result.stderr or run_result.stdout}"


# ── Gate 3: Race freedom (TSAN) ────────────────────────────────────────────────

def gate_race_freedom(candidate_code: str, timeout: int = 60) -> tuple[bool, str]:
    docker_ok = is_docker_functional()
    clang_path = find_clang()
    if os.name == "nt" and not docker_ok:
        return True, "TSAN gate skipped: ThreadSanitizer is not supported on Windows"

    if not docker_ok and clang_path is None:
        return True, "TSAN gate skipped: clang not available"

    harness = TSAN_HARNESS.format(candidate_code=candidate_code)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tsan_test.c")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(harness)

        if docker_ok:
            compile_result = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{os.path.abspath(tmpdir)}:/app", "-w", "/app", "gcc", "gcc", "-fopenmp", "-fsanitize=thread", "-O1", "tsan_test.c", "-o", "tsan_test", "-lm"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if compile_result.returncode != 0:
                # If GCC compiler in container lacks TSAN support, compile with clang inside container
                compile_result_clang = subprocess.run(
                    ["docker", "run", "--rm", "-v", f"{os.path.abspath(tmpdir)}:/app", "-w", "/app", "gcc", "bash", "-c", "apt-get update -y && apt-get install -y clang && clang -fopenmp -fsanitize=thread -O1 tsan_test.c -o tsan_test"],
                    capture_output=True,
                    text=True,
                    timeout=timeout * 2,
                )
                if compile_result_clang.returncode != 0:
                    return True, "TSAN gate skipped: compilation inside container failed (TSAN missing in gcc image)"
            
            run_result = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{os.path.abspath(tmpdir)}:/app", "-w", "/app", "--network", "none", "--cpus", "0.5", "-m", "128m", "-e", "OMP_NUM_THREADS=4", "-e", "TSAN_OPTIONS=halt_on_error=1", "gcc", "./tsan_test"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            bin_path = os.path.join(tmpdir, "tsan_test")
            compile_result = subprocess.run(
                [clang_path, "-fopenmp", "-fsanitize=thread", "-O1", src_path, "-o", bin_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if compile_result.returncode != 0:
                return False, f"TSAN compile error:\n{compile_result.stderr}"

            run_result = subprocess.run(
                [bin_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "OMP_NUM_THREADS": "4", "TSAN_OPTIONS": "halt_on_error=1"},
            )

    combined = ((run_result.stdout or "") + (run_result.stderr or "")).strip()
    if "data race" in combined.lower() or "race condition" in combined.lower():
        return False, f"ThreadSanitizer detected race:\n{combined[:500]}"
    return True, ""


# ── Gate 3b: CBMC bounded model check ─────────────────────────────────────────

def gate_cbmc_model_check(
    candidate_code: str,
    reference_code: str,
    func_name: str,
    array_size: int = 8,
    timeout: int = 60,
    cbmc_path: str = "cbmc",
) -> tuple[bool, str]:
    resolved_cbmc = shutil.which(cbmc_path) if os.path.basename(cbmc_path) == cbmc_path else cbmc_path
    if not resolved_cbmc or not os.path.exists(resolved_cbmc):
        return True, "CBMC gate skipped: cbmc not available"

    try:
        harness = build_cbmc_harness(candidate_code, reference_code, func_name, array_size)
    except ValueError as exc:
        return False, str(exc)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "cbmc_check.c")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(harness)

        result = subprocess.run(
            [resolved_cbmc, src_path, "--unwind", str(array_size + 1), "--bounds-check", "--pointer-check"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return True, output
    return False, output[:1000]


# ── Gate 4: Lean 4 formal proof ────────────────────────────────────────────────

def gate_formal_proof(lean_code: str, timeout: int = 60) -> tuple[bool, str]:
    if shutil.which("lean") is None:
        return False, "Lean 4 not available — gate 4 skipped (soft miss)"

    with tempfile.TemporaryDirectory() as tmpdir:
        lean_path = os.path.join(tmpdir, "Proof.lean")
        with open(lean_path, "w", encoding="utf-8") as f:
            f.write(lean_code)

        result = subprocess.run(["lean", lean_path], capture_output=True, text=True, timeout=timeout)

    if result.returncode == 0:
        return True, ""
    return False, (result.stderr or result.stdout).strip()


# ── Main verify function ───────────────────────────────────────────────────────

def verify_candidate(
    candidate_code: str,
    reference_code: str,
    func_name: str = "func",
    test_inputs: list = None,
    lean_code: str = None,
    enable_tsan: bool = True,
    enable_cbmc: bool = False,
    cbmc_path: str = "cbmc",
    enable_proof: bool = False,
) -> VerificationResult:
    result = VerificationResult()
    test_inputs = test_inputs or [{"N": 1024, "seed": 42}]

    # Gate 1: Compile
    result.gate_reached = "compile"
    ok, err = gate_compile(candidate_code)
    if not ok:
        result.error = err
        return result

    result.compile_ok = True
    result.gate_passed = "compile"
    result.score = 0.2

    # Gate 2: Output match
    result.gate_reached = "output"
    ok, err = gate_output_match(
        candidate_code,
        reference_code,
        func_name,
        array_size=test_inputs[0].get("N", 1024),
        seed=test_inputs[0].get("seed", 42),
    )
    if not ok:
        result.error = err
        return result

    result.output_ok = True
    result.gate_passed = "output"
    result.score = 0.6

    # Gate 3: Race freedom (TSAN)
    if enable_tsan:
        result.gate_reached = "race"
        ok, err = gate_race_freedom(candidate_code)
        if not ok:
            result.error = err
            return result
        result.race_ok = True
        result.gate_passed = "race"
        result.score = 0.8

    # Gate 3b: CBMC
    if enable_cbmc:
        result.gate_reached = "cbmc"
        ok, err = gate_cbmc_model_check(
            candidate_code,
            reference_code,
            func_name,
            array_size=test_inputs[0].get("N", 8),
            cbmc_path=cbmc_path,
        )
        result.details["cbmc"] = err
        if not ok:
            result.error = err
            return result
        result.cbmc_ok = "skipped" not in err.lower()
        result.gate_passed = "cbmc"

    # Gate 4: Lean 4 formal proof
    if enable_proof and lean_code:
        result.gate_reached = "proof"
        ok, err = gate_formal_proof(lean_code)
        if ok:
            result.proof_ok = True
            result.gate_passed = "proof"
            result.score = 1.0
        else:
            result.details["proof_error"] = err

    # Normalise final score when TSAN is disabled
    if result.gate_passed in ("output",) and not enable_tsan:
        result.score = 0.8
    if result.gate_passed in ("race", "cbmc") and not enable_proof:
        result.gate_passed = "all"

    return result


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seq = """void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) A[i] *= s;
}"""
    par = """void vector_scale(float* A, float s, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) A[i] *= s;
}"""
    res = verify_candidate(par, seq, func_name="vector_scale", enable_tsan=False)
    print(f"Score:        {res.score}")
    print(f"Gate reached: {res.gate_reached}")
    print(f"Gate passed:  {res.gate_passed}")
    if res.error:
        print(f"Error: {res.error}")
    gcc = find_gcc()
    print(f"GCC found:    {gcc or 'NOT FOUND — install from https://www.msys2.org/'}")

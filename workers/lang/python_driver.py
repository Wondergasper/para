import re
import os
import sys
import tempfile
import subprocess
from workers.lang.base_driver import LanguageDriver
from workers.spec_generator.verifier import VerificationResult

SYSTEM_PROMPT_PYTHON_T1 = """You are a performance analysis engine specialised in Python and Numba.
Given a Python function, classify its loops and summarise its computation.

OUTPUT RULES (follow exactly):
1. Output ONLY valid JSON. No explanation, no markdown, no code fences.
2. The JSON must match this schema exactly:
   {
     "type": "<polyhedral|irregular|sequential>",
     "summary": "<one sentence describing what the function computes>",
     "affine_dimensions": ["<loop var>", ...],
     "dependencies": {
       "RAW": [["<written>", "<read>"], ...],
       "WAR": [["<read>",    "<written>"], ...],
       "WAW": [["<w1>",      "<w2>"], ...]
     },
     "reduction_variables": ["<var>", ...]
   }
"""

SYSTEM_PROMPT_PYTHON_T2 = """You are an expert performance Python developer specialised in compiler-based parallelisation via Numba.
Given a Python function and its dependency graph, produce a correct parallelised version using Numba `@jit(nopython=True, parallel=True)`.

OUTPUT RULES (follow exactly):
1. Output ONLY valid Python code. No explanations. No markdown fences. No comments outside the code.
2. The output must be a complete, compilable Python function.

PARALLELISATION RULES (strict):
- Decorate the parallel function with `@jit(nopython=True, parallel=True)`.
- Import `prange` from `numba` and use `prange` instead of `range` for loop parallelization (e.g. `for i in prange(N):`).
- Ensure no unsupported Numba types/operations are used in the function.
"""

class PythonDriver(LanguageDriver):
    def detect_func_name(self, source_code: str, default_name: str = "func") -> str:
        match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', source_code)
        if match:
            return match.group(1)
        return default_name

    def rename_main(self, source_code: str) -> str:
        return re.sub(r'__main__', '__main_original__', source_code)

    def extract_functions(self, source_code: str) -> list[dict]:
        lines = source_code.splitlines()
        funcs = []
        in_func = False
        func_name = ""
        func_lines = []
        base_indent = 0

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                if in_func:
                    func_lines.append(line)
                continue
            
            match = re.match(r'^(\s*)def\s+([a-zA-Z0-9_]+)\s*\(', line)
            if match and not in_func:
                in_func = True
                base_indent = len(match.group(1))
                func_name = match.group(2)
                func_lines = [line]
                continue
            
            if in_func:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line_strip:
                    in_func = False
                    body = "\n".join(func_lines)
                    funcs.append({
                        "name": func_name,
                        "signature": func_lines[0].strip(),
                        "body": body,
                        "full": body
                    })
                    func_lines = []
                    func_name = ""
                    
                    match2 = re.match(r'^(\s*)def\s+([a-zA-Z0-9_]+)\s*\(', line)
                    if match2:
                        in_func = True
                        base_indent = len(match2.group(1))
                        func_name = match2.group(2)
                        func_lines = [line]
                else:
                    func_lines.append(line)

        if in_func:
            body = "\n".join(func_lines)
            funcs.append({
                "name": func_name,
                "signature": func_lines[0].strip(),
                "body": body,
                "full": body
            })
        return funcs

    def analyze_loops(self, func_body: str) -> dict:
        issues = []
        has_io = False
        has_early_exit = False

        lines = func_body.splitlines()
        in_loop = False
        loop_indent = 0
        loop_lines = []

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
            current_indent = len(line) - len(line.lstrip())
            
            if in_loop and current_indent <= loop_indent:
                in_loop = False
                loop_body = "\n".join(loop_lines)
                if re.search(r'\b(break|return)\b', loop_body):
                    has_early_exit = True
                    issues.append("Python loop contains a premature exit/return statement.")
                if re.search(r'\b(print|input)\b', loop_body):
                    has_io = True
                    issues.append("Python loop contains console I/O operations.")
                loop_lines = []

            if (line_strip.startswith("for ") or line_strip.startswith("while ")) and ":" in line_strip:
                in_loop = True
                loop_indent = current_indent
                continue

            if in_loop:
                loop_lines.append(line_strip)

        if in_loop:
            loop_body = "\n".join(loop_lines)
            if re.search(r'\b(break|return)\b', loop_body):
                has_early_exit = True
                issues.append("Python loop contains a premature exit/return statement.")
            if re.search(r'\b(print|input)\b', loop_body):
                has_io = True
                issues.append("Python loop contains console I/O operations.")

        return {
            "safe_for_openmp": not (has_early_exit or has_io),
            "has_early_exit": has_early_exit,
            "has_io": has_io,
            "issues": issues
        }

    def get_t1_system_prompt(self) -> str:
        return SYSTEM_PROMPT_PYTHON_T1

    def get_t1_prompt(self, source_code: str) -> str:
        return f"Analyse this Python function:\n\n{source_code.strip()}\n\nOutput AnnotatedIR JSON:"

    def get_t2_system_prompt(self, target: str) -> str:
        return SYSTEM_PROMPT_PYTHON_T2

    def get_t2_prompt_hint(self, annotated_ir: dict) -> str:
        if not annotated_ir:
            return ""
        hints = []
        if annotated_ir.get("reduction_variables"):
            rvars = annotated_ir["reduction_variables"]
            hints.append(f"Reduction variables: {rvars}. Ensure you accumulate them in parallel.")
        return "\n".join(hints)

    def parse_params(self, source_code: str, func_name: str) -> list[str]:
        # Simple Python parameter name extractor
        for line in source_code.splitlines():
            match = re.search(r'def\s+' + re.escape(func_name) + r'\s*\(([^)]*)\)', line)
            if match:
                return [p.split(":")[0].strip() for p in match.group(1).split(",") if p.strip()]
        return []

    def verify_candidate(
        self,
        candidate_code: str,
        reference_code: str,
        func_name: str,
        test_inputs: list,
        cfg,
    ) -> VerificationResult:
        result = VerificationResult()
        
        # Test imports and syntax
        check_code = f"""
from numba import jit, prange
import numpy as np
{candidate_code}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = os.path.join(tmpdir, "check.py")
            with open(chk_path, "w", encoding="utf-8") as f:
                f.write(check_code)

            comp_res = subprocess.run(
                [sys.executable, chk_path],
                capture_output=True,
                text=True,
                timeout=15
            )
            if comp_res.returncode != 0:
                result.error = f"Python syntax / Numba compilation error:\n{comp_res.stderr or comp_res.stdout}"
                return result

        result.compile_ok = True
        result.gate_passed = "compile"
        result.score = 0.2

        # Gate 2: Output Match
        result.gate_reached = "output"
        ref_name = f"apg_ref_{func_name}"
        cand_name = f"apg_cand_{func_name}"

        ref_code_renamed = re.sub(r'def\s+' + re.escape(func_name) + r'\b', f"def {ref_name}", reference_code)
        cand_code_renamed = re.sub(r'def\s+' + re.escape(func_name) + r'\b', f"def {cand_name}", candidate_code)

        test_inputs = test_inputs or [{"N": 1024, "seed": 42}]
        array_size = test_inputs[0].get("N", 1024)
        params = self.parse_params(reference_code, func_name)

        allocs = []
        args_ref = []
        args_cand = []
        initializers = []
        diff_checks = []

        for p in params:
            if p.lower() in ("n", "size", "rows", "cols"):
                allocs.append(f"{p} = {array_size}")
                args_ref.append(p)
                args_cand.append(p)
            elif p.lower() in ("s", "scalar", "scale"):
                allocs.append(f"{p} = 2.5")
                args_ref.append(p)
                args_cand.append(p)
            else:
                # Array/list parameter - assume numpy float32 array
                allocs.append(f"{p}_ref = np.arange({array_size}, dtype=np.float32)")
                allocs.append(f"{p}_cand = np.arange({array_size}, dtype=np.float32)")
                args_ref.append(f"{p}_ref")
                args_cand.append(f"{p}_cand")
                diff_checks.append(f"max_diff = max(max_diff, np.max(np.abs({p}_ref - {p}_cand)))")

        harness_code = f"""
import time
import numpy as np
from numba import jit, prange

{ref_code_renamed}

{cand_code_renamed}

# Allocate
{chr(10).join(allocs)}

# Warmup JIT compiler
try:
    {ref_name}({",".join(args_ref)})
    {cand_name}({",".join(args_cand)})
except Exception as e:
    print("COMPILE_ERR:", e)
    import sys
    sys.exit(2)

# Re-initialize for timing
{chr(10).join(allocs)}

t0 = time.perf_counter()
{ref_name}({",".join(args_ref)})
t1 = time.perf_counter()
time_ref = t1 - t0

t0 = time.perf_counter()
{cand_name}({",".join(args_cand)})
t1 = time.perf_counter()
time_par = t1 - t0

max_diff = 0.0
{chr(10).join(diff_checks)}

if max_diff < 1e-4:
    print(f"PASS max_diff={{max_diff}} time_ref={{time_ref}} time_par={{time_par}}")
else:
    print(f"FAIL max_diff={{max_diff}}")
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            harness_path = os.path.join(tmpdir, "harness.py")
            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(harness_code)

            run_res = subprocess.run(
                [sys.executable, harness_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            output = (run_res.stdout or "").strip()
            if run_res.returncode != 0 or "PASS" not in output:
                result.error = f"Python output mismatch:\n{output or run_res.stderr}"
                return result

            # Parse float times
            match = re.search(r"time_ref=([0-9.Ee+-]+)\s*time_par=([0-9.Ee+-]+)", output)
            time_ref, time_par = 0.0, 0.0
            if match:
                time_ref = float(match.group(1))
                time_par = float(match.group(2))

            # Autotuning: sweep NUMBA_NUM_THREADS
            thread_counts = [1, 2, 4, 8]
            runs = []
            best_speedup = -1.0
            best_threads = 1
            best_time_par = time_par

            for T in thread_counts:
                at_res = subprocess.run(
                    [sys.executable, harness_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "NUMBA_NUM_THREADS": str(T)},
                )
                at_output = (at_res.stdout or "").strip()
                if "PASS" not in at_output:
                    continue  # skip non-passing thread counts
                at_match = re.search(r"time_ref=([0-9.Ee+-]+)\s*time_par=([0-9.Ee+-]+)", at_output)
                t_ref, t_par = time_ref, time_par
                if at_match:
                    t_ref = float(at_match.group(1))
                    t_par = float(at_match.group(2))
                sp = t_ref / t_par if t_par > 0 else 1.0
                runs.append({"threads": T, "time_ref": t_ref, "time_par": t_par, "speedup": sp})
                if sp > best_speedup or best_speedup < 0:
                    best_speedup = sp
                    best_threads = T
                    best_time_par = t_par

            if not runs:
                runs = [{"threads": 1, "time_ref": time_ref, "time_par": time_par, "speedup": time_ref / time_par if time_par > 0 else 1.0}]

            result.details["autotuning"] = {
                "runs": runs,
                "best_threads": best_threads,
                "best_speedup": best_speedup,
                "best_time_par": best_time_par,
                "time_ref": time_ref,
            }

        result.output_ok = True
        result.gate_passed = "output"
        result.score = 0.8
        return result

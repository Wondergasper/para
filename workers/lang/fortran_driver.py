import re
import os
import shutil
import tempfile
import subprocess
from workers.lang.base_driver import LanguageDriver
from workers.spec_generator.verifier import VerificationResult

SYSTEM_PROMPT_FORTRAN_T1 = """You are a polyhedral analysis engine specialised in Fortran code.
Given a Fortran subroutine/function, classify its parallelism type and summarise its computation.

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

CLASSIFICATION RULES (strict):
- If ANY array index is indirect (e.g. A(idx(i))) -> type MUST be "irregular"
- If loop bounds are non-affine expressions -> "irregular"
- If ALL loop bounds and ALL array indices are affine -> "polyhedral"
- If there are no loops -> "sequential"
"""

SYSTEM_PROMPT_FORTRAN_T2 = """You are an expert HPC programmer specialised in OpenMP parallelisation of Fortran code.
Given a Fortran subroutine/function and its dependency graph, produce a correct OpenMP-parallelised version.

OUTPUT RULES (follow exactly):
1. Output ONLY valid Fortran code. No explanations. No markdown fences. No comments outside the code.
2. The output must be a complete, compilable Fortran subroutine/function.

PARALLELISATION RULES (strict):
- Use standard Fortran OpenMP directives (e.g. !$omp parallel do, !$omp end parallel do).
- Declared private loop variables must be listed as private(var).
- For reduction operations, use: !$omp parallel do reduction(op:var)
- Do NOT add external interface definitions unless necessary.
"""

class FortranDriver(LanguageDriver):
    def detect_func_name(self, source_code: str, default_name: str = "func") -> str:
        match = re.search(r'(?i)\b(subroutine|function)\s+([a-zA-Z0-9_]+)', source_code)
        if match:
            return match.group(2)
        return default_name

    def rename_main(self, source_code: str) -> str:
        return re.sub(r'(?i)\bprogram\s+main\b', 'program main_original', source_code)

    def extract_functions(self, source_code: str) -> list[dict]:
        lines = source_code.splitlines()
        funcs = []
        in_func = False
        func_name = ""
        func_lines = []

        for line in lines:
            line_strip = line.strip()
            match = re.match(r'(?i)^\s*(recursive\s+)?(subroutine|function)\s+([a-zA-Z0-9_]+)', line_strip)
            if match and not in_func:
                in_func = True
                func_name = match.group(3)
                func_lines = [line]
                continue

            if in_func:
                func_lines.append(line)
                if re.match(r'(?i)^\s*end\s*(subroutine|function)?(\s+' + re.escape(func_name) + r')?\b', line_strip):
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

        return funcs

    def analyze_loops(self, func_body: str) -> dict:
        issues = []
        has_io = False
        has_early_exit = False

        lines = func_body.splitlines()
        in_loop = False
        loop_lines = []

        for line in lines:
            line_strip = line.strip().lower()
            if re.match(r'^do\s+[a-zA-Z0-9_]+', line_strip):
                in_loop = True
                continue
            if in_loop:
                if line_strip.startswith("end do") or line_strip.startswith("enddo"):
                    in_loop = False
                    loop_body = "\n".join(loop_lines)
                    if re.search(r'\b(exit|return|stop)\b', loop_body):
                        has_early_exit = True
                        issues.append("Fortran loop contains a premature exit/return statement.")
                    if re.search(r'\b(print|write|read)\b', loop_body):
                        has_io = True
                        issues.append("Fortran loop contains I/O operations (print/write).")
                    loop_lines = []
                else:
                    loop_lines.append(line_strip)

        return {
            "safe_for_openmp": not (has_early_exit or has_io),
            "has_early_exit": has_early_exit,
            "has_io": has_io,
            "issues": issues
        }

    def get_t1_system_prompt(self) -> str:
        return SYSTEM_PROMPT_FORTRAN_T1

    def get_t1_prompt(self, source_code: str) -> str:
        return f"Analyse this Fortran subroutine/function:\n\n{source_code.strip()}\n\nOutput AnnotatedIR JSON:"

    def get_t2_system_prompt(self, target: str) -> str:
        return SYSTEM_PROMPT_FORTRAN_T2

    def get_t2_prompt_hint(self, annotated_ir: dict) -> str:
        if not annotated_ir:
            return ""
        hints = []
        if annotated_ir.get("reduction_variables"):
            rvars = annotated_ir["reduction_variables"]
            hints.append(f"Reduction variables: {rvars}. Use reduction(op:var) in !$omp parallel do.")
        return "\n".join(hints)

    def parse_params(self, source_code: str, func_name: str) -> list[dict]:
        lines = source_code.splitlines()
        arg_names = []
        for line in lines:
            match = re.search(r'(?i)^\s*subroutine\s+' + re.escape(func_name) + r'\s*\(([^)]*)\)', line.strip())
            if match:
                arg_names = [a.strip() for a in match.group(1).split(",") if a.strip()]
                break

        params = []
        for arg in arg_names:
            is_array = False
            base_type = "real"
            for line in lines:
                line_strip = line.strip().lower()
                if arg.lower() in line_strip and "::" in line_strip:
                    if "integer" in line_strip:
                        base_type = "integer"
                    elif "real" in line_strip:
                        base_type = "real"
                    elif "double" in line_strip:
                        base_type = "double precision"
                    if "(" in line_strip or "dimension" in line_strip:
                        is_array = True
                    break
            params.append({
                "name": arg,
                "type": base_type,
                "is_array": is_array
            })
        return params

    def verify_candidate(
        self,
        candidate_code: str,
        reference_code: str,
        func_name: str,
        test_inputs: list,
        cfg,
    ) -> VerificationResult:
        result = VerificationResult()
        gfortran = shutil.which("gfortran")
        if not gfortran:
            result.error = "gfortran compiler not found on PATH. Install gfortran."
            return result

        test_inputs = test_inputs or [{"N": 1024, "seed": 42}]
        array_size = test_inputs[0].get("N", 1024)

        # Gate 1: Compile
        result.gate_reached = "compile"
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "candidate.f90")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(candidate_code)

            obj_path = os.path.join(tmpdir, "candidate.o")
            comp_res = subprocess.run(
                [gfortran, "-fopenmp", "-O1", "-c", src_path, "-o", obj_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if comp_res.returncode != 0:
                result.error = f"Fortran compile error:\n{comp_res.stderr or comp_res.stdout}"
                return result

        result.compile_ok = True
        result.gate_passed = "compile"
        result.score = 0.2

        # Gate 2: Output Match
        result.gate_reached = "output"
        ref_name = f"apg_ref_{func_name}"
        cand_name = f"apg_cand_{func_name}"

        ref_code_renamed = re.sub(r'(?i)\bsubroutine\s+' + re.escape(func_name) + r'\b', f"subroutine {ref_name}", reference_code)
        ref_code_renamed = re.sub(r'(?i)\bend\s+subroutine\s+' + re.escape(func_name) + r'\b', f"end subroutine {ref_name}", ref_code_renamed)
        ref_code_renamed = re.sub(r'(?i)\bend\s+subroutine\b', f"end subroutine {ref_name}", ref_code_renamed)

        cand_code_renamed = re.sub(r'(?i)\bsubroutine\s+' + re.escape(func_name) + r'\b', f"subroutine {cand_name}", candidate_code)
        cand_code_renamed = re.sub(r'(?i)\bend\s+subroutine\s+' + re.escape(func_name) + r'\b', f"end subroutine {cand_name}", cand_code_renamed)
        cand_code_renamed = re.sub(r'(?i)\bend\s+subroutine\b', f"end subroutine {cand_name}", cand_code_renamed)

        params = self.parse_params(reference_code, func_name)

        allocs = []
        args_ref = []
        args_cand = []
        initializers = []
        diff_checks = []

        for p in params:
            name = p["name"]
            if p["is_array"]:
                allocs.append(f"    {p['type']}, dimension({array_size}) :: ref_{name}, cand_{name}")
                if p["type"] == "integer":
                    initializers.append(f"    do i = 1, {array_size}\n        ref_{name}(i) = i\n        cand_{name}(i) = i\n    end do")
                else:
                    initializers.append(f"    do i = 1, {array_size}\n        ref_{name}(i) = float(i)\n        cand_{name}(i) = float(i)\n    end do")
                args_ref.append(f"ref_{name}")
                args_cand.append(f"cand_{name}")
                diff_checks.append(f"    do i = 1, {array_size}\n        max_diff = max(max_diff, abs(ref_{name}(i) - cand_{name}(i)))\n    end do")
            else:
                allocs.append(f"    {p['type']} :: ref_{name}, cand_{name}")
                if name.lower() == "n":
                    initializers.append(f"    ref_{name} = {array_size}\n    cand_{name} = {array_size}")
                elif p["type"] == "integer":
                    initializers.append(f"    ref_{name} = 2\n    cand_{name} = 2")
                else:
                    initializers.append(f"    ref_{name} = 2.5\n    cand_{name} = 2.5")
                args_ref.append(f"ref_{name}")
                args_cand.append(f"cand_{name}")

        harness_code = f"""
{ref_code_renamed}

{cand_code_renamed}

program test_harness
    implicit none
    integer :: i
    real :: max_diff
    double precision :: start_time, end_time, time_ref, time_par
{chr(10).join(allocs)}

    max_diff = 0.0
{chr(10).join(initializers)}

    call cpu_time(start_time)
    call {ref_name}({",".join(args_ref)})
    call cpu_time(end_time)
    time_ref = end_time - start_time

    call cpu_time(start_time)
    call {cand_name}({",".join(args_cand)})
    call cpu_time(end_time)
    time_par = end_time - start_time

{chr(10).join(diff_checks)}

    if (max_diff < 1e-4) then
        print *, "PASS max_diff=", max_diff, " time_ref=", time_ref, " time_par=", time_par
    else
        print *, "FAIL max_diff=", max_diff
    end if
end program
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            harness_path = os.path.join(tmpdir, "harness.f90")
            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(harness_code)

            bin_path = os.path.join(tmpdir, "harness")
            comp_res = subprocess.run(
                [gfortran, "-fopenmp", "-O1", harness_path, "-o", bin_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if comp_res.returncode != 0:
                result.error = f"Harness compile error:\n{comp_res.stderr or comp_res.stdout}"
                return result

            # Run autotuning over thread counts
            runs = []
            best_speedup = -1.0
            best_threads = 1
            best_time_par = 0.0

            for T in [1, 2, 4, 8]:
                run_res = subprocess.run(
                    [bin_path],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env={**os.environ, "OMP_NUM_THREADS": str(T)}
                )
                output = (run_res.stdout or "").strip()
                if "PASS" not in output:
                    result.error = f"Fortran output mismatch: {output or run_res.stderr}"
                    return result

                # Parse float times from output: time_ref=%.6f time_par=%.6f
                match = re.search(r"time_ref=\s*([0-9.Ee+-]+)\s*time_par=\s*([0-9.Ee+-]+)", output)
                time_ref, time_par = 0.0, 0.0
                if match:
                    time_ref = float(match.group(1))
                    time_par = float(match.group(2))

                speedup = time_ref / time_par if time_par > 0 else 1.0
                runs.append({
                    "threads": T,
                    "time_ref": time_ref,
                    "time_par": time_par,
                    "speedup": speedup
                })
                if speedup > best_speedup or best_speedup < 0:
                    best_speedup = speedup
                    best_threads = T
                    best_time_par = time_par

            result.details["autotuning"] = {
                "runs": runs,
                "best_threads": best_threads,
                "best_speedup": best_speedup,
                "best_time_par": best_time_par,
                "time_ref": runs[0]["time_ref"]
            }

        result.output_ok = True
        result.gate_passed = "output"
        result.score = 0.8

        # Gate 3: Race freedom via gfortran TSAN (Linux/macOS only)
        if os.name != "nt":
            gfortran_tsan = shutil.which("gfortran")
            if gfortran_tsan:
                with tempfile.TemporaryDirectory() as tdir:
                    tsan_path = os.path.join(tdir, "tsan_test.f90")
                    with open(tsan_path, "w", encoding="utf-8") as f:
                        f.write(candidate_code)
                    tsan_bin = os.path.join(tdir, "tsan_test")
                    comp_tsan = subprocess.run(
                        [gfortran_tsan, "-fopenmp", "-fsanitize=thread", "-O1", tsan_path, "-o", tsan_bin],
                        capture_output=True, text=True, timeout=30
                    )
                    if comp_tsan.returncode == 0:
                        run_tsan = subprocess.run(
                            [tsan_bin],
                            capture_output=True, text=True, timeout=15,
                            env={**os.environ, "OMP_NUM_THREADS": "4", "TSAN_OPTIONS": "halt_on_error=1"}
                        )
                        combined = (run_tsan.stdout or "") + (run_tsan.stderr or "")
                        if "data race" in combined.lower():
                            result.error = f"ThreadSanitizer detected race in Fortran code:\n{combined[:400]}"
                            result.gate_passed = "output"  # revert
                            result.score = 0.6
                            return result
                        result.race_ok = True
                        result.gate_passed = "race"
                        result.score = 1.0
        else:
            # On Windows, TSAN not available for Fortran — keep score at 0.8
            result.details["tsan_note"] = "TSAN gate skipped: not supported on Windows for Fortran"

        return result

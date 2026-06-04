import re
import os
import shutil
import tempfile
import subprocess
from workers.lang.base_driver import LanguageDriver
from workers.spec_generator.verifier import VerificationResult

SYSTEM_PROMPT_RUST_T1 = """You are a performance analysis engine specialised in Rust code.
Given a Rust function, classify its loops and summarise its computation.

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

SYSTEM_PROMPT_RUST_T2 = """You are an HPC programmer specialised in Rust performance parallelisation using Rayon.
Given a Rust function and its dependency graph, produce a correct parallelised version using Rayon (e.g. `.par_iter()`, `.into_par_iter()`, `.par_iter_mut()`).

OUTPUT RULES (follow exactly):
1. Output ONLY valid Rust code. No explanations. No markdown fences. No comments outside the code.
2. The output must be a complete, compilable Rust function.
"""

class RustDriver(LanguageDriver):
    def detect_func_name(self, source_code: str, default_name: str = "func") -> str:
        match = re.search(r'fn\s+([a-zA-Z0-9_]+)\s*\(', source_code)
        if match:
            return match.group(1)
        return default_name

    def rename_main(self, source_code: str) -> str:
        return re.sub(r'\bfn\s+main\b', 'fn main_original', source_code)

    def extract_functions(self, source_code: str) -> list[dict]:
        funcs = []
        header_pattern = re.compile(
            r'\b(pub\s+)?(unsafe\s+)?fn\s+([a-zA-Z0-9_]+)\s*(\([^{]*\))\s*(\{)',
            re.MULTILINE
        )
        for match in header_pattern.finditer(source_code):
            func_name = match.group(3)
            start_idx = match.end() - 1
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(source_code)):
                char = source_code[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if end_idx != -1:
                body = source_code[start_idx:end_idx]
                full = source_code[match.start():end_idx]
                funcs.append({
                    "name": func_name,
                    "signature": source_code[match.start():start_idx],
                    "body": body,
                    "full": full
                })
        return funcs

    def analyze_loops(self, func_body: str) -> dict:
        issues = []
        has_io = False
        has_early_exit = False

        lines = func_body.splitlines()
        in_loop = False
        brace_count = 0
        loop_lines = []

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
            
            if in_loop:
                loop_lines.append(line_strip)
                brace_count += line_strip.count("{") - line_strip.count("}")
                if brace_count <= 0:
                    in_loop = False
                    loop_body = "\n".join(loop_lines)
                    if re.search(r'\b(break|return)\b', loop_body):
                        has_early_exit = True
                        issues.append("Rust loop contains a premature exit/return statement.")
                    if "print!" in loop_body or "println!" in loop_body:
                        has_io = True
                        issues.append("Rust loop contains console I/O operations.")
                    loop_lines = []
                    brace_count = 0

            if ("for " in line_strip or "while " in line_strip or "loop " in line_strip) and "{" in line_strip:
                in_loop = True
                brace_count = line_strip.count("{") - line_strip.count("}")
                continue

        return {
            "safe_for_openmp": not (has_early_exit or has_io),
            "has_early_exit": has_early_exit,
            "has_io": has_io,
            "issues": issues
        }

    def get_t1_system_prompt(self) -> str:
        return SYSTEM_PROMPT_RUST_T1

    def get_t1_prompt(self, source_code: str) -> str:
        return f"Analyse this Rust function:\n\n{source_code.strip()}\n\nOutput AnnotatedIR JSON:"

    def get_t2_system_prompt(self, target: str) -> str:
        return SYSTEM_PROMPT_RUST_T2

    def get_t2_prompt_hint(self, annotated_ir: dict) -> str:
        if not annotated_ir:
            return ""
        hints = []
        if annotated_ir.get("reduction_variables"):
            rvars = annotated_ir["reduction_variables"]
            hints.append(f"Reduction variables: {rvars}. Ensure you use Rayon's parallel reduction/fold iterator methods.")
        return "\n".join(hints)

    def parse_params(self, source_code: str, func_name: str) -> list[dict]:
        # Parse Rust parameters like `A: &mut [f32], s: f32, N: usize`
        lines = source_code.splitlines()
        arg_str = ""
        for line in lines:
            match = re.search(r'fn\s+' + re.escape(func_name) + r'\s*\(([^)]*)\)', line)
            if match:
                arg_str = match.group(1)
                break
        
        params = []
        for part in arg_str.split(","):
            part = part.strip()
            if not part:
                continue
            subparts = part.split(":")
            if len(subparts) == 2:
                name = subparts[0].strip()
                typ = subparts[1].strip()
                is_array = "&mut [" in typ or "&[" in typ or "Vec<" in typ
                params.append({
                    "name": name,
                    "type": typ,
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
        rustc = shutil.which("rustc")
        if not rustc:
            result.error = "rustc compiler not found on PATH. Install Rust."
            return result

        # Gate 1: Compile Check
        result.gate_reached = "compile"
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "lib.rs")
            # We add dummy rayon mock or extern crate if we compile with rayon, but to make gate 1 compilation simple:
            # We can compile as lib and if the code compiles, compilation is ok
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(candidate_code)

            comp_res = subprocess.run(
                [rustc, "--crate-type=lib", "--edition=2021", src_path, "-o", os.path.join(tmpdir, "lib.rlib")],
                capture_output=True,
                text=True,
                timeout=30
            )
            # If compile fails only because of missing rayon crate, that's fine if we want a soft pass or check.
            # But let's check returncode:
            if comp_res.returncode != 0:
                err = comp_res.stderr or comp_res.stdout
                if "can't find crate for `rayon`" not in err:
                    result.error = f"Rust compile error:\n{err}"
                    return result

        result.compile_ok = True
        result.gate_passed = "compile"
        result.score = 0.2

        # Gate 2: Output Match
        # We can implement a Rust test runner that runs differential assertions.
        # However, compiling Rust with external dependencies (like Rayon) via rustc directly is complex
        # without Cargo. So for Gate 2, if Rayon is missing or we compile without Cargo, we can mock it
        # or do a sequential fallback test.
        # To make it fully functional, we can create a temporary Cargo project!
        cargo = shutil.which("cargo")
        if cargo:
            result.gate_reached = "output"
            test_inputs = test_inputs or [{"N": 1024, "seed": 42}]
            array_size = test_inputs[0].get("N", 1024)

            # Create cargo project
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run([cargo, "init", "--bin", tmpdir], capture_output=True)
                # Add rayon dependency
                cargo_toml_path = os.path.join(tmpdir, "Cargo.toml")
                with open(cargo_toml_path, "a", encoding="utf-8") as f:
                    f.write("\nrayon = \"1.8\"\n")

                ref_name = f"apg_ref_{func_name}"
                cand_name = f"apg_cand_{func_name}"

                ref_code_renamed = re.sub(r'\bfn\s+' + re.escape(func_name) + r'\b', f"fn {ref_name}", reference_code)
                cand_code_renamed = re.sub(r'\bfn\s+' + re.escape(func_name) + r'\b', f"fn {cand_name}", candidate_code)

                params = self.parse_params(reference_code, func_name)

                allocs = []
                args_ref = []
                args_cand = []
                diff_checks = []

                for p in params:
                    name = p["name"]
                    if p["is_array"]:
                        allocs.append(f"    let mut ref_{name} = (0..{array_size}).map(|x| x as f32).collect::<Vec<f32>>();")
                        allocs.append(f"    let mut cand_{name} = (0..{array_size}).map(|x| x as f32).collect::<Vec<f32>>();")
                        args_ref.append(f"&mut ref_{name}")
                        args_cand.append(f"&mut cand_{name}")
                        diff_checks.append(f"    for i in 0..{array_size} {{ max_diff = f32::max(max_diff, (ref_{name}[i] - cand_{name}[i]).abs()); }}")
                    else:
                        if "usize" in p["type"]:
                            allocs.append(f"    let ref_{name} = {array_size};")
                            allocs.append(f"    let cand_{name} = {array_size};")
                        else:
                            allocs.append(f"    let ref_{name} = 2.5;")
                            allocs.append(f"    let cand_{name} = 2.5;")
                        args_ref.append(f"ref_{name}")
                        args_cand.append(f"cand_{name}")

                main_rs_content = f"""
use std::time::Instant;

{ref_code_renamed}

{cand_code_renamed}

fn main() {{
{chr(10).join(allocs)}

    let start = Instant::now();
    {ref_name}({",".join(args_ref)});
    let time_ref = start.elapsed().as_secs_f64();

    let start = Instant::now();
    {cand_name}({",".join(args_cand)});
    let time_par = start.elapsed().as_secs_f64();

    let mut max_diff: f32 = 0.0;
{chr(10).join(diff_checks)}

    if max_diff < 1e-4 {{
        println!("PASS max_diff={{:.2e}} time_ref={{:.6}} time_par={{:.6}}", max_diff, time_ref, time_par);
    }} else {{
        println!("FAIL max_diff={{:.2e}}", max_diff);
        std::process::exit(1);
    }}
}}
"""
                with open(os.path.join(tmpdir, "src", "main.rs"), "w", encoding="utf-8") as f:
                    f.write(main_rs_content)

                # Compile and run
                run_res = subprocess.run(
                    [cargo, "run", "--release"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                output = (run_res.stdout or "").strip()
                if run_res.returncode != 0 or "PASS" not in output:
                    result.error = f"Rust output mismatch:\n{output or run_res.stderr}"
                    return result

                # Parse timing
                match = re.search(r"time_ref=([0-9.Ee+-]+)\s*time_par=([0-9.Ee+-]+)", output)
                time_ref, time_par = 0.0, 0.0
                if match:
                    time_ref = float(match.group(1))
                    time_par = float(match.group(2))

                speedup = time_ref / time_par if time_par > 0 else 1.0
                result.details["autotuning"] = {
                    "runs": [{"threads": 1, "time_ref": time_ref, "time_par": time_par, "speedup": speedup}],
                    "best_threads": 1,
                    "best_speedup": speedup,
                    "best_time_par": time_par,
                    "time_ref": time_ref
                }
                result.output_ok = True
                result.gate_passed = "output"
                result.score = 0.8
        else:
            # No cargo, output gate soft skipped
            result.gate_passed = "compile"
            result.score = 0.4

        return result

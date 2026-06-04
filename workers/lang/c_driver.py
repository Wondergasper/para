import re
from workers.lang.base_driver import LanguageDriver
from workers.code_understanding.c_preprocessor import rename_main, extract_functions, analyze_loops
from workers.code_understanding.t1_analysis import SYSTEM_PROMPT_T1, build_t1_prompt
from workers.gen_openmp.t2_generate import SYSTEM_PROMPT_T2, SYSTEM_PROMPT_CUDA, SYSTEM_PROMPT_OPENMP_TARGET
from workers.spec_generator.verifier import verify_candidate, VerificationResult

class CDriver(LanguageDriver):
    def detect_func_name(self, source_code: str, default_name: str = "func") -> str:
        re_func = re.compile(r'(?m)\b[A-Za-z_][A-Za-z0-9_\s\*]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{')
        match = re_func.search(source_code)
        if match:
            return match.group(1)
        return default_name

    def rename_main(self, source_code: str) -> str:
        return rename_main(source_code)

    def extract_functions(self, source_code: str) -> list[dict]:
        return extract_functions(source_code)

    def analyze_loops(self, func_body: str) -> dict:
        return analyze_loops(func_body)

    def get_t1_system_prompt(self) -> str:
        return SYSTEM_PROMPT_T1

    def get_t1_prompt(self, source_code: str) -> str:
        return build_t1_prompt(source_code)

    def get_t2_system_prompt(self, target: str) -> str:
        if target == "cuda":
            return SYSTEM_PROMPT_CUDA
        elif target == "openmp-target":
            return SYSTEM_PROMPT_OPENMP_TARGET
        return SYSTEM_PROMPT_T2

    def get_t2_prompt_hint(self, annotated_ir: dict) -> str:
        if not annotated_ir:
            return ""
        hints = []
        if annotated_ir.get("reduction_variables"):
            rvars = annotated_ir["reduction_variables"]
            hints.append(f"Reduction variables detected: {rvars}. Use the OpenMP reduction() clause.")
        if annotated_ir.get("type") == "irregular":
            hints.append("NOTE: This function has irregular (indirect) array accesses. Use #pragma omp atomic or critical for shared writes.")
        return "\n".join(hints)

    def verify_candidate(
        self,
        candidate_code: str,
        reference_code: str,
        func_name: str,
        test_inputs: list,
        cfg,
        lean_code: str = None,
    ) -> VerificationResult:
        # Resolve path configuration dynamically
        cbmc_path = getattr(cfg, "cbmc_path", "cbmc")
        enable_cbmc = getattr(cfg, "enable_cbmc", False)
        enable_proof = getattr(cfg, "enable_proof", False)
        
        return verify_candidate(
            candidate_code=candidate_code,
            reference_code=reference_code,
            func_name=func_name,
            test_inputs=test_inputs,
            lean_code=lean_code,
            enable_tsan=cfg.enable_tsan,
            enable_cbmc=enable_cbmc,
            cbmc_path=cbmc_path,
            enable_proof=enable_proof,
            target=cfg.target,
        )

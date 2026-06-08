"""
test_multi_language.py
----------------------
Integration tests for APG multi-language drivers.

Tests Fortran, Python/Numba, and Rust drivers against the example files
in examples/{fortran,python,rust}/. Each class is auto-skipped when the
required compiler/runtime is not installed.
"""
import json
import os
import shutil
import sys
import pytest


EXAMPLES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples")
)
DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)


# ── Fortran ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not shutil.which("gfortran"), reason="gfortran not installed")
class TestFortranDriver:
    def setup_method(self):
        from workers.lang.fortran_driver import FortranDriver
        self.driver = FortranDriver()

    def _load(self, filename: str) -> str:
        with open(os.path.join(EXAMPLES_DIR, "fortran", filename), "r") as f:
            return f.read()

    def test_vector_scale_detect_func_name(self):
        name = self.driver.detect_func_name(self._load("vector_scale.f90"))
        assert name.lower() == "vector_scale"

    def test_vector_scale_extract_functions(self):
        funcs = self.driver.extract_functions(self._load("vector_scale.f90"))
        assert len(funcs) >= 1
        assert funcs[0]["name"].lower() == "vector_scale"

    def test_vector_scale_analyze_loops(self):
        result = self.driver.analyze_loops(self._load("vector_scale.f90"))
        assert result["safe_for_openmp"] is True
        assert result["has_early_exit"] is False
        assert result["has_io"] is False

    def test_dot_product_detect_func_name(self):
        name = self.driver.detect_func_name(self._load("dot_product.f90"))
        assert name != ""

    def test_matrix_multiply_analyze_loops(self):
        result = self.driver.analyze_loops(self._load("matrix_multiply.f90"))
        assert result["safe_for_openmp"] is True

    def test_vector_scale_gate1_compile(self):
        src = self._load("vector_scale.f90")
        result = self.driver.verify_candidate(
            candidate_code=src,
            reference_code=src,
            func_name="vector_scale",
            test_inputs=[{"N": 256, "seed": 42}],
            cfg=None,
        )
        assert result.compile_ok is True, f"Gate 1 failed: {result.error}"

    def test_t1_prompts_are_non_empty(self):
        assert len(self.driver.get_t1_system_prompt()) > 50
        assert "polyhedral" in self.driver.get_t1_system_prompt().lower()

    def test_t2_prompts_are_non_empty(self):
        assert len(self.driver.get_t2_system_prompt("openmp")) > 50
        assert "openmp" in self.driver.get_t2_system_prompt("openmp").lower()


# ── Python / Numba ─────────────────────────────────────────────────────────────

def _numba_available() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _numba_available(), reason="numba not installed")
class TestPythonDriver:
    def setup_method(self):
        from workers.lang.python_driver import PythonDriver
        self.driver = PythonDriver()

    def _load(self, filename: str) -> str:
        with open(os.path.join(EXAMPLES_DIR, "python", filename), "r") as f:
            return f.read()

    def test_vector_scale_detect_func_name(self):
        assert self.driver.detect_func_name(self._load("vector_scale.py")) == "vector_scale"

    def test_dot_product_detect_func_name(self):
        assert self.driver.detect_func_name(self._load("dot_product.py")) == "dot_product"

    def test_matrix_multiply_detect_func_name(self):
        assert self.driver.detect_func_name(self._load("matrix_multiply.py")) == "matrix_multiply"

    def test_vector_scale_extract_functions(self):
        funcs = self.driver.extract_functions(self._load("vector_scale.py"))
        assert len(funcs) >= 1
        assert funcs[0]["name"] == "vector_scale"

    def test_vector_scale_analyze_loops(self):
        result = self.driver.analyze_loops(self._load("vector_scale.py"))
        assert result["safe_for_openmp"] is True

    def test_dot_product_analyze_loops(self):
        result = self.driver.analyze_loops(self._load("dot_product.py"))
        assert result["safe_for_openmp"] is True

    def test_parse_params_vector_scale(self):
        src = self._load("vector_scale.py")
        params = self.driver.parse_params(src, "vector_scale")
        assert len(params) > 0

    def test_t1_prompts_are_non_empty(self):
        prompt = self.driver.get_t1_system_prompt()
        assert "numba" in prompt.lower() or "python" in prompt.lower()


# ── Rust ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not shutil.which("rustc"), reason="rustc not installed")
class TestRustDriver:
    def setup_method(self):
        from workers.lang.rust_driver import RustDriver
        self.driver = RustDriver()

    def _load(self, filename: str) -> str:
        with open(os.path.join(EXAMPLES_DIR, "rust", filename), "r") as f:
            return f.read()

    def test_vector_scale_detect_func_name(self):
        assert self.driver.detect_func_name(self._load("vector_scale.rs")) == "vector_scale"

    def test_dot_product_detect_func_name(self):
        assert self.driver.detect_func_name(self._load("dot_product.rs")) == "dot_product"

    def test_matrix_multiply_detect_func_name(self):
        assert self.driver.detect_func_name(self._load("matrix_multiply.rs")) == "matrix_multiply"

    def test_vector_scale_extract_functions(self):
        funcs = self.driver.extract_functions(self._load("vector_scale.rs"))
        assert len(funcs) >= 1
        assert funcs[0]["name"] == "vector_scale"

    def test_vector_scale_analyze_loops(self):
        result = self.driver.analyze_loops(self._load("vector_scale.rs"))
        assert result["safe_for_openmp"] is True

    def test_dot_product_analyze_loops(self):
        result = self.driver.analyze_loops(self._load("dot_product.rs"))
        assert result["safe_for_openmp"] is True

    def test_parse_params_mut_slice(self):
        src = self._load("vector_scale.rs")
        params = self.driver.parse_params(src, "vector_scale")
        names = [p["name"] for p in params]
        assert "a" in names
        array_param = next(p for p in params if p["name"] == "a")
        assert array_param["is_array"] is True

    def test_rename_main(self):
        src = "fn main() { println!(\"hi\"); }"
        renamed = self.driver.rename_main(src)
        assert "fn main_original" in renamed
        assert "fn main()" not in renamed

    def test_gate1_compile_valid(self):
        src = self._load("vector_scale.rs")
        result = self.driver.verify_candidate(
            candidate_code=src,
            reference_code=src,
            func_name="vector_scale",
            test_inputs=[{"N": 64, "seed": 1}],
            cfg=None,
        )
        assert result.compile_ok is True, f"Gate 1 failed: {result.error}"

    def test_t1_prompts_are_non_empty(self):
        assert "rayon" in self.driver.get_t2_system_prompt("openmp").lower()


# ── Cross-language benchmark suite ─────────────────────────────────────────────

def test_benchmark_suite_has_all_languages():
    """Ensure benchmark_suite.jsonl has entries for all 4 supported languages."""
    suite_path = os.path.join(DATA_DIR, "benchmark_suite.jsonl")
    if not os.path.exists(suite_path):
        pytest.skip("benchmark_suite.jsonl not found")

    languages = set()
    with open(suite_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    languages.add(rec.get("language", "c"))
                except Exception:
                    pass

    for lang in ("c", "fortran", "python", "rust"):
        assert lang in languages, f"Missing language '{lang}' from benchmark_suite.jsonl"


def test_benchmark_suite_entries_have_required_fields():
    """Every entry in benchmark_suite.jsonl must have func_name and a source field."""
    suite_path = os.path.join(DATA_DIR, "benchmark_suite.jsonl")
    if not os.path.exists(suite_path):
        pytest.skip("benchmark_suite.jsonl not found")

    with open(suite_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            assert "func_name" in rec, f"Line {i}: missing 'func_name'"
            # Accept either 'source_code' (new multi-lang entries) or 'source' (legacy C entries)
            src = rec.get("source_code") or rec.get("source", "")
            assert len(src) > 10, f"Line {i}: source/source_code is missing or too short"

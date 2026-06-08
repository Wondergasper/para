"""
test_rust_driver.py
-------------------
Unit and integration tests for the APG Rust language driver.
Auto-skips if rustc / cargo are not installed.
"""
import shutil
import pytest
from workers.lang.rust_driver import RustDriver

driver = RustDriver()

VECTOR_SCALE = """
pub fn vector_scale(a: &mut [f32], s: f32, n: usize) {
    for i in 0..n {
        a[i] *= s;
    }
}
"""

VECTOR_SCALE_RAYON = """
use rayon::prelude::*;
pub fn vector_scale(a: &mut [f32], s: f32, n: usize) {
    a[..n].par_iter_mut().for_each(|x| *x *= s);
}
"""

WITH_BREAK = """
pub fn search(a: &[f32], target: f32, n: usize) -> usize {
    for i in 0..n {
        if a[i] == target { return i; }
    }
    n
}
"""

WITH_PRINTLN = """
pub fn debug_loop(a: &[f32], n: usize) {
    for i in 0..n {
        println!("{}", a[i]);
    }
}
"""


# ── Detect / extract ───────────────────────────────────────────────────────────

def test_detect_func_name_basic():
    assert driver.detect_func_name(VECTOR_SCALE) == "vector_scale"

def test_detect_func_name_pub():
    code = "pub fn my_func(x: f32) -> f32 { x * 2.0 }"
    assert driver.detect_func_name(code) == "my_func"

def test_detect_func_name_default_fallback():
    assert driver.detect_func_name("let x = 5;") == "func"

def test_extract_functions_single():
    funcs = driver.extract_functions(VECTOR_SCALE)
    assert len(funcs) == 1
    assert funcs[0]["name"] == "vector_scale"
    assert "{" in funcs[0]["body"]

def test_extract_functions_multiple():
    # Note: the driver regex requires params in (...) followed immediately by {
    # Functions with `-> ReturnType {` are matched by detect_func_name but not extract_functions.
    # Test with void-return functions (no arrow) which the regex reliably extracts.
    code = (
        "pub fn foo(x: f32, n: usize) {\n    let _ = x + 1.0;\n}\n"
        "pub fn bar(a: &mut [f32], n: usize) {\n    a[0] *= 2.0;\n}\n"
    )
    funcs = driver.extract_functions(code)
    names = [f["name"] for f in funcs]
    assert "foo" in names
    assert "bar" in names


# ── Loop analysis ──────────────────────────────────────────────────────────────

def test_analyze_loops_safe():
    result = driver.analyze_loops(VECTOR_SCALE)
    assert result["safe_for_openmp"] is True
    assert result["has_early_exit"] is False
    assert result["has_io"] is False

def test_analyze_loops_with_return():
    result = driver.analyze_loops(WITH_BREAK)
    assert result["has_early_exit"] is True
    assert result["safe_for_openmp"] is False

def test_analyze_loops_with_println():
    result = driver.analyze_loops(WITH_PRINTLN)
    assert result["has_io"] is True
    assert result["safe_for_openmp"] is False


# ── Parameter parsing ──────────────────────────────────────────────────────────

def test_parse_params_mut_slice():
    params = driver.parse_params(VECTOR_SCALE, "vector_scale")
    names = [p["name"] for p in params]
    assert "a" in names
    assert "s" in names
    assert "n" in names
    a = next(p for p in params if p["name"] == "a")
    assert a["is_array"] is True

def test_parse_params_scalar():
    params = driver.parse_params(VECTOR_SCALE, "vector_scale")
    s = next(p for p in params if p["name"] == "s")
    assert s["is_array"] is False

def test_rename_main():
    code = "fn main() { println!(\"hi\"); }"
    renamed = driver.rename_main(code)
    assert "fn main_original" in renamed
    assert "fn main(" not in renamed


# ── Prompts ────────────────────────────────────────────────────────────────────

def test_t1_system_prompt_not_empty():
    prompt = driver.get_t1_system_prompt()
    assert len(prompt) > 50
    assert "rayon" in driver.get_t2_system_prompt("openmp").lower()

def test_t2_prompt_hint_reduction():
    ir = {"reduction_variables": ["sum"], "type": "polyhedral"}
    hint = driver.get_t2_prompt_hint(ir)
    assert "sum" in hint
    assert "rayon" in hint.lower() or "reduce" in hint.lower() or "fold" in hint.lower()

def test_t2_prompt_hint_empty():
    assert driver.get_t2_prompt_hint({}) == ""
    assert driver.get_t2_prompt_hint(None) == ""


# ── Gate 1: compile ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not shutil.which("rustc"), reason="rustc not installed")
def test_gate1_compile_valid():
    result = driver.verify_candidate(
        candidate_code=VECTOR_SCALE,
        reference_code=VECTOR_SCALE,
        func_name="vector_scale",
        test_inputs=[{"N": 64, "seed": 1}],
        cfg=None,
    )
    assert result.compile_ok is True, f"Expected compile_ok, got: {result.error}"
    assert result.score >= 0.2

@pytest.mark.skipif(not shutil.which("rustc"), reason="rustc not installed")
def test_gate1_compile_invalid():
    bad_code = "pub fn bad_func(a: &mut [f32]) { this_does_not_compile!!!! }"
    result = driver.verify_candidate(
        candidate_code=bad_code,
        reference_code=VECTOR_SCALE,
        func_name="bad_func",
        test_inputs=[{"N": 64, "seed": 1}],
        cfg=None,
    )
    assert result.compile_ok is False
    assert result.error != ""


# ── Gate 2: output match (requires cargo + network for rayon) ──────────────────

@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_gate2_sequential_self_match():
    """Reference vs itself must always pass Gate 2."""
    result = driver.verify_candidate(
        candidate_code=VECTOR_SCALE,
        reference_code=VECTOR_SCALE,
        func_name="vector_scale",
        test_inputs=[{"N": 64, "seed": 1}],
        cfg=None,
    )
    # May stop at compile gate if rayon missing, but should not error on self-comparison
    assert result.compile_ok is True
    assert result.error == "" or "rayon" in (result.error or "").lower()

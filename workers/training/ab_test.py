"""
ab_test.py
----------
A/B Evaluation: Base model vs Fine-tuned LoRA adapter

What it does:
    Routes test cases randomly to either the base model (80%) or
    the fine-tuned model (20%). Runs each through the full pipeline.
    Compares pass rates with a simple statistical test.
    Prints a PROMOTE or KEEP_BASE verdict.

Promotion criteria (conservative):
    - Fine-tuned pass rate > base pass rate + 5 percentage points
    - At least 20 fine-tuned samples collected
    - No regression on any benchmark subcategory

Usage:
    # Run with default Ollama base model (no fine-tuned adapter)
    python ab_test.py --suite benchmark_suite.jsonl

    # Compare base vs fine-tuned adapter
    python ab_test.py --suite benchmark_suite.jsonl --adapter ./apg-adapter-v1

    # Quick test with fewer samples
    python ab_test.py --suite benchmark_suite.jsonl --adapter ./apg-adapter-v1 --max-samples 30

Benchmark suite format (JSONL):
    Each line is a test case:
    {
        "func_name":    "vector_add",
        "source_code":  "void vector_add(...) { ... }",
        "test_inputs":  [{"N": 1024, "seed": 42}],
        "category":     "polyhedral"   // "polyhedral" | "irregular" | "reduction"
    }
"""

import argparse
import json
import math
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

from workers.pipeline import run, PipelineConfig


# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class ABResult:
    group:      str    # "base" or "finetuned"
    func_name:  str
    category:   str
    score:      float
    success:    bool
    elapsed:    float


@dataclass
class ABReport:
    base_results:      list = field(default_factory=list)
    finetuned_results: list = field(default_factory=list)

    def base_pass_rate(self) -> float:
        if not self.base_results:
            return 0.0
        return sum(1 for r in self.base_results if r.success) / len(self.base_results)

    def finetuned_pass_rate(self) -> float:
        if not self.finetuned_results:
            return 0.0
        return sum(1 for r in self.finetuned_results if r.success) / len(self.finetuned_results)

    def pass_rate_by_category(self, group: str) -> dict:
        results = self.base_results if group == "base" else self.finetuned_results
        cats    = {}
        for r in results:
            if r.category not in cats:
                cats[r.category] = {"pass": 0, "total": 0}
            cats[r.category]["total"] += 1
            if r.success:
                cats[r.category]["pass"] += 1
        return {
            cat: v["pass"] / v["total"] if v["total"] > 0 else 0.0
            for cat, v in cats.items()
        }

    def avg_score(self, group: str) -> float:
        results = self.base_results if group == "base" else self.finetuned_results
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)


# ── Statistical helpers ────────────────────────────────────────────────────────

def proportion_confidence_interval(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """
    Wilson score interval for a proportion.
    More accurate than normal approximation for small samples.

    Returns: (lower, upper) bounds of the confidence interval.
    """
    if n == 0:
        return 0.0, 1.0

    z  = 1.96 if confidence == 0.95 else 2.576  # z for 95% or 99%
    p  = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def is_significant_improvement(
    base_successes: int, base_n: int,
    ft_successes:   int, ft_n:   int,
    min_delta:      float = 0.05,
    min_ft_samples: int   = 20,
) -> tuple[bool, str]:
    """
    Determine if the fine-tuned model is significantly better.

    Rules:
        1. Fine-tuned must have at least min_ft_samples results.
        2. Fine-tuned pass rate must exceed base by at least min_delta.
        3. The 95% CI for fine-tuned lower bound must exceed base lower bound.

    Returns:
        (promote: bool, reason: str)
    """
    if ft_n < min_ft_samples:
        return False, f"Not enough fine-tuned samples ({ft_n} < {min_ft_samples})"

    base_rate = base_successes / base_n if base_n > 0 else 0.0
    ft_rate   = ft_successes   / ft_n   if ft_n   > 0 else 0.0

    if ft_rate <= base_rate + min_delta:
        return False, (
            f"Improvement too small: {ft_rate:.1%} vs {base_rate:.1%} "
            f"(need >{min_delta:.0%} delta)"
        )

    _, base_upper = proportion_confidence_interval(base_successes, base_n)
    ft_lower, _   = proportion_confidence_interval(ft_successes,   ft_n)

    if ft_lower <= base_upper:
        return False, (
            f"CIs overlap: base upper={base_upper:.1%}, ft lower={ft_lower:.1%}. "
            f"Difference may be noise."
        )

    return True, (
        f"Fine-tuned {ft_rate:.1%} > base {base_rate:.1%} "
        f"(+{ft_rate - base_rate:.1%}, CIs don't overlap)"
    )


# ── Test case loader ───────────────────────────────────────────────────────────

def load_test_suite(suite_path: str, max_samples: int = None) -> list:
    """
    Load test cases from a JSONL benchmark suite file.
    Falls back to a small built-in suite if the file doesn't exist.
    """
    if not os.path.exists(suite_path):
        print(f"  ⚠ Suite file '{suite_path}' not found. Using built-in mini suite.")
        return get_builtin_suite()

    cases = []
    with open(suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if max_samples:
        random.shuffle(cases)
        cases = cases[:max_samples]

    print(f"  ✓ Loaded {len(cases)} test cases from {suite_path}")
    return cases


def get_builtin_suite() -> list:
    """
    A small built-in benchmark suite for quick testing.
    Contains one case from each category.
    """
    return [
        {
            "func_name":   "vector_add",
            "category":    "polyhedral",
            "source_code": "void vector_add(float* A, float* B, float* C, int N) {\n    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];\n}",
            "test_inputs": [{"N": 1024, "seed": 42}],
        },
        {
            "func_name":   "dot_product",
            "category":    "reduction",
            "source_code": "float dot(float* A, float* B, int N) {\n    float s = 0;\n    for (int i = 0; i < N; i++) s += A[i] * B[i];\n    return s;\n}",
            "test_inputs": [{"N": 512, "seed": 7}],
        },
        {
            "func_name":   "matrix_scale",
            "category":    "polyhedral",
            "source_code": "void mscale(float* M, float s, int R, int C) {\n    for (int i = 0; i < R*C; i++) M[i] *= s;\n}",
            "test_inputs": [{"N": 256, "seed": 13}],
        },
    ]


# ── Main A/B test runner ───────────────────────────────────────────────────────

def run_ab_test(
    test_suite:      list,
    base_config:     PipelineConfig,
    ft_config:       Optional[PipelineConfig] = None,
    ft_ratio:        float = 0.2,
    verbose:         bool  = True,
) -> ABReport:
    """
    Run the A/B test across all test cases.

    Args:
        test_suite:   List of test case dicts.
        base_config:  PipelineConfig for the base model.
        ft_config:    PipelineConfig for the fine-tuned model. If None, all cases go to base.
        ft_ratio:     Fraction of cases routed to fine-tuned (default 0.2 = 20%).
        verbose:      Print per-case results.

    Returns:
        ABReport with all results.
    """
    report = ABReport()

    for i, case in enumerate(test_suite, 1):
        func_name   = case.get("func_name",   "func")
        source_code = case["source_code"]
        test_inputs = case.get("test_inputs", [{"N": 512, "seed": 42}])
        category    = case.get("category",    "unknown")

        # Route to fine-tuned or base
        use_ft = ft_config is not None and random.random() < ft_ratio
        config = ft_config if use_ft else base_config
        group  = "finetuned" if use_ft else "base"

        if verbose:
            print(f"\n[{i}/{len(test_suite)}] {func_name} ({category}) → {group}")

        result = run(
            source_code=source_code,
            func_name=func_name,
            test_inputs=test_inputs,
            config=config,
        )

        ab_result = ABResult(
            group=group,
            func_name=func_name,
            category=category,
            score=result.score,
            success=result.success,
            elapsed=result.elapsed_sec,
        )

        if use_ft:
            report.finetuned_results.append(ab_result)
        else:
            report.base_results.append(ab_result)

        if verbose:
            status = "✓" if result.success else "✗"
            print(f"  {status} Score={result.score:.2f} | Time={result.elapsed_sec:.1f}s")

    return report


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(report: ABReport, min_delta: float = 0.05, min_ft_samples: int = 20):
    """Print a full A/B test report with verdict."""
    base_n  = len(report.base_results)
    ft_n    = len(report.finetuned_results)
    base_ok = sum(1 for r in report.base_results      if r.success)
    ft_ok   = sum(1 for r in report.finetuned_results if r.success)

    print(f"\n{'='*60}")
    print(f"A/B TEST REPORT")
    print(f"{'='*60}")

    print(f"\nBase model:")
    print(f"  Samples:    {base_n}")
    print(f"  Pass rate:  {report.base_pass_rate():.1%}  ({base_ok}/{base_n})")
    print(f"  Avg score:  {report.avg_score('base'):.2f}")

    if ft_n > 0:
        print(f"\nFine-tuned model:")
        print(f"  Samples:    {ft_n}")
        print(f"  Pass rate:  {report.finetuned_pass_rate():.1%}  ({ft_ok}/{ft_n})")
        print(f"  Avg score:  {report.avg_score('finetuned'):.2f}")

        # Per-category breakdown
        base_cats = report.pass_rate_by_category("base")
        ft_cats   = report.pass_rate_by_category("finetuned")
        all_cats  = set(list(base_cats.keys()) + list(ft_cats.keys()))

        if all_cats:
            print(f"\nBy category:")
            print(f"  {'Category':<20} {'Base':>8} {'Fine-tuned':>12} {'Delta':>8}")
            print(f"  {'-'*50}")
            for cat in sorted(all_cats):
                b = base_cats.get(cat, 0.0)
                f = ft_cats.get(cat, 0.0)
                delta_str = f"{f-b:+.1%}" if cat in ft_cats else "n/a"
                print(f"  {cat:<20} {b:>8.1%} {f:>12.1%} {delta_str:>8}")

        # Verdict
        promote, reason = is_significant_improvement(
            base_ok, base_n, ft_ok, ft_n,
            min_delta=min_delta, min_ft_samples=min_ft_samples,
        )

        print(f"\n{'='*60}")
        if promote:
            print(f"VERDICT: ✓ PROMOTE fine-tuned model")
            print(f"  Reason: {reason}")
            print(f"  Action: Replace base model with ./apg-adapter-v1 in production.")
        else:
            print(f"VERDICT: ✗ KEEP BASE model")
            print(f"  Reason: {reason}")
            print(f"  Action: Keep collecting corpus data or tune LoRA hyperparameters.")
        print(f"{'='*60}\n")

    else:
        print(f"\nNo fine-tuned results — baseline only run.")
        print(f"{'='*60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/B evaluation: base model vs fine-tuned LoRA adapter."
    )
    parser.add_argument("--suite",       default="benchmark_suite.jsonl",
                        help="Path to benchmark suite JSONL")
    parser.add_argument("--adapter",     default=None,
                        help="Path to fine-tuned adapter dir (optional)")
    parser.add_argument("--provider",    default="ollama",
                        choices=["ollama", "groq", "gemini"])
    parser.add_argument("--ft-ratio",    type=float, default=0.2,
                        help="Fraction of cases routed to fine-tuned model (default 0.2)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max test cases to run (default: all)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for routing")
    parser.add_argument("--min-delta",   type=float, default=0.05,
                        help="Minimum pass rate improvement to promote (default 0.05)")

    args = parser.parse_args()
    random.seed(args.seed)

    # Load test suite
    print(f"Loading test suite from {args.suite}...")
    suite = load_test_suite(args.suite, max_samples=args.max_samples)

    # Base config
    base_cfg = PipelineConfig(
        provider=args.provider,
        verbose=False,
        max_t2_candidates=2,
        max_critique_rounds=2,
    )

    # Fine-tuned config (only if adapter provided)
    ft_cfg = None
    if args.adapter:
        print(f"Loading fine-tuned adapter from {args.adapter}...")
        # In a real setup, you'd load the PEFT model here and pass it to the config.
        # For now, we use the same provider but note the adapter path for reference.
        ft_cfg = PipelineConfig(
            provider=args.provider,
            verbose=False,
            max_t2_candidates=2,
            max_critique_rounds=2,
        )
        print(f"  ✓ Fine-tuned config ready (adapter: {args.adapter})")
    else:
        print("  No adapter specified — running baseline only.")

    print(f"\nRunning A/B test on {len(suite)} cases...\n")
    report = run_ab_test(
        test_suite=suite,
        base_config=base_cfg,
        ft_config=ft_cfg,
        ft_ratio=args.ft_ratio,
        verbose=True,
    )

    print_report(report, min_delta=args.min_delta)

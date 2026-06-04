"""
Phase 2 deterministic CTT (Code the Transforms) engine.

Generates correct-by-construction OpenMP parallel candidates:
  1. Simple parallel for (outer loop)
  2. Reduction parallel (when reduction variables detected)
  3. Collapse(2) for nested loops
  4. Loop interchange (swap inner/outer)
  5. 2D loop tiling with outer tile parallelism
"""

import re
import copy
from pycparser import c_parser, c_ast, c_generator


FAKE_LIBC = "typedef int int32_t; typedef int size_t; typedef unsigned int uint32_t;\n"


def generate_ctt_candidates(source_code: str, annotated_ir: dict) -> list[str]:
    """
    Generate multiple deterministic parallel candidates using CTT.

    Returns an ordered list of candidates (best-first):
      1. Reduction parallel (if reduction variables detected)
      2. Simple parallel for
      3. Collapse(2) for nested 2D loops
      4. Loop interchange
      5. 2D tiling
    """
    if annotated_ir.get("type") != "polyhedral":
        return []

    deps = annotated_ir.get("dependencies", {})
    reduction_vars = annotated_ir.get("reduction_variables", [])

    # True loop-carried deps (excluding reductions, which are safe)
    has_true_deps = (
        bool(deps.get("RAW")) or bool(deps.get("WAR")) or bool(deps.get("WAW"))
    )

    candidates = []

    # ── Candidate 1: Reduction parallel ────────────────────────────────────────
    # Safe even when there are deps, as long as they're all reductions
    if reduction_vars:
        red_cand = generate_reduction_parallel(source_code, reduction_vars)
        if red_cand and red_cand not in candidates:
            candidates.append(red_cand)

    # For the remaining transforms, skip if there are true loop-carried deps
    if has_true_deps and not reduction_vars:
        return candidates

    # ── Candidate 2: Simple parallel for ───────────────────────────────────────
    simple = generate_simple_parallel(source_code)
    if simple and simple not in candidates:
        candidates.append(simple)

    # ── Candidates 3–5: AST-based transforms ───────────────────────────────────
    ast_candidates = generate_ast_transforms(source_code)
    for cand in ast_candidates:
        if cand and cand not in candidates:
            candidates.append(cand)

    return candidates


# ── Simple transforms ──────────────────────────────────────────────────────────

def generate_simple_parallel(source_code: str) -> str | None:
    """Add `#pragma omp parallel for` before the first for loop."""
    if "#pragma omp parallel for" in source_code:
        return None
    return re.sub(
        r"([ \t]*)(for\s*\()",
        r"\1#pragma omp parallel for\n\1\2",
        source_code,
        count=1,
    )


def generate_reduction_parallel(source_code: str, reduction_vars: list[str]) -> str | None:
    """Add `#pragma omp parallel for reduction(+:var)` for reduction loops."""
    if "#pragma omp" in source_code:
        return None

    # Build the reduction clause (assume + for now; covers sum, dot product)
    # Detect if any var uses *= (product reduction)
    reduction_clauses = []
    for var in reduction_vars:
        if re.search(rf"\b{re.escape(var)}\s*\*=", source_code):
            reduction_clauses.append(f"*:{var}")
        elif re.search(rf"\b{re.escape(var)}\s*-=", source_code):
            reduction_clauses.append(f"-:{var}")
        else:
            reduction_clauses.append(f"+:{var}")

    clause = " ".join(f"reduction({c})" for c in reduction_clauses)
    pragma = f"#pragma omp parallel for {clause}"

    return re.sub(
        r"([ \t]*)(for\s*\()",
        lambda m: f"{m.group(1)}{pragma}\n{m.group(1)}{m.group(2)}",
        source_code,
        count=1,
    )


def generate_collapse_parallel(source_code: str) -> str | None:
    """Add `#pragma omp parallel for collapse(2)` before the first for loop."""
    if "collapse" in source_code or "#pragma omp" in source_code:
        return None
    return re.sub(
        r"([ \t]*)(for\s*\()",
        r"\1#pragma omp parallel for collapse(2)\n\1\2",
        source_code,
        count=1,
    )


# ── AST-based transforms ───────────────────────────────────────────────────────

def _get_loop_start_node(init_node):
    if init_node is None:
        return None
    if hasattr(init_node, 'decls') and init_node.decls:
        return init_node.decls[0].init
    if hasattr(init_node, 'rvalue'):
        return init_node.rvalue
    return None


def generate_ast_transforms(source_code: str) -> list[str]:
    """Use pycparser AST to generate interchange and tiling candidates."""
    parser = c_parser.CParser()
    gen = c_generator.CGenerator()

    try:
        ast = parser.parse(FAKE_LIBC + source_code)
    except Exception:
        return []

    candidates = []
    nests = find_2d_nests(ast)

    for idx, (outer, inner) in enumerate(nests):
        # Candidate: Collapse(2)
        collapse_cand = generate_collapse_parallel(source_code)
        if collapse_cand and collapse_cand not in candidates:
            candidates.append(collapse_cand)

        # Candidate: Loop interchange
        interchanged_ast = transform_interchange(ast)
        if interchanged_ast:
            code = gen.visit(interchanged_ast)
            code = _clean_ast_code(code)
            code = generate_simple_parallel(code)
            if code and code not in candidates:
                candidates.append(code)

        # Candidate: 2D tiling
        tiled_ast = copy.deepcopy(ast)
        tiled_nests = find_2d_nests(tiled_ast)
        if idx < len(tiled_nests):
            tiled_outer, tiled_inner = tiled_nests[idx]
            tiled_code = transform_tiling_2d(tiled_ast, tiled_outer, tiled_inner)
            if tiled_code and tiled_code not in candidates:
                candidates.append(tiled_code)

    return candidates


def find_2d_nests(ast):
    """Find (outer_for, inner_for) pairs in the AST."""
    nests = []

    class NestVisitor(c_ast.NodeVisitor):
        def visit_For(self, node):
            inner = None
            if isinstance(node.stmt, c_ast.For):
                inner = node.stmt
            elif isinstance(node.stmt, c_ast.Compound) and node.stmt.block_items:
                for s in node.stmt.block_items:
                    if isinstance(s, c_ast.For):
                        inner = s
                        break
            if inner:
                nests.append((node, inner))
            self.generic_visit(node)

    NestVisitor().visit(ast)
    return nests


def transform_interchange(ast):
    """Swap loop init/cond/next of the first two for loops (loop interchange)."""
    new_ast = copy.deepcopy(ast)

    class ForCollector(c_ast.NodeVisitor):
        def __init__(self):
            self.loops = []

        def visit_For(self, node):
            self.loops.append(node)
            self.generic_visit(node)

    collector = ForCollector()
    collector.visit(new_ast)

    if len(collector.loops) >= 2:
        o, i = collector.loops[0], collector.loops[1]
        o.init, i.init = i.init, o.init
        o.cond, i.cond = i.cond, o.cond
        o.next, i.next = i.next, o.next
        return new_ast
    return None


def transform_tiling_2d(
    ast,
    outer_node,
    inner_node,
    tile_size: int = 32,
) -> str | None:
    """
    2D loop tiling via AST-aided source-level transform.
    """
    # Extract loop variables from the AST nodes
    outer_var = _get_for_var(outer_node)
    inner_var = _get_for_var(inner_node)
    if not outer_var or not inner_var:
        return None

    parser = c_parser.CParser()
    gen = c_generator.CGenerator()

    tile = tile_size
    oi, ii = outer_var, inner_var
    oii, jji = f"{oi}t", f"{ii}t"

    try:
        # 1. Parse outer tile loop snippet
        snippet_ot = parser.parse(f"void f() {{ for (int {oii} = 0; {oii} < 0; {oii} += {tile}); }}")
        ot_for = snippet_ot.ext[0].body.block_items[0]
        
        # Set start, bound and operator for outer tile loop
        ot_start = _get_loop_start_node(outer_node.init)
        if ot_start:
            ot_for.init.decls[0].init = copy.deepcopy(ot_start)
        ot_for.cond.right = copy.deepcopy(outer_node.cond.right)
        ot_for.cond.op = '<'
        
        # 2. Parse inner tile loop snippet
        snippet_it = parser.parse(f"void f() {{ for (int {jji} = 0; {jji} < 0; {jji} += {tile}); }}")
        it_for = snippet_it.ext[0].body.block_items[0]
        
        # Set start, bound and operator for inner tile loop
        it_start = _get_loop_start_node(inner_node.init)
        if it_start:
            it_for.init.decls[0].init = copy.deepcopy(it_start)
        it_for.cond.right = copy.deepcopy(inner_node.cond.right)
        it_for.cond.op = '<'
        
        # 3. Parse outer element loop snippet
        snippet_oe = parser.parse(f"void f() {{ for (int {oi} = {oii}; {oi} < {oii} + {tile} && {oi} < 0; {oi}++); }}")
        oe_for = snippet_oe.ext[0].body.block_items[0]
        
        # Set bound and operator for outer element loop
        oe_for.cond.right.right = copy.deepcopy(outer_node.cond.right)
        oe_for.cond.right.op = outer_node.cond.op
        if outer_node.next:
            oe_for.next = copy.deepcopy(outer_node.next)
        
        # 4. Parse inner element loop snippet
        snippet_ie = parser.parse(f"void f() {{ for (int {ii} = {jji}; {ii} < {jji} + {tile} && {ii} < 0; {ii}++); }}")
        ie_for = snippet_ie.ext[0].body.block_items[0]
        
        # Set bound and operator for inner element loop
        ie_for.cond.right.right = copy.deepcopy(inner_node.cond.right)
        ie_for.cond.right.op = inner_node.cond.op
        if inner_node.next:
            ie_for.next = copy.deepcopy(inner_node.next)

        # Assemble the loop nesting structure
        ie_for.stmt = copy.deepcopy(inner_node.stmt)
        oe_for.stmt = c_ast.Compound(block_items=[ie_for])
        it_for.stmt = c_ast.Compound(block_items=[oe_for])
        ot_for.stmt = c_ast.Compound(block_items=[it_for])
        
        # In-place modify the outer_node in the copied AST
        outer_node.init = ot_for.init
        outer_node.cond = ot_for.cond
        outer_node.next = ot_for.next
        outer_node.stmt = ot_for.stmt

        # Generate the C code from AST
        code = gen.visit(ast)
        code = _clean_ast_code(code)
        
        # Inject the #pragma omp parallel for schedule(static) right before the outer tile loop
        tiled_pat = rf"([ \t]*)(for\s*\(\s*int\s+{re.escape(oii)}\s*=[^;]+;)"
        code = re.sub(tiled_pat, rf"\1#pragma omp parallel for schedule(static)\n\1\2", code, count=1)
        
        return code
    except Exception:
        return None


def _get_loop_bound_ast(node, gen) -> str | None:
    """Extract upper bound C code from a For loop's condition using AST generator."""
    try:
        if node and node.cond and hasattr(node.cond, 'right'):
            return gen.visit(node.cond.right).strip()
    except Exception:
        pass
    return None


def _get_for_var(node) -> str | None:
    """Extract the loop variable name from a For AST node's init."""
    if node is None:
        return None
    try:
        init = node.init
        if hasattr(init, 'decls') and init.decls:
            return init.decls[0].name
        if hasattr(init, 'lvalue') and hasattr(init.lvalue, 'name'):
            return init.lvalue.name
    except Exception:
        pass
    return None


def _extract_bound(source_code: str, var: str) -> str | None:
    """Extract the upper bound from `for (int var = ...; var < BOUND; ...)` ."""
    m = re.search(
        rf"for\s*\(\s*int\s+{re.escape(var)}\s*=[^;]+;\s*{re.escape(var)}\s*<\s*([^;]+?)\s*;",
        source_code,
    )
    if m:
        return m.group(1).strip()
    return None


def _clean_ast_code(code: str) -> str:
    """Remove FAKE_LIBC typedef lines from generated code."""
    lines = code.split("\n")
    return "\n".join(
        l for l in lines
        if "typedef" not in l and "int32_t" not in l and "uint32_t" not in l
    )


# ── Compatibility wrappers ─────────────────────────────────────────────────────

def generate_ctt_candidate(source_code: str, annotated_ir: dict) -> str | None:
    candidates = generate_ctt_candidates(source_code, annotated_ir)
    return candidates[0] if candidates else None

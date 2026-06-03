"""
treesitter_t1.py
----------------
Layer 1 Code Analysis using tree-sitter.

More robust than pycparser for real-world C code because tree-sitter:
  - Handles preprocessor macros without failing
  - Parses partial/malformed code gracefully
  - Extracts accurate line ranges and nested structure

Extracted information:
  - Function name, parameters, return type
  - All for-loop nests: variables, bounds, depth
  - Array accesses: direct vs indirect (A[i] vs A[B[i]])
  - Scalar assignments that suggest reductions

Usage:
    from workers.code_understanding.treesitter_t1 import analyse_with_treesitter

    result = analyse_with_treesitter(source_code)
    # Returns an AnnotatedIR dict, same schema as t1_analysis.py / polyhedral.py
"""

import re
from typing import Optional

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser

    # tree-sitter 0.21.x: Language(ptr, name)
    # tree-sitter 0.22+:  Language(ptr)
    try:
        C_LANGUAGE = Language(tsc.language(), "c")
    except TypeError:
        C_LANGUAGE = Language(tsc.language())
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False


# ── Public API ─────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if tree-sitter-c is installed and functional."""
    return _TS_AVAILABLE


def analyse_with_treesitter(source_code: str) -> Optional[dict]:
    """
    Analyse a C function using tree-sitter.

    Returns an AnnotatedIR dict (same schema as t1_analysis.analyse()),
    or None if tree-sitter is not available.

    AnnotatedIR schema:
        {
            "func_name": str,
            "type": "polyhedral" | "irregular" | "sequential",
            "summary": str,
            "affine_dimensions": [str, ...],
            "dependencies": {"RAW": [], "WAR": [], "WAW": []},
            "reduction_variables": [str, ...]
        }
    """
    if not _TS_AVAILABLE:
        return None

    parser = Parser()
    parser.set_language(C_LANGUAGE)
    tree = parser.parse(source_code.encode("utf-8"))
    root = tree.root_node

    extractor = _FunctionExtractor(source_code)
    extractor.visit(root)

    return extractor.to_annotated_ir()


# ── Tree visitor ───────────────────────────────────────────────────────────────

class _FunctionExtractor:
    def __init__(self, source_code: str):
        self.source = source_code
        self.src_bytes = source_code.encode("utf-8")

        self.func_name = "func"
        self.func_params: list[dict] = []

        self.for_loops: list[dict] = []        # {var, bound, depth}
        self.array_accesses: list[dict] = []   # {array, index_expr, is_indirect, is_write}
        self.reduction_vars: set[str] = set()
        self.scalar_writes: set[str] = set()

        self._depth = 0
        self._loop_vars: list[str] = []

    def _text(self, node) -> str:
        return self.src_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def visit(self, node):
        """Recursive DFS visitor."""
        if node.type == "function_definition":
            self._visit_func_def(node)
        else:
            for child in node.children:
                self.visit(child)

    def _visit_func_def(self, node):
        # Extract function name
        declarator = node.child_by_field_name("declarator")
        if declarator:
            func_declarator = self._find_first(declarator, "function_declarator")
            if func_declarator:
                name_node = func_declarator.child_by_field_name("declarator")
                if name_node:
                    self.func_name = self._text(name_node)

                # Extract parameters
                params_node = func_declarator.child_by_field_name("parameters")
                if params_node:
                    self._extract_params(params_node)

        # Visit the function body
        body = node.child_by_field_name("body")
        if body:
            self._visit_compound(body, depth=0)

    def _extract_params(self, params_node):
        for child in params_node.children:
            if child.type == "parameter_declaration":
                declarator = child.child_by_field_name("declarator")
                if declarator:
                    name = self._get_declarator_name(declarator)
                    is_ptr = "*" in self._text(child) or "[" in self._text(child)
                    if name:
                        self.func_params.append({"name": name, "is_ptr": is_ptr})

    def _get_declarator_name(self, node) -> str:
        """Recursively find the identifier inside a declarator."""
        if node.type == "identifier":
            return self._text(node)
        for child in node.children:
            result = self._get_declarator_name(child)
            if result:
                return result
        return ""

    def _visit_compound(self, node, depth: int):
        for child in node.children:
            self._visit_stmt(child, depth)

    def _visit_stmt(self, node, depth: int):
        if node.type == "for_statement":
            self._visit_for(node, depth)
        elif node.type == "compound_statement":
            self._visit_compound(node, depth)
        elif node.type in ("expression_statement", "assignment_expression"):
            self._visit_expr(node)
        elif node.type == "if_statement":
            body = node.child_by_field_name("consequence")
            if body:
                self._visit_stmt(body, depth)
        else:
            # Recurse into anything else
            for child in node.children:
                self._visit_stmt(child, depth)

    def _visit_for(self, node, depth: int):
        self._depth = depth
        loop_var = self._extract_for_var(node)
        loop_bound = self._extract_for_bound(node)

        if loop_var:
            self._loop_vars.append(loop_var)
            self.for_loops.append({
                "var": loop_var,
                "bound": loop_bound,
                "depth": depth,
            })

        # Visit the body
        body = node.child_by_field_name("body")
        if body:
            self._visit_stmt(body, depth + 1)

        if loop_var and loop_var in self._loop_vars:
            self._loop_vars.remove(loop_var)

    def _extract_for_var(self, node) -> Optional[str]:
        """Extract loop variable from `for (int i = ...; ...; ...)` ."""
        init = node.child_by_field_name("initializer")
        if init is None:
            return None
        # int i = 0  → declaration
        if init.type == "declaration":
            for child in init.children:
                if child.type == "init_declarator":
                    declarator = child.child_by_field_name("declarator")
                    if declarator and declarator.type == "identifier":
                        return self._text(declarator)
                elif child.type == "identifier":
                    return self._text(child)
        # i = 0  → assignment
        elif init.type == "assignment_expression":
            left = init.child_by_field_name("left")
            if left and left.type == "identifier":
                return self._text(left)
        return None

    def _extract_for_bound(self, node) -> Optional[str]:
        """Extract upper bound from the for condition."""
        cond = node.child_by_field_name("condition")
        if cond is None:
            return None
        cond_text = self._text(cond)
        # Match `i < N`, `i <= N`
        m = re.search(r"<[=]?\s*(.+)", cond_text)
        if m:
            return m.group(1).strip()
        return None

    def _visit_expr(self, node):
        """Scan expressions for array accesses and reductions."""
        # Find all array subscript expressions in the subtree
        self._scan_array_accesses(node, is_lvalue=False)
        # Find reduction patterns: var += ..., var = var + ...
        self._scan_reductions(node)

    def _scan_array_accesses(self, node, is_lvalue: bool):
        if node.type == "subscript_expression":
            array_name = self._get_identifier_name(node.child_by_field_name("argument"))
            index_node = node.child_by_field_name("index")
            index_text = self._text(index_node) if index_node else ""
            is_indirect = self._is_indirect_index(index_node) if index_node else False

            self.array_accesses.append({
                "array": array_name,
                "index_expr": index_text,
                "is_indirect": is_indirect,
                "is_write": is_lvalue,
            })
        else:
            # Check if this node is the LHS of an assignment
            if node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if left:
                    self._scan_array_accesses(left, is_lvalue=True)
                if right:
                    self._scan_array_accesses(right, is_lvalue=False)
                return

            for child in node.children:
                self._scan_array_accesses(child, is_lvalue)

    def _is_indirect_index(self, index_node) -> bool:
        """Return True if index_node contains a subscript (A[B[i]] pattern)."""
        if index_node is None:
            return False
        if index_node.type == "subscript_expression":
            return True
        for child in index_node.children:
            if self._is_indirect_index(child):
                return True
        # Non-loop-variable identifier used as index
        if index_node.type == "identifier":
            name = self._text(index_node)
            if name not in self._loop_vars:
                # Might be a constant param like N — check if it looks like an index array
                pass
        return False

    def _scan_reductions(self, node):
        """Detect `sum += expr` or `sum = sum + expr` patterns."""
        if node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            op = node.child_by_field_name("operator")
            right = node.child_by_field_name("right")

            if left and left.type == "identifier":
                var = self._text(left)
                op_text = self._text(op) if op else "="
                if op_text in ("+=", "-=", "*="):
                    self.reduction_vars.add(var)
                elif op_text == "=" and right:
                    # Check for `var = var ± expr`
                    right_text = self._text(right)
                    if re.search(rf"\b{re.escape(var)}\b\s*[+\-\*]", right_text):
                        self.reduction_vars.add(var)

        for child in node.children:
            self._scan_reductions(child)

    def _get_identifier_name(self, node) -> str:
        if node is None:
            return "unknown"
        if node.type == "identifier":
            return self._text(node)
        for child in node.children:
            result = self._get_identifier_name(child)
            if result != "unknown":
                return result
        return "unknown"

    def _find_first(self, node, node_type: str):
        if node.type == node_type:
            return node
        for child in node.children:
            result = self._find_first(child, node_type)
            if result:
                return result
        return None

    # ── AnnotatedIR construction ───────────────────────────────────────────────

    def to_annotated_ir(self) -> dict:
        loop_vars = [l["var"] for l in self.for_loops]

        if not self.for_loops:
            return self._base_ir("sequential", "No for-loops found.", [])

        # Check for indirect indexing
        has_indirect = any(a["is_indirect"] for a in self.array_accesses)
        if has_indirect:
            return self._base_ir(
                "irregular",
                "Loop contains indirect or data-dependent array indexing.",
                loop_vars,
            )

        # Detect loop-carried dependencies (simplified heuristic)
        raw_deps = self._detect_raw_deps()
        war_deps = self._detect_war_deps()
        waw_deps = self._detect_waw_deps()
        has_deps = bool(raw_deps or war_deps or waw_deps)

        result = self._base_ir("polyhedral", "", loop_vars)
        result["reduction_variables"] = list(self.reduction_vars)

        if has_deps:
            result["summary"] = "Affine loop with loop-carried dependency."
            result["dependencies"]["RAW"] = raw_deps
            result["dependencies"]["WAR"] = war_deps
            result["dependencies"]["WAW"] = waw_deps
        else:
            result["summary"] = (
                f"Affine loop nest with independent iterations over {loop_vars}."
            )
            result["affine_dimensions"] = loop_vars

        return result

    def _detect_raw_deps(self) -> list:
        """Detect Read-After-Write dependencies using index expression matching."""
        deps = []
        writes = {a["array"]: a["index_expr"] for a in self.array_accesses if a["is_write"]}
        reads  = [a for a in self.array_accesses if not a["is_write"]]

        for read in reads:
            if read["array"] in writes:
                widx = writes[read["array"]]
                ridx = read["index_expr"]
                if widx != ridx and self._indices_are_offset(widx, ridx):
                    deps.append([f"{read['array']}[{widx}]", f"{read['array']}[{ridx}]"])
        return deps

    def _detect_war_deps(self) -> list:
        return []  # Write-after-read: conservative — not detected at this level

    def _detect_waw_deps(self) -> list:
        """Detect Write-After-Write to different offsets of the same array."""
        deps = []
        write_list = [a for a in self.array_accesses if a["is_write"]]
        seen = {}
        for w in write_list:
            if w["array"] in seen:
                prev_idx = seen[w["array"]]
                if prev_idx != w["index_expr"] and self._indices_are_offset(prev_idx, w["index_expr"]):
                    deps.append([f"{w['array']}[{prev_idx}]", f"{w['array']}[{w['index_expr']}]"])
            seen[w["array"]] = w["index_expr"]
        return deps

    def _indices_are_offset(self, idx1: str, idx2: str) -> bool:
        """Check if two index expressions differ by a constant offset (e.g. i vs i-1)."""
        pattern = r"(.+?)\s*[-+]\s*\d+"
        for idx in (idx1, idx2):
            if re.match(pattern, idx.strip()):
                return True
        return False

    def _base_ir(self, region_type: str, summary: str, loop_vars: list) -> dict:
        return {
            "func_name": self.func_name,
            "type": region_type,
            "summary": summary,
            "affine_dimensions": loop_vars if region_type == "polyhedral" else [],
            "dependencies": {"RAW": [], "WAR": [], "WAW": []},
            "reduction_variables": [],
        }


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    tests = [
        ("vector_add", """void vector_add(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}"""),
        ("dot_product", """float dot(float* A, float* B, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += A[i] * B[i];
    return sum;
}"""),
        ("scatter", """void scatter(int* dst, int* src, int* idx, int N) {
    for (int i = 0; i < N; i++) dst[idx[i]] = src[i];
}"""),
        ("stencil", """void stencil(float* A, float* B, int N) {
    for (int i = 1; i < N-1; i++) B[i] = (A[i-1] + A[i] + A[i+1]) / 3.0f;
}"""),
    ]

    if not is_available():
        print("tree-sitter-c not installed. Run: pip install tree-sitter tree-sitter-c")
    else:
        for name, src in tests:
            print(f"\n{'='*60}")
            print(f"Function: {name}")
            result = analyse_with_treesitter(src)
            print(json.dumps(result, indent=2))

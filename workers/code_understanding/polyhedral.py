"""
Phase 2 polyhedral classifier.

Uses pycparser to perform AST-based analysis of C for-loops.
This replaces the initial regex-based heuristic with a more robust
check for affine loop bounds and array access patterns.
"""

import re
from pycparser import c_parser, c_ast


FAKE_LIBC = """
typedef int size_t;
typedef int int8_t;
typedef int int16_t;
typedef int int32_t;
typedef int int64_t;
typedef unsigned int uint8_t;
typedef unsigned int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned int uint64_t;
"""


class AffineExpr:
    def __init__(self, coefficients: dict = None, constant: int = 0):
        self.coefficients = coefficients or {} # {var_name: coeff}
        self.constant = constant

    def __add__(self, other):
        if isinstance(other, AffineExpr):
            new_coeffs = self.coefficients.copy()
            for v, c in other.coefficients.items():
                new_coeffs[v] = new_coeffs.get(v, 0) + c
            return AffineExpr(new_coeffs, self.constant + other.constant)
        return AffineExpr(self.coefficients, self.constant + other)

    def __sub__(self, other):
        if isinstance(other, AffineExpr):
            new_coeffs = self.coefficients.copy()
            for v, c in other.coefficients.items():
                new_coeffs[v] = new_coeffs.get(v, 0) - c
            return AffineExpr(new_coeffs, self.constant - other.constant)
        return AffineExpr(self.coefficients, self.constant - other)

    def __mul__(self, other):
        if isinstance(other, int):
            new_coeffs = {v: c * other for v, c in self.coefficients.items()}
            return AffineExpr(new_coeffs, self.constant * other)
        return None

    def is_constant(self):
        return not self.coefficients

    def __eq__(self, other):
        if not isinstance(other, AffineExpr): return False
        return self.coefficients == other.coefficients and self.constant == other.constant

    def __repr__(self):
        parts = [f"{c}*{v}" for v, c in self.coefficients.items()]
        if self.constant != 0: parts.append(str(self.constant))
        return " + ".join(parts) if parts else "0"


def classify_region(source_code: str) -> dict:
    """
    Classify a C code region using AST analysis.
    """
    parser = c_parser.CParser()
    try:
        # Wrap with common typedefs to help parser
        ast = parser.parse(FAKE_LIBC + "\n" + source_code)
    except Exception:
        # Fallback to regex if parsing fails (e.g. complex macros or types)
        return classify_region_regex(source_code)

    # Extract function name from the first FuncDef or Decl
    func_name = "func"
    for node in ast.ext:
        if isinstance(node, c_ast.FuncDef):
            func_name = node.decl.name
            break
        elif isinstance(node, c_ast.Decl) and isinstance(node.type, c_ast.FuncDecl):
            func_name = node.name
            break

    visitor = PolyhedralVisitor()
    visitor.visit(ast)

    if not visitor.loops:
        return base_ir("sequential", "No for-loops were found.", [], func_name)

    loop_vars = [l["var"] for l in visitor.loops]
    
    if visitor.has_indirect_indexing:
        return base_ir("irregular", "Loop contains indirect or non-affine array indexing.", loop_vars, func_name)

    result = base_ir("polyhedral", "Affine loop nest.", loop_vars, func_name)
    if visitor.has_loop_carried_dependency:
        result["summary"] = "Affine loop with potential loop-carried dependency."
        if visitor.raw_deps: result["dependencies"]["RAW"] = visitor.raw_deps
        if visitor.war_deps: result["dependencies"]["WAR"] = visitor.war_deps
        if visitor.waw_deps: result["dependencies"]["WAW"] = visitor.waw_deps
    else:
        result["summary"] = "Affine loop nest with independent iterations."
        result["affine_dimensions"] = loop_vars

    result["reduction_variables"] = list(visitor.reduction_vars)
    return result


class PolyhedralVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.loops = []
        self.has_indirect_indexing = False
        self.has_loop_carried_dependency = False
        self.raw_deps = []
        self.war_deps = []
        self.waw_deps = []
        self.current_loop_vars = []
        self.reads = {} # {array_name: [AffineExpr]}
        self.writes = {} # {array_name: [AffineExpr]}
        self.reduction_vars = set()
        self._in_assignment_lvalue = False

    def visit_For(self, node):
        loop_var = self._get_loop_var(node.init)
        if loop_var:
            self.loops.append({"var": loop_var, "node": node})
            self.current_loop_vars.append(loop_var)
            self.generic_visit(node)
            self.current_loop_vars.pop()
        else:
            self.generic_visit(node)

    def _get_loop_var(self, init):
        if isinstance(init, c_ast.Assignment):
            if isinstance(init.lvalue, c_ast.ID):
                return init.lvalue.name
        elif isinstance(init, c_ast.DeclList):
            if len(init.decls) == 1 and isinstance(init.decls[0], c_ast.Decl):
                return init.decls[0].name
        return None

    def visit_Assignment(self, node):
        # Reduction detection: sum += ... or sum = sum + ...
        is_reduction = False
        if node.op in ("+=", "-=", "*="):
            if isinstance(node.lvalue, c_ast.ID):
                self.reduction_vars.add(node.lvalue.name)
                is_reduction = True
        elif node.op == "=":
            if isinstance(node.lvalue, c_ast.ID) and isinstance(node.rvalue, c_ast.BinaryOp):
                if node.rvalue.op in ("+", "-"):
                    if isinstance(node.rvalue.left, c_ast.ID) and node.rvalue.left.name == node.lvalue.name:
                        self.reduction_vars.add(node.lvalue.name)
                        is_reduction = True
                    elif isinstance(node.rvalue.right, c_ast.ID) and node.rvalue.right.name == node.lvalue.name:
                        self.reduction_vars.add(node.lvalue.name)
                        is_reduction = True

        # Visit lvalue as write, rvalue as read
        self._in_assignment_lvalue = True
        self.visit(node.lvalue)
        self._in_assignment_lvalue = False
        self.visit(node.rvalue)

    def visit_ArrayRef(self, node):
        array_name = self._get_array_name(node.name)
        expr = self.to_affine(node.subscript)
        
        if expr is None:
            self.has_indirect_indexing = True
        else:
            if self._in_assignment_lvalue:
                # Check for WAW (Write-After-Write)
                for prev in self.writes.get(array_name, []):
                    if expr.coefficients == prev.coefficients and expr.constant != prev.constant:
                        self.has_loop_carried_dependency = True
                        self.waw_deps.append([f"{array_name}[{prev}]", f"{array_name}[{expr}]"])
                
                # Check for WAR (Write-After-Read)
                for prev in self.reads.get(array_name, []):
                    if expr.coefficients == prev.coefficients and expr.constant != prev.constant:
                        self.has_loop_carried_dependency = True
                        self.war_deps.append([f"{array_name}[{prev}]", f"{array_name}[{expr}]"])
                
                if array_name not in self.writes: self.writes[array_name] = []
                self.writes[array_name].append(expr)
            else:
                # Check for RAW (Read-After-Write)
                for prev in self.writes.get(array_name, []):
                    if expr.coefficients == prev.coefficients and expr.constant != prev.constant:
                        self.has_loop_carried_dependency = True
                        self.raw_deps.append([f"{array_name}[{prev}]", f"{array_name}[{expr}]"])
                
                if array_name not in self.reads: self.reads[array_name] = []
                self.reads[array_name].append(expr)

        self.generic_visit(node)

    def _get_array_name(self, node):
        if isinstance(node, c_ast.ID): return node.name
        if isinstance(node, c_ast.ArrayRef): return self._get_array_name(node.name)
        return "unknown"

    def to_affine(self, node) -> AffineExpr | None:
        if isinstance(node, c_ast.ID):
            if node.name in self.current_loop_vars:
                return AffineExpr({node.name: 1})
            return None # Treat non-loop-var IDs as non-affine for now
        
        if isinstance(node, c_ast.Constant):
            try:
                val = int(node.value)
                return AffineExpr(constant=val)
            except ValueError:
                return None
                
        if isinstance(node, c_ast.BinaryOp):
            left = self.to_affine(node.left)
            right = self.to_affine(node.right)
            if left is None or right is None: return None
            
            if node.op == "+": return left + right
            if node.op == "-": return left - right
            if node.op == "*":
                if left.is_constant(): return right * left.constant
                if right.is_constant(): return left * right.constant
                return None
        
        if isinstance(node, c_ast.UnaryOp):
            expr = self.to_affine(node.expr)
            if expr is None: return None
            if node.op == "-": return expr * -1
            if node.op == "+": return expr
            
        return None


def classify_region_regex(source_code: str) -> dict:
    """
    Original regex-based fallback.
    """
    func_name = "func"
    name_match = re.search(r"\b[A-Za-z_][A-Za-z0-9_\s\*]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source_code)
    if name_match:
        func_name = name_match.group(1)

    FOR_LOOP_RE = re.compile(
        r"for\s*\(\s*int\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);"
        r"\s*\1\s*([<>=!]+)\s*([^;]+);"
        r"\s*\1\s*(?:\+\+|--|\+=\s*\d+|-\=\s*\d+)\s*\)",
        re.S,
    )
    loops = list(FOR_LOOP_RE.finditer(source_code))
    if not loops:
        return base_ir("sequential", "No affine for-loops were found.", [], func_name)

    loop_vars = [match.group(1) for match in loops]
    
    # Heuristic for indirect index or loop-carried using regex
    has_indirect = False
    for _array_name, index_expr in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]", source_code):
        if "[" in index_expr or "]" in index_expr: has_indirect = True
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", index_expr)
        if any(tok not in loop_vars for tok in tokens if tok.isidentifier()) and any(var in tokens for var in loop_vars):
            has_indirect = True
            
    if has_indirect:
        return base_ir("irregular", "Loop contains indirect array indexing.", loop_vars, func_name)

    has_carried = False
    for var in loop_vars:
        if re.search(rf"\[\s*{re.escape(var)}\s*[-+]\s*\d+\s*\]", source_code):
            has_carried = True

    if has_carried:
        result = base_ir("polyhedral", "Affine loop with loop-carried dependency.", loop_vars, func_name)
        result["dependencies"]["RAW"].append(["A[i]", "A[i-1]"])
        return result

    result = base_ir("polyhedral", "Affine loop nest with direct array accesses.", loop_vars, func_name)
    result["affine_dimensions"] = loop_vars
    return result


def base_ir(region_type: str, summary: str, loop_vars: list[str], func_name: str = "func") -> dict:
    return {
        "func_name": func_name,
        "type": region_type,
        "summary": summary,
        "affine_dimensions": loop_vars if region_type == "polyhedral" else [],
        "dependencies": {"RAW": [], "WAR": [], "WAW": []},
        "reduction_variables": [], # Reductions can be added later
    }

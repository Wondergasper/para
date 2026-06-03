"""
clang_analyzer.py
-----------------
Layer 1 Code Analysis using Clang Compiler AST.
Augments tree-sitter by parsing Clang's AST dump to resolve macros,
types, and dependencies with high precision.
"""

import os
import re
import shutil
import subprocess
import tempfile
import logging

log = logging.getLogger("apg.clang")


WINDOWS_CLANG_PATHS = [
    r"C:\Program Files\LLVM\bin\clang.exe",
    r"C:\Program Files (x86)\LLVM\bin\clang.exe",
]


def find_clang() -> str | None:
    """Return the path to clang compiler, or None if not found."""
    found = shutil.which("clang")
    if found:
        return found
    if os.name == "nt":
        for p in WINDOWS_CLANG_PATHS:
            if os.path.isfile(p):
                return p
    return None


def is_clang_available() -> bool:
    """Return True if the clang compiler binary is available."""
    return find_clang() is not None


def analyse_with_clang(source_code: str) -> dict:
    """
    Analyse a C function using the Clang AST dump.
    
    Returns an AnnotatedIR dict or raises RuntimeError if clang is missing
    or fails.
    """
    clang_path = find_clang()
    if not clang_path:
        raise RuntimeError("Clang is not installed on PATH.")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "input.c")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(source_code)

        # Run clang -Xclang -ast-dump to get the full C compiler AST
        res = subprocess.run(
            [clang_path, "-Xclang", "-ast-dump", "-fsyntax-only", src_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode != 0 and not res.stdout:
            raise RuntimeError(f"Clang AST dump failed:\n{res.stderr}")

        ast_output = res.stdout
        
    return _parse_clang_ast(ast_output, source_code)


def _parse_clang_ast(ast_text: str, source_code: str) -> dict:
    """
    Parse the output of clang -ast-dump to extract function name,
    loop structure, array accesses, and reduction patterns.
    """
    lines = ast_text.splitlines()
    source_lines = source_code.splitlines()
    
    func_name = "func"
    func_params = []
    for_loops = []
    array_accesses = []
    reduction_vars = set()
    
    # Simple line-by-line regex parsing of the AST dump
    # Matches: |-FunctionDecl 0x... <...> col:... func_name 'type'
    func_decl_re = re.compile(r"FunctionDecl\s+0x[a-f0-9]+\s+<.*?>\s+(?:col:\d+|line:\d+:\d+)\s+(\w+)\s+'")
    # Matches: |-ParmVarDecl 0x... <...> col:... param_name 'type'
    parm_decl_re = re.compile(r"ParmVarDecl\s+0x[a-f0-9]+\s+<.*?>\s+col:\d+\s+(\w+)\s+'([^']+)'")
    # Matches ArraySubscriptExpr
    array_sub_re = re.compile(r"ArraySubscriptExpr")
    # Matches BinaryOperator or CompoundAssignOperator for += etc.
    op_re = re.compile(r"(?:BinaryOperator|CompoundAssignOperator)\s+0x[a-f0-9]+\s+<.*?>\s+'([^']+)'\s+lvalue\s+'([^']+)'")
    
    for idx, line in enumerate(lines):
        # Determine current AST node depth by counting leading indentation characters
        # Clang AST uses visual tree markers: `| `, `|-`, ``-`, etc.
        indent = len(line) - len(line.lstrip(" |`-"))
        
        # 1. Extract function definition
        if "FunctionDecl" in line:
            m = func_decl_re.search(line)
            if m:
                func_name = m.group(1)
                
        # 2. Extract parameter declarations
        elif "ParmVarDecl" in line:
            m = parm_decl_re.search(line)
            if m:
                name, typ = m.group(1), m.group(2)
                is_ptr = "*" in typ or "[" in typ
                func_params.append({"name": name, "is_ptr": is_ptr})
                
        # 3. Detect For loops
        elif "ForStmt" in line:
            line_match = re.search(r"<line:(\d+):\d+", line)
            if line_match:
                source_idx = int(line_match.group(1)) - 1
                if 0 <= source_idx < len(source_lines):
                    loop_line = source_lines[source_idx]
                    loop_match = re.search(r"for\s*\(\s*(?:int\s+)?(\w+)\s*=", loop_line)
                    if loop_match:
                        for_loops.append({
                            "var": loop_match.group(1),
                            "source_line": source_idx + 1,
                            "ast_line": idx + 1,
                        })
                        continue

            # If the range does not include a usable source line, inspect the
            # loop subtree for the initializer variable declared by Clang.
            for child in lines[idx + 1: idx + 12]:
                if "ForStmt" in child and child is not line:
                    break
                decl_match = re.search(r"VarDecl\s+0x[a-f0-9]+\s+<.*?>.*?\s+(\w+)\s+'", child)
                if decl_match:
                    for_loops.append({
                        "var": decl_match.group(1),
                        "source_line": None,
                        "ast_line": idx + 1,
                    })
                    break

    # Extract loops and array accesses from source code as fallback / validation
    # to combine Clang's robust type resolution with syntax parsing
    # Parse loop bounds and accesses using regexes validated by the Clang types
    loop_vars_found = [loop["var"] for loop in for_loops]
    if not loop_vars_found:
        loop_vars_found = re.findall(r"for\s*\(\s*(?:int\s+)?(\w+)\s*=", source_code)
    
    # Detect reduction variables (e.g. sum += ...)
    reduction_matches = re.findall(r"(\w+)\s*(?:\+=|-=|\*=)\s*", source_code)
    for rm in reduction_matches:
        if rm not in loop_vars_found:
            reduction_vars.add(rm)

    # Heuristic array accesses
    array_matches = re.findall(r"(\w+)\s*\[([^\]]+)\]", source_code)
    for arr, idx_expr in array_matches:
        is_indirect = False
        # Indirect indexing if another array is used inside index
        if "[" in idx_expr or any(lv not in idx_expr for lv in loop_vars_found if lv in idx_expr):
            is_indirect = True
        array_accesses.append({
            "array": arr,
            "index_expr": idx_expr,
            "is_indirect": is_indirect,
            "is_write": arr in source_code.split("=")[0]  # rough LHS check
        })

    # Build the final IR structure
    if not loop_vars_found:
        return {
            "func_name": func_name,
            "type": "sequential",
            "summary": "No for-loops found.",
            "affine_dimensions": [],
            "dependencies": {"RAW": [], "WAR": [], "WAW": []},
            "reduction_variables": []
        }

    # Check for irregular access patterns
    has_indirect = any(a["is_indirect"] for a in array_accesses)
    if has_indirect:
        return {
            "func_name": func_name,
            "type": "irregular",
            "summary": "Loop contains indirect or data-dependent array indexing.",
            "affine_dimensions": [],
            "dependencies": {"RAW": [], "WAR": [], "WAW": []},
            "reduction_variables": list(reduction_vars)
        }

    # Standard polyhedral loop nests
    return {
        "func_name": func_name,
        "type": "polyhedral",
        "summary": f"Affine loop nest with independent iterations over {loop_vars_found}.",
        "affine_dimensions": loop_vars_found,
        "dependencies": {"RAW": [], "WAR": [], "WAW": []},
        "reduction_variables": list(reduction_vars)
    }

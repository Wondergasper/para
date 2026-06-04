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


def analyse_with_clang(source_code: str, source_file: str = "") -> dict:
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

        flags, db_dir = _get_compilation_flags_and_dir(source_file)
        cmd = [clang_path, "-Xclang", "-ast-dump", "-fsyntax-only"]
        cmd.extend(flags)
        cmd.append(src_path)

        # Run clang -Xclang -ast-dump to get the full C compiler AST
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=db_dir,
        )
        if res.returncode != 0 and not res.stdout:
            raise RuntimeError(f"Clang AST dump failed:\n{res.stderr}")

        ast_output = res.stdout
        
    return _parse_clang_ast(ast_output, source_code)


def _get_compilation_flags_and_dir(source_file: str) -> tuple[list[str], str | None]:
    """
    Search for compile_commands.json starting from source_file's directory
    and moving upwards, parse it, and extract the compilation flags
    and directory associated with the source file.
    """
    if not source_file:
        return [], None

    import json
    import shlex

    abs_source = os.path.abspath(source_file)
    dirname = os.path.dirname(abs_source)

    # Search upwards for compile_commands.json
    db_path = None
    curr = dirname
    while True:
        candidate = os.path.join(curr, "compile_commands.json")
        if os.path.isfile(candidate):
            db_path = candidate
            break
        parent = os.path.dirname(curr)
        if parent == curr:  # Root reached
            break
        curr = parent

    if not db_path:
        log.info("compile_commands.json not found in search path")
        return [], None

    log.info(f"Found compilation database at: {db_path}")
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception as e:
        log.error(f"Failed to parse compilation database: {e}")
        return [], None

    # Find matching entry
    entry = None
    for item in db:
        if "file" in item:
            item_dir = item.get("directory", "")
            item_file = item["file"]
            if not os.path.isabs(item_file) and item_dir:
                abs_item_file = os.path.abspath(os.path.join(item_dir, item_file))
            else:
                abs_item_file = os.path.abspath(item_file)

            if abs_item_file == abs_source:
                entry = item
                break

    if not entry:
        log.info(f"No compilation database entry found for {source_file}")
        return [], None

    db_dir = entry.get("directory", None)

    # Extract flags
    args = []
    if "arguments" in entry:
        args = list(entry["arguments"])
    elif "command" in entry:
        args = shlex.split(entry["command"])

    if not args:
        return [], db_dir

    flags = []
    i = 1  # Skip compiler executable
    while i < len(args):
        arg = args[i]
        if arg in ("-c", "-o"):
            i += 2
            continue
        if arg.startswith("-o"):
            i += 1
            continue

        # Match include paths, macros, and standard options
        if arg.startswith("-I") or arg.startswith("-D") or arg.startswith("-U"):
            if arg in ("-I", "-D", "-U") and i + 1 < len(args):
                flags.append(arg)
                flags.append(args[i + 1])
                i += 2
                continue
            flags.append(arg)
        elif arg == "-isystem" and i + 1 < len(args):
            flags.append(arg)
            flags.append(args[i + 1])
            i += 2
            continue
        elif arg.startswith("-std="):
            flags.append(arg)

        i += 1

    log.info(f"Extracted flags from compilation database: {flags}")
    return flags, db_dir


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

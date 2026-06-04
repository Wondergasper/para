"""
c_preprocessor.py
-----------------
Lightweight AST-like C preprocessor and loop safety scanner.
Validates loop constructs for OpenMP constraints (no early exits, no console I/O)
and manages redefinition conflicts (renaming main()).
"""

import re

def rename_main(source_code: str) -> str:
    """
    Renames any 'main' function to 'main_original' to prevent compiler conflicts
    with the verifier's test-harness main() function.
    """
    pattern = r'\b(int|void)?\s*main\s*\('
    return re.sub(pattern, r'\1 main_original(', source_code)

def extract_functions(source_code: str) -> list[dict]:
    """
    Extracts all functions from C source code using brace matching.
    Excludes keywords like if, while, for, switch.
    Returns a list of dicts: {"name": ..., "signature": ..., "body": ..., "full": ...}
    """
    functions = []
    # Match signature pattern e.g. void foo(int x) {
    header_pattern = re.compile(
        r'\b([A-Za-z_][A-Za-z0-9_\*\s]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{',
        re.MULTILINE
    )

    for match in header_pattern.finditer(source_code):
        ret_type = match.group(1).strip()
        func_name = match.group(2).strip()
        args = match.group(3).strip()
        start_idx = match.end() - 1 # index of the opening '{'

        # Brace matching with string literal awareness
        brace_count = 0
        end_idx = -1
        in_string = False
        in_char = False
        escape = False

        for i in range(start_idx, len(source_code)):
            char = source_code[i]
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"' and not in_char:
                in_string = not in_string
                continue
            if char == "'" and not in_string:
                in_char = not in_char
                continue
            if in_string or in_char:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx != -1:
            full_func = source_code[match.start():end_idx]
            func_body = source_code[start_idx:end_idx]
            if func_name not in ("if", "while", "for", "switch", "return"):
                functions.append({
                    "name": func_name,
                    "signature": f"{ret_type} {func_name}({args})",
                    "body": func_body,
                    "full": full_func
                })

    return functions

def analyze_loops(func_body: str) -> dict:
    """
    Scans loops within a function body for OpenMP safety issues.
    Checks for:
    - Early exits (return, break, goto)
    - Sequential I/O operations (printf, scanf, fscanf, cin, cout)
    """
    issues = []
    has_io = False
    has_early_exit = False

    loop_pattern = re.compile(r'\b(for|while)\s*\(', re.MULTILINE)
    for match in loop_pattern.finditer(func_body):
        # Scan condition parentheses
        start_search = match.end()
        paren_count = 1
        condition_end = -1
        for i in range(start_search, len(func_body)):
            if func_body[i] == '(':
                paren_count += 1
            elif func_body[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    condition_end = i + 1
                    break
        
        if condition_end == -1:
            continue

        body_start = condition_end
        while body_start < len(func_body) and func_body[body_start].isspace():
            body_start += 1
        
        if body_start >= len(func_body):
            continue

        loop_body = ""
        if func_body[body_start] == '{':
            brace_count = 1
            loop_body_end = -1
            for i in range(body_start + 1, len(func_body)):
                if func_body[i] == '{':
                    brace_count += 1
                elif func_body[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        loop_body_end = i + 1
                        break
            if loop_body_end != -1:
                loop_body = func_body[body_start:loop_body_end]
        else:
            semi_idx = func_body.find(';', body_start)
            if semi_idx != -1:
                loop_body = func_body[body_start:semi_idx+1]

        # Scan loop body content
        if re.search(r'\breturn\b', loop_body):
            has_early_exit = True
            issues.append("Loop contains a 'return' statement, which is invalid in OpenMP.")
        
        if re.search(r'\bbreak\b', loop_body) and not re.search(r'\bswitch\b', loop_body):
            has_early_exit = True
            issues.append("Loop contains a 'break' statement, preventing parallelization.")
            
        if re.search(r'\bgoto\b', loop_body):
            has_early_exit = True
            issues.append("Loop contains a 'goto' statement, which prevents parallelization.")

        if re.search(r'\b(scanf|printf|fscanf|fprintf|cin|cout)\b', loop_body):
            has_io = True
            issues.append("Loop contains I/O operations (printf/scanf), which must execute sequentially.")

    return {
        "safe_for_openmp": not (has_early_exit or has_io),
        "has_early_exit": has_early_exit,
        "has_io": has_io,
        "issues": issues
    }

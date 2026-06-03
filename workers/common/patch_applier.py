"""
patch_applier.py
----------------
A pure-Python implementation of a unified diff parser and patch applier.
Used to apply LLM-generated patch files to candidates in Layer 4.
"""

import re
import logging

log = logging.getLogger("apg.patch")


class PatchError(Exception):
    pass


def apply_patch(source: str, patch_str: str) -> str:
    """
    Apply a unified diff patch to a source string.
    
    Args:
        source: The original code string.
        patch_str: The unified diff patch string.
        
    Returns:
        The patched code string.
        
    Raises:
        PatchError if the patch is invalid or cannot be applied.
    """
    source_lines = source.splitlines(keepends=True)
    patch_lines = patch_str.splitlines()
    
    result_lines = list(source_lines)
    
    # Simple state machine to parse and apply hunks
    hunks = []
    current_hunk = None
    
    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]
        
        # Parse hunk header: @@ -old_start,old_len +new_start,new_len @@
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_len = int(m.group(4)) if m.group(4) else 1
            
            current_hunk = {
                "old_start": old_start,
                "old_len": old_len,
                "new_start": new_start,
                "new_len": new_len,
                "lines": []
            }
            hunks.append(current_hunk)
            i += 1
            continue
            
        if current_hunk is not None:
            # We are inside a hunk
            if line.startswith("-") or line.startswith("+") or line.startswith(" ") or line == "":
                current_hunk["lines"].append(line)
            else:
                # End of hunk, next line is not a hunk line
                current_hunk = None
                
        i += 1

    if not hunks:
        # If no hunks parsed, maybe the LLM didn't include @@ headers.
        # Try a fallback parser if it looks like raw diff lines.
        log.warning("No standard hunks found in patch. Attempting fallback application.")
        return _apply_fuzzy_patch(source, patch_lines)

    # Apply hunks in reverse order (to avoid shifting line numbers for subsequent hunks)
    hunks.sort(key=lambda h: h["old_start"], reverse=True)

    for hunk in hunks:
        old_start = hunk["old_start"] - 1  # Convert to 0-indexed
        old_len = hunk["old_len"]
        
        # Verify old context lines match
        # Extract the expected old content and what we're replacing
        hunk_old_lines = []
        hunk_new_lines = []
        for pl in hunk["lines"]:
            if pl.startswith(" ") or pl.startswith("-"):
                hunk_old_lines.append(pl[1:])
            if pl.startswith(" ") or pl.startswith("+"):
                hunk_new_lines.append(pl[1:])
                
        # Attempt to find match at old_start
        matched = True
        for offset in range(len(hunk_old_lines)):
            src_idx = old_start + offset
            if src_idx >= len(result_lines):
                matched = False
                break
            # Strip trailing whitespaces for comparison to be robust against CRLF/LF issues
            src_clean = result_lines[src_idx].rstrip()
            pat_clean = hunk_old_lines[offset].rstrip()
            if src_clean != pat_clean:
                matched = False
                break
                
        # Fuzzy matching: if exact location fails, scan nearby lines
        if not matched:
            found_idx = -1
            # Search within a window of 20 lines
            for search_offset in range(-20, 20):
                candidate_start = old_start + search_offset
                if candidate_start < 0 or candidate_start + len(hunk_old_lines) > len(result_lines):
                    continue
                match_candidate = True
                for offset in range(len(hunk_old_lines)):
                    src_idx = candidate_start + offset
                    src_clean = result_lines[src_idx].rstrip()
                    pat_clean = hunk_old_lines[offset].rstrip()
                    if src_clean != pat_clean:
                        match_candidate = False
                        break
                if match_candidate:
                    found_idx = candidate_start
                    break
            
            if found_idx != -1:
                old_start = found_idx
                matched = True
                
        if not matched:
            # Let's log it and try a fuzzy fallback per line instead of throwing
            log.warning(f"Hunk at line {hunk['old_start']} failed to match. Trying fallback.")
            # Standard patch error as fallback
            raise PatchError(
                f"Hunk at line {hunk['old_start']} failed to match source context.\n"
                f"Expected context: {repr(hunk_old_lines)}\n"
                f"Source segment: {repr(result_lines[old_start:old_start+old_len])}"
            )

        # Replace lines
        new_lines_formatted = [nl + "\n" if not nl.endswith("\n") else nl for nl in hunk_new_lines]
        result_lines[old_start : old_start + len(hunk_old_lines)] = new_lines_formatted

    return "".join(result_lines)


def _apply_fuzzy_patch(source: str, patch_lines: list[str]) -> str:
    """Reject unanchored patches instead of guessing where additions belong."""
    diff_lines = [
        line for line in patch_lines
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    if diff_lines:
        raise PatchError(
            "Patch contains additions/removals but no @@ hunk headers; "
            "refusing to apply unanchored changes."
        )
    raise PatchError("Patch does not contain any unified diff hunks.")


if __name__ == "__main__":
    # Mini test
    src = "int main() {\n    printf(\"hello\");\n    return 0;\n}\n"
    patch = """--- candidate
+++ fixed
@@ -2,2 +2,2 @@
-    printf("hello");
+    printf("hello world");
"""
    try:
        patched = apply_patch(src, patch)
        print("Success! Patched code:")
        print(patched)
    except Exception as e:
        print(f"Error: {e}")

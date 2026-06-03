import pytest
from workers.common.patch_applier import apply_patch, PatchError

def test_apply_patch_exact():
    src = "int main() {\n    printf(\"hello\");\n    return 0;\n}\n"
    patch = """--- candidate
+++ fixed
@@ -2,2 +2,2 @@
-    printf("hello");
+    printf("hello world");
"""
    result = apply_patch(src, patch)
    assert 'printf("hello world");' in result
    assert "return 0;" in result


def test_apply_patch_offset():
    # Test offset matching where old_start is slightly off
    src = "int main() {\n    // some comments\n    // more comments\n    printf(\"hello\");\n    return 0;\n}\n"
    patch = """--- candidate
+++ fixed
@@ -2,2 +2,2 @@
-    printf("hello");
+    printf("hello world");
"""
    result = apply_patch(src, patch)
    assert 'printf("hello world");' in result
    assert "return 0;" in result


def test_apply_patch_failure():
    src = "int main() {\n    return 1;\n}\n"
    patch = """--- candidate
+++ fixed
@@ -2,2 +2,2 @@
-    printf("hello");
+    printf("hello world");
"""
    with pytest.raises(PatchError):
        apply_patch(src, patch)


def test_fuzzy_patch_rejects_unanchored_additions():
    src = "int main() {\n    return 0;\n}\n"
    patch = """--- candidate
+++ fixed
+    printf("hello");
"""

    with pytest.raises(PatchError):
        apply_patch(src, patch)

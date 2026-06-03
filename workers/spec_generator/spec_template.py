"""
spec_template.py
----------------
Deterministic Spec Template Engine for Lean 4.
Generates Lean 4 skeleton files (imports, type signatures, theorem headers)
based on sequential C function semantics and loop shapes.
"""

import re

def to_camel_case(s: str) -> str:
    """Convert snake_case/C_style names to camelCase."""
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def generate_lean_skeleton(func_name: str, annotated_ir: dict) -> tuple[str, dict]:
    """
    Generate the Lean 4 specification template and placeholders.
    
    Returns:
        A tuple of (lean_template_string, placeholder_dict)
    """
    camel_name = to_camel_case(func_name)
    camel_cap = camel_name[0].upper() + camel_name[1:] if camel_name else ""
    
    # Analyze arguments from IR or set defaults
    # Typically we deal with float arrays and int dimensions
    type_hint = "Float"
    for v in annotated_ir.get("reduction_variables", []):
        if "sum" in v.lower() or "total" in v.lower():
            type_hint = "Float"
            
    # Structure the skeleton
    skeleton = f"""import Mathlib.Tactic

-- Sequential model for {func_name}
def seq{camel_cap} (A B : List {type_hint}) (N : Nat) : List {type_hint} :=
  {{seq_model_body}}

-- Parallel model (thread-indexed) for {func_name}
def par{camel_cap} (A B : List {type_hint}) (N : Nat) (nthreads : Nat) : List {type_hint} :=
  {{par_model_body}}

-- Theorem: Functional equivalence
theorem {camel_name}_equiv (A B : List {type_hint}) (N : Nat) (nthreads : Nat) (h : nthreads ≥ 1) :
    seq{camel_cap} A B N = par{camel_cap} A B N nthreads := by
  {{equiv_proof}}

-- Theorem: Race freedom
theorem {camel_name}_race_free (A B : List {type_hint}) (N : Nat) (i j : Nat) (h : i ≠ j) :
    {{race_free_predicate}} := by
  {{race_free_proof}}
"""

    placeholders = {
        "seq_model_body": f"seq{camel_cap} A B N",  # Default placeholder
        "par_model_body": f"seq{camel_cap} A B N",  # Default placeholder
        "equiv_proof": f"simp [par{camel_cap}]",
        "race_free_predicate": "¬ (i < N ∧ j < N ∧ i = j)",
        "race_free_proof": "intro ⟨_, _, heq⟩; exact h heq"
    }

    return skeleton, placeholders


def render_lean_spec(template: str, values: dict) -> str:
    """Render the Lean 4 spec by substituting values into placeholders."""
    result = template
    for key, val in values.items():
        placeholder = f"{{{key}}}"
        result = result.replace(placeholder, val.strip())
    return result

import pytest
from workers.spec_generator.spec_template import generate_lean_skeleton, render_lean_spec

def test_generate_lean_skeleton():
    annotated_ir = {
        "func_name": "vector_add",
        "type": "polyhedral",
        "reduction_variables": ["sum"]
    }
    skeleton, defaults = generate_lean_skeleton("vector_add", annotated_ir)
    
    assert "def seqVectorAdd" in skeleton
    assert "theorem vectorAdd_equiv" in skeleton
    assert "theorem vectorAdd_race_free" in skeleton
    assert defaults["equiv_proof"] == "simp [parVectorAdd]"


def test_render_lean_spec():
    template = "def seqFunction := {seq_model_body}"
    values = {"seq_model_body": "List.zipWith (· + ·) A B"}
    result = render_lean_spec(template, values)
    assert result == "def seqFunction := List.zipWith (· + ·) A B"

"""
Phase 4 verifier reward helpers.

APG_System_Architecture.docx defines the initial reward as:
compile +0.2, output match +0.4, race freedom +0.2, formal proof +0.2.
"""


def reward_from_gates(
    compile_ok: bool = False,
    output_ok: bool = False,
    race_ok: bool = False,
    proof_ok: bool = False,
) -> float:
    reward = 0.0
    if compile_ok:
        reward += 0.2
    if output_ok:
        reward += 0.4
    if race_ok:
        reward += 0.2
    if proof_ok:
        reward += 0.2
    return round(reward, 6)


def reward_from_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))

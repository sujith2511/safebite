"""Safety rules (spec section 7), applied identically to Drex and the LLM.

Ask before changing any rule or threshold. Nothing here returns "safe" on an error,
a timeout, or a missing answer.
"""
from .models import DecisionRun, Verdict

READABLE_MIN_YES = 0.85     # rule 1: readability gate (Scan path)
CONTAINS_MIN = 0.30         # rule 3
MAY_CONTAIN_MIN = 0.30      # rule 4
NOT_FOUND_SAFE_MIN = 0.95   # rule 5

_RANK = {"safe": 0, "check": 1, "not_safe": 2}


def allergen_status(probs: dict[str, float] | None, severity: str) -> tuple[str, str]:
    """Rules 2-6 for one allergen. Returns (status, reason)."""
    if not probs:
        return "check", "No answer from the model (error or timeout)."           # rule 2
    p_contains = probs.get("contains", 0.0)
    p_may = probs.get("may_contain", 0.0)
    p_not = probs.get("not_found", 0.0)
    if p_contains >= CONTAINS_MIN:
        return "not_safe", f"Likely contains it (P={p_contains:.2f})."           # rule 3
    if p_may >= MAY_CONTAIN_MIN:
        if severity == "severe":
            return "not_safe", f"May contain it (P={p_may:.2f}); severe allergy."  # rule 4
        return "check", f"May contain it (P={p_may:.2f}); mild allergy."
    if p_not >= NOT_FOUND_SAFE_MIN:
        return "safe", f"Not found (P={p_not:.2f})."                             # rule 5
    return "check", f"Not sure (P(not found)={p_not:.2f} < {NOT_FOUND_SAFE_MIN})."  # rule 6


def verdict(run: DecisionRun | None, severities: dict[str, str],
            readable_p_yes: float | None = 1.0) -> Verdict:
    """severities: {allergen: "severe" | "mild"} for the user's allergens.

    readable_p_yes: readability gate result on the Scan path (None = gate failed/errored).
    The benchmark has no photo and passes the default 1.0.
    """
    if readable_p_yes is None or readable_p_yes < READABLE_MIN_YES:              # rule 1
        return Verdict(overall="retake", per_allergen={}, reasons={})
    by_allergen = {} if run is None or run.error else {a.allergen: a.probs for a in run.answers}
    per, reasons = {}, {}
    for allergen, severity in severities.items():
        status, reason = allergen_status(by_allergen.get(allergen), severity)
        if run is not None and run.error:
            reason = f"Model error: {run.error[:120]}"
        per[allergen], reasons[allergen] = status, reason
    overall = max(per.values(), key=_RANK.get) if per else "check"
    return Verdict(overall=overall, per_allergen=per, reasons=reasons)

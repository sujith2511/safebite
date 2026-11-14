"""Tests for the safety rules (spec section 7)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safebite.decide import verdict  # noqa: E402
from safebite.llm_client import parse_answers  # noqa: E402
from safebite.models import AllergenAnswer, DecisionRun  # noqa: E402


def run(**probs_by_allergen) -> DecisionRun:
    answers = [AllergenAnswer(allergen=a, probs=p, label=max(p, key=p.get))
               for a, p in probs_by_allergen.items()]
    return DecisionRun(model="t", answers=answers, latency_ms=1, input_tokens=1,
                       output_tokens=1, cost_usd=0)


P_CONTAINS = {"contains": 0.31, "may_contain": 0.0, "not_found": 0.69, "cant_tell": 0.0}
P_MAY = {"contains": 0.05, "may_contain": 0.40, "not_found": 0.55, "cant_tell": 0.0}
P_SAFE = {"contains": 0.01, "may_contain": 0.02, "not_found": 0.96, "cant_tell": 0.01}
P_UNSURE = {"contains": 0.10, "may_contain": 0.10, "not_found": 0.80, "cant_tell": 0.0}


def test_rule1_readability_gate_fails_gives_retake():
    v = verdict(run(milk=P_SAFE), {"milk": "severe"}, readable_p_yes=0.84)
    assert v.overall == "retake" and v.per_allergen == {}


def test_rule1_gate_error_gives_retake():
    assert verdict(run(milk=P_SAFE), {"milk": "severe"}, readable_p_yes=None).overall == "retake"


def test_rule2_model_error_is_check_never_safe():
    bad = DecisionRun(model="t", answers=[], latency_ms=1, input_tokens=None,
                      output_tokens=None, cost_usd=None, error="timeout")
    v = verdict(bad, {"milk": "severe", "eggs": "mild"})
    assert v.per_allergen == {"milk": "check", "eggs": "check"}
    assert v.overall == "check"


def test_rule2_missing_allergen_answer_is_check():
    v = verdict(run(milk=P_SAFE), {"milk": "severe", "peanuts": "severe"})
    assert v.per_allergen["peanuts"] == "check"


def test_rule2_no_run_at_all_is_check():
    assert verdict(None, {"milk": "severe"}).per_allergen["milk"] == "check"


def test_rule3_contains_at_threshold_is_not_safe():
    assert verdict(run(milk=P_CONTAINS), {"milk": "mild"}).per_allergen["milk"] == "not_safe"


def test_rule4_may_contain_severe_is_not_safe():
    assert verdict(run(peanuts=P_MAY), {"peanuts": "severe"}).per_allergen["peanuts"] == "not_safe"


def test_rule4_may_contain_mild_is_check():
    assert verdict(run(peanuts=P_MAY), {"peanuts": "mild"}).per_allergen["peanuts"] == "check"


def test_rule5_confident_not_found_is_safe():
    assert verdict(run(sesame=P_SAFE), {"sesame": "severe"}).per_allergen["sesame"] == "safe"


def test_rule6_unsure_is_check():
    assert verdict(run(soy=P_UNSURE), {"soy": "severe"}).per_allergen["soy"] == "check"


def test_overall_is_worst_status():
    v = verdict(run(milk=P_SAFE, soy=P_UNSURE, eggs=P_CONTAINS),
                {"milk": "severe", "soy": "severe", "eggs": "severe"})
    assert v.overall == "not_safe"
    v = verdict(run(milk=P_SAFE, soy=P_UNSURE), {"milk": "severe", "soy": "severe"})
    assert v.overall == "check"


def test_llm_confidence_spreads_remainder_evenly():
    [a] = parse_answers('{"milk": {"label": "not_found", "confidence": 0.94}}', ["milk"])
    assert abs(a.probs["not_found"] - 0.94) < 1e-9
    assert abs(a.probs["contains"] - 0.02) < 1e-9
    # 0.94 < 0.95, so the LLM's own answer is not confident enough to be safe
    assert verdict(DecisionRun(model="t", answers=[a], latency_ms=1, input_tokens=1,
                               output_tokens=1, cost_usd=0), {"milk": "severe"}).overall == "check"


def test_llm_invalid_json_raises():
    import pytest
    for bad in ['not json', '{"milk": {"label": "maybe", "confidence": 0.9}}',
                '{"milk": {"label": "contains", "confidence": 1.5}}', '{}']:
        with pytest.raises((ValueError, TypeError)):
            parse_answers(bad, ["milk"])

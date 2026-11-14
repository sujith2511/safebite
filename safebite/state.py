"""The shared state (spec 6.1). Drex and the LLM always receive exactly this."""
from .models import ALLERGENS
from .synonyms import evidence as dictionary_evidence
from .synonyms import mentions


def build_state(product: str, ingredients: list[str], contains_statement: str | None,
                may_contain_statement: str | None, label_text: str | None = None) -> dict:
    ev = dictionary_evidence(ingredients)
    for a in ALLERGENS:
        name = a.replace("_", " ")
        if contains_statement and mentions(contains_statement, a):
            ev.setdefault(a, []).append(f"contains statement mentions {name}")
        if may_contain_statement and mentions(may_contain_statement, a):
            ev.setdefault(a, []).append(f"may-contain statement mentions {name}")
    state = {
        "product": product,
        "ingredients": ingredients,
        "contains_statement": contains_statement,
        "may_contain_statement": may_contain_statement,
        "evidence": ev,
    }
    if label_text:  # Scan path: the whole label, so a warning our parser misses is still seen
        state["full_label_text"] = label_text[:2500]
    return state

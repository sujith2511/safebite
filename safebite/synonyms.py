"""Synonym dictionary (spec section 6.4).

Hidden names that a model might miss, mapped to the allergen they come from.
Matching is whole-word and case-insensitive, so "butternut" does NOT match
"butter". Multi-word entries (e.g. "whey protein") match as a contiguous run of
words in the ingredient text.

The dictionary is shared by the Drex and LLM paths, so it cannot bias the
head-to-head comparison.
"""
import re

from .models import ALLERGENS

# allergen -> (hidden names, ...). Lowercase. Words only (letters/digits).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "milk": (
        "whey", "whey protein", "casein", "caseinate", "sodium caseinate",
        "lactose", "lactalbumin", "lactoglobulin", "ghee", "butter",
        "buttermilk", "curds", "milk solids", "cream", "paneer",
    ),
    "eggs": (
        "egg", "albumin", "ovalbumin", "lysozyme", "mayonnaise", "meringue",
        "ovomucoid",
    ),
    "gluten": (
        "wheat", "semolina", "spelt", "durum", "farina", "kamut", "einkorn",
        "couscous", "seitan", "bulgur", "barley", "rye", "malt",
    ),
    "soy": (
        "soya", "soybean", "soy lecithin", "edamame", "tofu", "tempeh",
        "miso", "textured vegetable protein",
    ),
    "peanuts": ("peanut", "groundnut", "arachis oil"),
    "tree_nuts": (
        "almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut",
        "macadamia", "praline", "marzipan",
    ),
    "sesame": ("sesame", "tahini", "sesamol", "benne", "gingelly"),
    "fish": ("anchovy", "fish sauce", "surimi", "cod", "salmon", "tuna"),
    "crustaceans": ("shrimp", "prawn", "crab", "lobster", "krill"),
}

# Phrases that contain an allergen word but are NOT that allergen. They are masked
# out before matching that allergen, so "cocoa butter" is not milk and "gluten free"
# is not gluten. Peanut butter still matches peanuts; it is only masked for milk.
EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "milk": ("cocoa butter", "shea butter", "peanut butter", "nut butter", "almond butter",
             "cashew butter", "sunflower butter", "seed butter", "apple butter", "oat butter",
             "coconut cream", "coconut milk", "cream of tartar", "oat milk", "almond milk",
             "soy milk", "soya milk", "rice milk", "milk free", "dairy free", "milk thistle"),
    "eggs": ("egg free",),
    "gluten": ("gluten free", "wheat free"),
    "soy": ("soy free",),
    "peanuts": ("peanut free",),
    "tree_nuts": ("nut free", "tree nut free"),
    "sesame": ("sesame free",),
    "fish": ("fish free",),
    "crustaceans": (),
}

# Plain allergen words (beyond the hidden names). Used only for dataset sanity
# checks (bench/fetch_off.py), not as model evidence.
PLAIN_WORDS: dict[str, tuple[str, ...]] = {
    "milk": ("milk",), "eggs": ("egg",), "peanuts": ("peanut",), "tree_nuts": ("tree nut",),
    "gluten": ("gluten", "wheat"), "soy": ("soy", "soya"), "sesame": ("sesame",),
    "fish": ("fish",), "crustaceans": ("crustacean", "shellfish"),
}

_WORD = re.compile(r"[a-z0-9]+")


def _norm(w: str) -> str:
    """De-pluralise one lowercase word so plurals match their singular entry
    (hazelnuts -> hazelnut, eggs -> egg, cashews -> cashew). Matching stays
    whole-word, so 'butternut' still does NOT become 'butter'."""
    if len(w) <= 3:
        return w
    if w.endswith(("ches", "shes", "xes", "zes")):
        return w[:-2]  # peaches -> peach
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]  # nuts -> nut, eggs -> egg
    return w


def _tokens(text: str) -> list[str]:
    return [_norm(t) for t in _WORD.findall(text.lower())]


def _phrase_in(phrase: tuple[str, ...], tokens: list[str]) -> bool:
    """True if `phrase` appears as a contiguous run inside `tokens`."""
    if not phrase:
        return False
    for i in range(len(tokens) - len(phrase) + 1):
        if tokens[i:i + len(phrase)] == list(phrase):
            return True
    return False


def _masked(tokens: list[str], allergen: str) -> list[str]:
    """Replace every excluded phrase for `allergen` with placeholders."""
    out = list(tokens)
    for phrase in EXCLUSIONS.get(allergen, ()):
        p = _tokens(phrase)
        for i in range(len(out) - len(p) + 1):
            if out[i:i + len(p)] == p:
                out[i:i + len(p)] = ["_"] * len(p)
    return out


def match_ingredient(ingredient: str) -> list[tuple[str, str]]:
    """Return [(allergen, matched_synonym), ...] for one ingredient string."""
    tokens = _tokens(ingredient)
    if not tokens:
        return []
    hits: list[tuple[str, str]] = []
    for allergen in ALLERGENS:
        masked = _masked(tokens, allergen)
        for syn in SYNONYMS.get(allergen, ()):
            if _phrase_in(_tokens(syn), masked):
                hits.append((allergen, syn))
    return hits


def mentions(text: str, allergen: str) -> bool:
    """True if `text` names `allergen` by a plain word or a hidden name (exclusions applied)."""
    masked = _masked(_tokens(text), allergen)
    words = PLAIN_WORDS.get(allergen, ()) + SYNONYMS.get(allergen, ())
    return any(_phrase_in(_tokens(w), masked) for w in words)


def plain_mention(text: str, allergen: str) -> bool:
    """True if `text` names `allergen` by its plain word (milk, egg, wheat...)."""
    masked = _masked(_tokens(text), allergen)
    return any(_phrase_in(_tokens(w), masked) for w in PLAIN_WORDS.get(allergen, ()))


def evidence(
    ingredients: list[str],
    allergens: list[str] | None = None,
) -> dict[str, list[str]]:
    """Build the shared-state `evidence` field: {allergen: [line, ...]}.

    Each line is human readable for the "Why" panel, e.g.
    "whey powder -> 'whey' = milk (dictionary)".
    """
    wanted = allergens or list(ALLERGENS)
    out: dict[str, list[str]] = {}
    for ing in ingredients:
        for allergen, syn in match_ingredient(ing):
            if allergen in wanted:
                out.setdefault(allergen, []).append(
                    f"{ing} -> '{syn}' = {allergen.replace('_', ' ')} (dictionary)"
                )
    return out

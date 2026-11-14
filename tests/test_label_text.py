"""Statement parsing: a missed may-contain warning becomes a dangerous SAFE, so test the phrasings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safebite.label_text import parse_statements  # noqa: E402

WARNINGS = [
    "PRODUCED ON EQUIPMENT THAT ALSO PROCESSES PEANUTS, TREE NUTS, SESAME, DAIRY AND SOY.",
    "May contain traces of peanuts.",
    "Manufactured in a facility that also processes tree nuts.",
    "Made on shared equipment with milk and eggs.",
    "Processed in a plant that handles sesame.",
    "Packaged on the same line as products containing peanuts.",
    "This facility also processes wheat.",
]
NOT_WARNINGS = [
    "Made in a dedicated Peanut and Tree Nut free facility.",
    "Produced in a peanut-free facility.",
]


def test_may_contain_phrasings_are_caught():
    for w in WARNINGS:
        assert parse_statements(w)[1], w


def test_free_facility_is_not_a_warning():
    for t in NOT_WARNINGS:
        assert parse_statements(t)[1] is None, t


def test_croissant_label():
    text = ("INGREDIENTS: ENRICHED FLOUR (WHEAT FLOUR), BUTTER, EGGS, SOY FLOUR. "
            "ALLERGEN INFORMATION: CONTAINS: WHEAT, EGG, MILK, SOY.\n"
            "PRODUCED ON EQUIPMENT THAT ALSO PROCESSES PEANUTS, TREE NUTS, SESAME, DAIRY AND SOY.")
    con, may = parse_statements(text)
    assert con.lower().startswith("contains: wheat")
    assert "peanuts" in may.lower() and "sesame" in may.lower()

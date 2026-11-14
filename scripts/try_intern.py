"""Try the Intern Detective on a sample product.  Usage: .venv/bin/python scripts/try_intern.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safebite import intern_client  # noqa: E402
from safebite.synonyms import evidence  # noqa: E402

ingredients = ["sugar", "enriched wheat flour", "palm oil", "cocoa", "lecithin", "natural flavors", "salt"]
product = {
    "product": "Chocolate Sandwich Cookies",
    "ingredients": ingredients,
    "contains_statement": "Contains: wheat",
    "may_contain_statement": None,
    "evidence": evidence(ingredients),
}
result = intern_client.investigate(
    product, "soy", "severe",
    verdict={"soy": "check"},
    answers={"drex": {"soy": "cant_tell"}, "llm": {"soy": "may_contain"}},
)
print(json.dumps(result, indent=2))

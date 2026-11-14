"""Pydantic schemas (spec section 5). Ask before changing these."""
from typing import Literal

from pydantic import BaseModel

Allergen = Literal["milk", "eggs", "peanuts", "tree_nuts", "gluten", "soy",
                   "sesame", "fish", "crustaceans"]
Label = Literal["contains", "may_contain", "not_found", "cant_tell"]

ALLERGENS: list[str] = ["milk", "eggs", "peanuts", "tree_nuts", "gluten", "soy",
                        "sesame", "fish", "crustaceans"]
LABELS: list[str] = ["contains", "may_contain", "not_found", "cant_tell"]

# Open Food Facts tag -> our allergen name (ground truth mapping)
OFF_TAG_TO_ALLERGEN: dict[str, str] = {
    "en:milk": "milk",
    "en:eggs": "eggs",
    "en:peanuts": "peanuts",
    "en:nuts": "tree_nuts",
    "en:gluten": "gluten",
    "en:soybeans": "soy",
    "en:sesame-seeds": "sesame",
    "en:fish": "fish",
    "en:crustaceans": "crustaceans",
}


class Ingredient(BaseModel):  # NDI output, normalized
    text: str  # lowercase, stripped
    kind: Literal["ingredient", "contains_statement", "may_contain_statement"]
    page: int = 1
    box: list[float] | None  # [x1,y1,x2,y2] in image pixels
    score: float | None


class Evidence(BaseModel):
    allergen: Allergen
    ingredient: str
    layer: Literal["dictionary", "statement", "fuzzy"]
    detail: str


class AllergenAnswer(BaseModel):
    allergen: Allergen
    probs: dict[Label, float]  # Drex: real probabilities; LLM: confidence on its label
    label: Label  # argmax


class DecisionRun(BaseModel):  # one call to one model for one product
    model: str  # "drex-max-v1.1" or LLM model id
    answers: list[AllergenAnswer]
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    error: str | None = None


class Verdict(BaseModel):
    overall: Literal["safe", "check", "not_safe", "retake"]
    per_allergen: dict[Allergen, Literal["safe", "check", "not_safe"]]
    reasons: dict[Allergen, str]

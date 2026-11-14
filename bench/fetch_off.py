"""Build data/off_products.json from Open Food Facts (spec section 8.1).

Source: the official Open Food Facts dump on Hugging Face, read remotely with DuckDB
(only the needed columns are downloaded). The live OFF APIs rate-limit us (429/503).

Ground truth per allergen: tag in allergens_tags -> contains; else in traces_tags ->
may_contain; else not_found. Products whose tags plainly contradict their own
ingredient text (e.g. "milk" in the text but no milk tag) are dropped and counted.

Run:  .venv/bin/python bench/fetch_off.py [--target 240] [--seed 42]
Check (M2): 150+ products, >=40% with an allergen, hidden-name and may-contain
cases present; 20 samples printed for a hand spot-check.
"""
import argparse
import html
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from safebite.models import ALLERGENS, OFF_TAG_TO_ALLERGEN  # noqa: E402
from safebite.label_text import parse_statements as _shared_parse  # noqa: E402
from safebite.synonyms import match_ingredient, plain_mention, mentions  # noqa: E402

OUT_PATH = ROOT / "data" / "off_products.json"
PARQUET = ("https://huggingface.co/datasets/openfoodfacts/product-database/resolve/"
           "refs%2Fconvert%2Fparquet/default/food/0000.parquet")

SELECT = """
SELECT code,
  list_filter(product_name, lambda x: x.lang IN ('en','main'))[1].text AS product_name,
  list_filter(ingredients_text, lambda x: x.lang='en')[1].text AS ingredients_text_en,
  allergens_tags, traces_tags
FROM read_parquet('{url}')
WHERE list_contains(countries_tags,'en:united-states')
  AND len(list_filter(ingredients_text, lambda x: x.lang='en' AND length(x.text)>20))>0
  {extra}
LIMIT {limit}
"""


RAW_CACHE = ROOT / "data" / ".off_raw_rows.json"  # so re-runs never hit Hugging Face again


def fetch_rows(limit: int = 6000) -> list[tuple]:
    if RAW_CACHE.exists():
        rows = [tuple(r) for r in json.loads(RAW_CACHE.read_text())]
        print(f"  using cached rows: {len(rows)} ({RAW_CACHE.name})", flush=True)
        return rows
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    for attempt in range(3):
        try:
            t = time.time()
            rows = con.execute(SELECT.format(url=PARQUET, extra="", limit=limit)).fetchall()
            print(f"  US products with English ingredients: {len(rows)} rows in "
                  f"{time.time() - t:.1f}s", flush=True)
            break
        except duckdb.Error as e:
            if "429" not in str(e) or attempt == 2:
                raise
            print("  Hugging Face rate limit (429): waiting 90 s, then retrying ...", flush=True)
            time.sleep(90)
    rows = [(c, n, t, list(a or []), list(tr or [])) for c, n, t, a, tr in rows]
    RAW_CACHE.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


# --- record building -------------------------------------------------------

def _looks_english(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return False
    ascii_letters = [c for c in letters if "a" <= c.lower() <= "z"]
    return len(ascii_letters) / len(letters) > 0.9


def _split_ingredients(text: str) -> list[str]:
    """Split on commas; pull parenthetical/braced content out as separate items."""
    out: list[str] = []
    for part in re.split(r"[,;]", text):
        paren = re.findall(r"[(\[{]([^)\]}]*)[)\]}]", part)
        base = re.sub(r"[(\[{][^)\]}]*[)\]}]", "", part).strip(" .-:/*\r\n\t").lower()
        if base:
            out.append(base)
        for extra in paren:
            for item in extra.split(","):
                item = item.strip(" .-:/*").lower()
                if item:
                    out.append(item)
    return out


def _parse_statements(text: str) -> tuple[str | None, str | None]:
    may = re.search(r"(may contain|may also contain|traces? of|manufactured in a facility|"
                    r"processed in a facility|produced in a facility|shared equipment)[^.;]*",
                    text, re.IGNORECASE)
    con = re.search(r"\bcontains?\s*:?\s*(milk|egg|wheat|soy|peanut|tree nut|sesame|fish|"
                    r"shellfish|crustacean|almond|cashew|walnut|pecan|hazelnut)[^.;]*",
                    text, re.IGNORECASE)
    return (con.group(0).strip(" .;") if con else None,
            may.group(0).strip(" .;") if may else None)


def _ground_truth(allergens_tags: list, traces_tags: list) -> dict[str, str]:
    gt = {a: "not_found" for a in ALLERGENS}
    for tag in allergens_tags or []:
        if a := OFF_TAG_TO_ALLERGEN.get(tag):
            gt[a] = "contains"
    for tag in traces_tags or []:
        a = OFF_TAG_TO_ALLERGEN.get(tag)
        if a and gt[a] != "contains":
            gt[a] = "may_contain"
    return gt


# Names that hide the allergen (spec 8.1: whey, casein, semolina, tahini ...).
# Obvious names such as "almonds" or "soybean oil" do not count as hidden.
HIDDEN_NAMES: dict[str, set[str]] = {
    "milk": {"whey", "whey protein", "casein", "caseinate", "sodium caseinate", "lactose",
             "lactalbumin", "lactoglobulin", "ghee", "curds", "paneer"},
    "eggs": {"albumin", "ovalbumin", "lysozyme", "mayonnaise", "meringue", "ovomucoid"},
    "gluten": {"semolina", "spelt", "durum", "farina", "kamut", "einkorn", "couscous",
               "seitan", "bulgur", "barley", "rye", "malt"},
    "soy": {"edamame", "tofu", "tempeh", "miso", "textured vegetable protein"},
    "peanuts": {"groundnut", "arachis oil"},
    "tree_nuts": {"praline", "marzipan"},
    "sesame": {"tahini", "sesamol", "benne", "gingelly"},
    "fish": {"surimi", "anchovy"},
    "crustaceans": {"krill"},
}


def _hidden_only(gt: dict[str, str], ingredients: list[str], text: str) -> list[str]:
    """Allergens present only under a hidden name (e.g. whey but never the word milk)."""
    hits = {a for ing in ingredients for a, syn in match_ingredient(ing)
            if syn in HIDDEN_NAMES.get(a, set())}
    return [a for a, lab in gt.items()
            if lab == "contains" and a in hits and not plain_mention(text, a)]


def _clean(text: str) -> str:
    """Strip OFF's HTML markup (e.g. <span class="allergen">) so it can't leak the answer."""
    text = html.unescape(re.sub(r"<[^>]+>", "", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def build_record(row: tuple) -> tuple[dict | None, str]:
    code, name, text, allergens_tags, traces_tags = row
    text, name = _clean(text), _clean(name)
    if not text or not _looks_english(text):
        return None, "not_english"
    gt = _ground_truth(list(allergens_tags or []), list(traces_tags or []))
    # Tags clearly wrong: the label text names an allergen that has no tag at all.
    missing = [a for a in ALLERGENS if gt[a] == "not_found" and mentions(text, a)]
    if missing:
        return None, "tags_contradict_text"
    ingredients = _split_ingredients(text)
    con, may = _shared_parse(text)  # same parser as the Scan path
    return {
        "code": code,
        "product_name": (name or "").strip(),
        "ingredients_text_en": text,
        "allergens_tags": list(allergens_tags or []),
        "traces_tags": list(traces_tags or []),
        "ingredients": ingredients,
        "contains_statement": con,
        "may_contain_statement": may,
        "ground_truth": gt,
        "hidden_name_allergens": _hidden_only(gt, ingredients, text),
    }, "ok"


def _has_allergen(rec: dict) -> bool:
    return any(v != "not_found" for v in rec["ground_truth"].values())


def _has_may(rec: dict) -> bool:
    return any(v == "may_contain" for v in rec["ground_truth"].values())


def balance(records: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Pick a balanced, reproducible set: hidden-name, may-contain, other allergen, none."""
    rng.shuffle(records)
    hidden = [r for r in records if r["hidden_name_allergens"]]
    may = [r for r in records if _has_may(r) and r not in hidden]
    used = {r["code"] for r in hidden + may}
    other = [r for r in records if _has_allergen(r) and r["code"] not in used]
    none = [r for r in records if not _has_allergen(r)]
    quota = {"hidden": int(target * 0.20), "may": int(target * 0.20),
             "none": int(target * 0.35)}
    pick = hidden[:quota["hidden"]] + may[:quota["may"]] + none[:quota["none"]]
    pick += other[:target - len(pick)]
    rng.shuffle(pick)
    return pick


def report(records: list[dict], dropped: dict[str, int]) -> None:
    n = len(records)
    with_allergen = sum(_has_allergen(r) for r in records)
    print("\n=== summary ===")
    print(f"total products: {n}   (need >= 150)")
    print(f"with >= 1 allergen: {with_allergen} ({100 * with_allergen // max(n, 1)}%)   (need >= 40%)")
    print(f"with a may-contain label: {sum(_has_may(r) for r in records)}")
    print(f"with an allergen only under a hidden name: "
          f"{sum(bool(r['hidden_name_allergens']) for r in records)}")
    print(f"dropped before sampling: {dropped}")
    print("per allergen (contains / may_contain):")
    for a in ALLERGENS:
        c = sum(r["ground_truth"][a] == "contains" for r in records)
        m = sum(r["ground_truth"][a] == "may_contain" for r in records)
        print(f"   {a:12} {c:4} / {m}")


def spot_check(records: list[dict], rng: random.Random) -> None:
    sample = rng.sample(records, min(20, len(records)))
    print("\n=== spot-check these 20 by hand (does 'truth' match the ingredients?) ===")
    for i, r in enumerate(sample, 1):
        truth = {a: v for a, v in r["ground_truth"].items() if v != "not_found"}
        print(f"\n[{i}] {r['code']}  {r['product_name'][:50]}")
        print(f"    truth: {truth or 'none'}"
              + (f"   hidden-name: {r['hidden_name_allergens']}" if r["hidden_name_allergens"] else ""))
        print(f"    ingredients: {r['ingredients_text_en'][:160]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("Reading Open Food Facts (Hugging Face copy) ...", flush=True)
    rows = fetch_rows()
    records, dropped = [], {"not_english": 0, "tags_contradict_text": 0}
    for row in rows:
        rec, why = build_record(row)
        if rec:
            records.append(rec)
        else:
            dropped[why] += 1
    print(f"usable products: {len(records)} of {len(rows)}")

    chosen = balance(records, args.target, rng)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chosen, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(chosen)} products -> {OUT_PATH.relative_to(ROOT)}")
    report(chosen, dropped)
    spot_check(chosen, rng)


if __name__ == "__main__":
    main()

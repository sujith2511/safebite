"""Turn label text into an ingredient list and the contains / may-contain statements."""
import re

_CONTAINS = re.compile(
    r"\bcontains?\s*:?\s*(milk|eggs?|wheat|soy|soya|peanuts?|tree nuts?|sesame|fish|shellfish|"
    r"crustacean|almonds?|cashews?|walnuts?|pecans?|hazelnuts?)[^.;\n]*", re.IGNORECASE)
_MAY = re.compile(
    r"(may contain|may also contain|traces? of|"
    r"(?:produced|made|manufactured|processed|packaged|packed|prepared|baked)\s+(?:in|on|at|using)\s+"
    r"[^.;\n]*?(?:facility|facilities|equipment|line|lines|plant|premises)|"
    r"(?:shared|same)\s+(?:equipment|facility|line|production)|also (?:processes|handles|produces|packages))"
    r"[^.;\n]*", re.IGNORECASE)
_NEGATED = re.compile(r"\bfree\b|\bdoes not\b|\bno\b|\bwithout\b", re.IGNORECASE)


def split_ingredients(text: str) -> list[str]:
    """Split on commas/semicolons; parenthetical content becomes separate items."""
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


def parse_statements(text: str) -> tuple[str | None, str | None]:
    """(contains statement, may-contain statement). Several may-contain sentences are joined.

    'Made in a dedicated peanut-free facility' is a negation, not a warning, and is skipped.
    """
    con = _CONTAINS.search(text)
    mays = [m.group(0).strip(" .;") for m in _MAY.finditer(text) if not _NEGATED.search(m.group(0))]
    mays = list(dict.fromkeys(mays))
    return (con.group(0).strip(" .;") if con else None, "; ".join(mays) if mays else None)


def ingredient_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the ingredient list inside `text`, without the 'Ingredients:' header."""
    m = re.search(r"ingredients?\s*[:.]?\s*", text, re.IGNORECASE)
    start = m.end() if m else 0
    end = len(text)
    for stop in (_CONTAINS, _MAY, re.compile(r"allergen information", re.IGNORECASE)):
        s = stop.search(text, start)
        if s:
            end = min(end, s.start())
    return (start, end) if end > start else None

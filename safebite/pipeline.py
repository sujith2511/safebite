"""Scan path (spec 3): photo -> NDI + readability gate -> shared state -> Drex / LLM -> rules.

Every step is cached on disk by the image hash, so the demo also works offline.
Cached results keep the latency that was really measured when they were first run.
"""
import asyncio
import hashlib
import io

from PIL import Image, ImageOps

from . import cache, drex_client, llm_client, ndi_client
from .models import ALLERGENS, DecisionRun, Ingredient
from .state import build_state
from .synonyms import match_ingredient, mentions


def normalize_image(raw: bytes, max_side: int = 2000) -> tuple[bytes, Image.Image]:
    """Apply EXIF rotation and resize, so NDI's coordinates match the image we draw on."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), img


def image_key(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


async def read(image_bytes: bytes, img: Image.Image, use_cache: bool = True) -> dict:
    """NDI read + Drex readability gate, in parallel."""
    k = cache.key("scan-read-v2", image_key(image_bytes))
    if use_cache and (hit := cache.get(k)):
        hit["ingredients"] = [Ingredient(**i) for i in hit["ingredients"]]
        hit["cached"] = True
        return hit
    try:
        label = await ndi_client.read_label(image_bytes, img.width, img.height,
                                            filename=f"scan_{image_key(image_bytes)[:20]}.jpg")
    except Exception as e:  # noqa: BLE001  NDI failed: no ingredients -> the app shows RETAKE
        label = {"ingredients": [], "text": "", "ingredient_text": "", "contains_statement": None,
                 "may_contain_statement": None, "block_box": None, "ms": 0.0}
        return {**label, "readable_p_yes": None, "readable_error": f"NDI: {e}", "cached": False}
    # Readability gate on the label area NDI found (plus margin), not the whole scene.
    gate_bytes = image_bytes
    if label.get("block_box"):
        x1, y1, x2, y2 = label["block_box"]
        mx, my = 0.15 * (x2 - x1) + 20, 0.6 * (y2 - y1) + 20
        crop = img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx),
                         min(img.height, y2 + my)))
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=92)
        gate_bytes = buf.getvalue()
    p_yes, gate_err = await drex_client.readable(gate_bytes)
    result = {**label, "readable_p_yes": p_yes, "readable_error": gate_err}
    if p_yes is not None:
        cache.put(k, {**result, "ingredients": [i.model_dump() for i in label["ingredients"]]})
    result["cached"] = False
    return result


def state_from_read(r: dict, product: str = "Scanned packet") -> dict:
    ingredients = [i.text for i in r["ingredients"] if i.kind == "ingredient"]
    text = (r.get("text") or "").replace("--- Page 1 ---", "").strip()
    return build_state(product, ingredients, r.get("contains_statement"), r.get("may_contain_statement"),
                       label_text=text)


async def _decide(name: str, fn, state: dict, use_cache: bool) -> tuple[DecisionRun, bool]:
    k = cache.key("scan-decide-v2", name, state, ALLERGENS)
    if use_cache and (hit := cache.get(k)):
        return DecisionRun(**hit), True
    run = await fn(state)
    if not run.error:
        cache.put(k, run.model_dump())
    return run, False


def decide_drex(state: dict, use_cache: bool = True) -> tuple[DecisionRun, bool]:
    return asyncio.run(_decide("drex", lambda s: drex_client.decide(s, ALLERGENS), state, use_cache))


def decide_llm(state: dict, use_cache: bool = True, model: str | None = None) -> tuple[DecisionRun, bool]:
    model = model or llm_client.model_id()
    llm_client._clients.clear()  # SDK clients are tied to an event loop; each asyncio.run is a new loop
    return asyncio.run(_decide(model, lambda s: llm_client.decide(s, ALLERGENS, model=model),
                               state, use_cache))


def marks_for(read_result: dict, statuses: dict[str, str]) -> list[dict]:
    """Highlights for the user's flagged allergens: red = contains, yellow = may contain."""
    flagged = [a for a, s in statuses.items() if s != "safe"]
    marks = []
    for ing in read_result["ingredients"]:
        if ing.kind == "may_contain_statement":
            hit = [a for a in flagged if mentions(ing.text, a)]
            if hit:
                marks.append({"box": ing.box, "color": "yellow",
                              "label": "may contain " + ", ".join(a.replace("_", " ") for a in hit)})
            continue
        hits = [a for a, _ in match_ingredient(ing.text) if a in flagged]
        hits += [a for a in flagged if a not in hits and mentions(ing.text, a)]
        if hits:
            name = "contains" if ing.kind == "contains_statement" else ing.text
            marks.append({"box": ing.box, "color": "red",
                          "label": f"{name[:28]} -> {', '.join(a.replace('_', ' ') for a in hits)}"})
    return marks

"""Drex client (Nace hosted). All Drex HTTP lives here.

API (console.nace.ai/docs/drex/api-reference/systemone):
  POST {base}/v1/systemone
  body: {"model", "state", "questions": {id: {"type": "choice", "instructions", "criteria": {label: desc}}},
         "images"?: ["data:image/png;base64,..."]}   # images: drex-max-v1.1 only, max 8
  200:  {"model", "answers": {id: {"type":"choice","choice","confidence","probabilities": {label: p}}},
         "usage": {"input_tokens","output_tokens","image_tokens"?}, "evaluation_time_ms", "request_id"}
  err:  {"error": {"type","message","issues"?}, "request_id"}
"""
import asyncio
import base64
import json
import time
from typing import Any

import httpx

from . import config
from .http_retry import send
from .models import LABELS, AllergenAnswer, DecisionRun
from .pricing import drex_cost


def _headers() -> dict[str, str]:
    if not config.NACE_API_KEY:
        raise RuntimeError("NACE_API_KEY is not set (.env)")
    return {"Authorization": f"Bearer {config.NACE_API_KEY}", "Content-Type": "application/json"}


def image_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


async def raw_systemone(
    state: Any,
    questions: dict[str, dict],
    images: list[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[dict, float]:
    """One POST /v1/systemone. Returns (response_json, latency_ms). Raises on HTTP error."""
    body: dict[str, Any] = {"model": config.DREX_MODEL, "state": state, "questions": questions}
    if images:
        body["images"] = images
    own = client is None
    client = client or httpx.AsyncClient(timeout=config.HTTP_TIMEOUT_S)
    try:
        t0 = time.perf_counter()
        resp = await send(client, "POST", f"{config.NACE_BASE_URL}/v1/systemone", json=body, headers=_headers())
        latency_ms = (time.perf_counter() - t0) * 1000
    finally:
        if own:
            await client.aclose()
    if resp.status_code != 200:
        raise RuntimeError(f"Drex HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json(), latency_ms


def allergen_question(allergen: str) -> dict:
    """Spec 6.2: one choice question per allergen."""
    name = allergen.replace("_", " ")
    return {
        "type": "choice",
        "instructions": f"Does this product contain {name} or an ingredient derived from {name}?",
        "criteria": {
            "contains": f"An ingredient or the contains statement shows {name} or something made from it.",
            "may_contain": f"Only a may-contain / traces / shared-facility statement mentions {name}.",
            "not_found": f"The full ingredient list is readable and nothing in it is or comes from {name}.",
            "cant_tell": "The ingredient information is missing, unreadable or too ambiguous to decide.",
        },
    }


READABLE_QUESTION = {
    "type": "choice",
    "instructions": "Is the ingredient list on the food packet in Image 1 fully readable?",
    "criteria": {
        "yes": "Every ingredient word can be read clearly.",
        "partly": "Some words are blurry, cut off, glared or too small.",
        "no": "The ingredient list cannot be read, or Image 1 is not an ingredient label.",
    },
}

async def decide(state: dict, allergens: list[str],
                 client: httpx.AsyncClient | None = None) -> DecisionRun:
    """ONE Drex request with one choice question per allergen. Errors become run.error (never safe)."""
    questions = {a: allergen_question(a) for a in allergens}
    resp, err, latency_ms = None, None, 0.0
    for attempt in range(3):
        try:
            resp, latency_ms = await raw_systemone(state, questions, client=client)
            err = None
            break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            retryable = any(f"HTTP {c}" in err for c in (429, 500, 503, 529))
            if not retryable:
                break
            await asyncio.sleep(1.5 * (attempt + 1))  # back off, then retry
    if resp is None:
        return DecisionRun(model=config.DREX_MODEL, answers=[], latency_ms=latency_ms,
                           input_tokens=None, output_tokens=None, cost_usd=None,
                           error=err or "no response")

    answers = []
    for a in allergens:
        probs = (resp.get("answers", {}).get(a) or {}).get("probabilities") or {}
        probs = {k: float(v) for k, v in probs.items() if k in LABELS}
        if not probs:
            continue  # missing answer -> decide.py treats it as an error (check)
        answers.append(AllergenAnswer(allergen=a, probs=probs, label=max(probs, key=probs.get)))
    usage = resp.get("usage") or {}
    in_tok = usage.get("input_tokens")
    if in_tok is None:  # spec 6.2: estimate when not reported
        in_tok = len(json.dumps({"state": state, "questions": questions})) // 4
    return DecisionRun(model=resp.get("model", config.DREX_MODEL), answers=answers,
                       latency_ms=latency_ms, input_tokens=in_tok,
                       output_tokens=usage.get("output_tokens"), cost_usd=drex_cost(in_tok))


async def readable(image_bytes: bytes, mime: str = "image/jpeg") -> tuple[float | None, str | None]:
    """Readability gate (Scan path). Returns (P(yes), error). On error P is None (-> retake)."""
    try:
        resp, _ = await raw_systemone(
            "Image 1 is a photo of the back of a food packet.",
            {"readable": READABLE_QUESTION}, images=[image_data_url(image_bytes, mime)])
        p = (resp["answers"]["readable"].get("probabilities") or {}).get("yes")
        return (float(p) if p is not None else None), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"

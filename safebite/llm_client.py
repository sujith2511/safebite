"""Frontier LLM baseline client. All LLM HTTP lives here.

Provider from LLM_PROVIDER (openai | anthropic | google | deepinfra), model from LLM_MODEL.
Each provider is called through its official Python SDK.
"""
import json
import time

from . import config
from .models import LABELS, AllergenAnswer, DecisionRun
from .pricing import llm_cost

SYSTEM_PROMPT = (
    "You are a food allergen checker. For each allergen, answer exactly one of contains, "
    "may_contain, not_found, cant_tell and a confidence from 0 to 1. Reply with JSON only."
)

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "", "google": "",
                  "deepinfra": "deepseek-ai/DeepSeek-V4-Flash-0731"}


def model_id() -> str:
    m = config.LLM_MODEL or DEFAULT_MODELS.get(config.LLM_PROVIDER, "")
    if not m:
        raise RuntimeError(f"Set LLM_MODEL in .env for provider {config.LLM_PROVIDER!r}")
    return m


_clients: dict = {}


def _schema() -> dict:
    from .models import ALLERGENS
    item = {"type": "object", "additionalProperties": False, "required": ["label", "confidence"],
            "properties": {"label": {"type": "string", "enum": list(LABELS)},
                           "confidence": {"type": "number"}}}
    return {"type": "object", "additionalProperties": False, "required": list(ALLERGENS),
            "properties": {a: item for a in ALLERGENS}}


async def raw_complete(user_text: str, model: str | None = None) -> dict:
    """One request, JSON-only answer. Returns {text, input_tokens, output_tokens, latency_ms, model}."""
    model = model or model_id()
    provider, key = config.provider_for(model)

    if provider == "anthropic":
        import anthropic
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (.env)")
        async with anthropic.AsyncAnthropic(api_key=key, timeout=config.HTTP_TIMEOUT_S * 2) as client:
            t0 = time.perf_counter()
        # Current Claude models reject temperature (400) and keep thinking on;
        # low effort is the closest to a deterministic, fast baseline. No server-side
        # fallback: a refusal must not silently switch the model being measured.
            # Structured outputs = Claude's JSON mode (the other providers get response_format JSON).
            resp = await client.messages.create(
                model=model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema()}},
                messages=[{"role": "user", "content": user_text}],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
        if resp.stop_reason == "refusal":
            raise RuntimeError("LLM refused the request")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"text": text, "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens, "latency_ms": latency_ms, "model": model}

    if provider in ("openai", "deepinfra"):
        from openai import AsyncOpenAI
        # DeepInfra (hackathon partner) serves open frontier models behind an OpenAI-compatible API.
        if provider not in _clients:
            base_url = "https://api.deepinfra.com/v1/openai" if provider == "deepinfra" else None
            if not key:
                raise RuntimeError(f"No API key for provider {provider!r} (.env)")
            _clients[provider] = AsyncOpenAI(api_key=key, base_url=base_url,
                                             timeout=config.HTTP_TIMEOUT_S * 2, max_retries=2)
        client = _clients[provider]
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_text}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"text": resp.choices[0].message.content or "", "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens, "latency_ms": latency_ms, "model": model}

    if provider == "google":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        t0 = time.perf_counter()
        resp = await client.aio.models.generate_content(
            model=model,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0,
                response_mime_type="application/json"),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        u = resp.usage_metadata
        return {"text": resp.text or "", "input_tokens": u.prompt_token_count,
                "output_tokens": u.candidates_token_count, "latency_ms": latency_ms, "model": model}

    raise RuntimeError(f"Unknown LLM_PROVIDER {provider!r} (use openai | anthropic | google | deepinfra)")


def user_prompt(state: dict, allergens: list[str]) -> str:
    example = ", ".join(f'"{a}": {{"label": "...", "confidence": 0.0}}' for a in allergens)
    return ("Allergens to check: " + ", ".join(allergens) + "\n"
            "Product state:\n" + json.dumps(state, indent=2) + "\n"
            "Question for each allergen: does this product contain the allergen or an ingredient "
            "derived from it? Labels: contains (an ingredient or the contains statement shows it), "
            "may_contain (only a may-contain / traces statement mentions it), not_found (nothing in "
            "the list is or comes from it), cant_tell (information missing or unreadable).\n"
            "Reply with exactly this JSON shape and every allergen key: {" + example + "}")


def parse_answers(text: str, allergens: list[str]) -> list[AllergenAnswer]:
    """Strict parse. Raises ValueError on anything malformed."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    data = json.loads(t)
    if not isinstance(data, dict):
        raise ValueError("top level is not an object")
    out = []
    for a in allergens:
        item = data.get(a)
        if not isinstance(item, dict):
            raise ValueError(f"missing allergen {a}")
        label, conf = item.get("label"), item.get("confidence")
        if label not in LABELS:
            raise ValueError(f"bad label for {a}: {label!r}")
        conf = float(conf)
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence out of range for {a}: {conf}")
        # spec 7: P(label) = stated confidence, remainder spread evenly over the other options
        rest = (1.0 - conf) / (len(LABELS) - 1)
        probs = {lab: (conf if lab == label else rest) for lab in LABELS}
        out.append(AllergenAnswer(allergen=a, probs=probs, label=label))
    return out


async def decide(state: dict, allergens: list[str], model: str | None = None) -> DecisionRun:
    """ONE request per product (retried once on invalid JSON). Errors become run.error (never safe)."""
    model = model or model_id()
    prompt = user_prompt(state, allergens)
    latency, in_tok, out_tok, err = 0.0, 0, 0, None
    for _attempt in range(2):
        try:
            out = await raw_complete(prompt, model=model)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            break
        latency += out["latency_ms"]
        in_tok += out["input_tokens"] or 0
        out_tok += out["output_tokens"] or 0
        try:
            answers = parse_answers(out["text"], allergens)
            return DecisionRun(model=model, answers=answers, latency_ms=latency,
                               input_tokens=in_tok, output_tokens=out_tok,
                               cost_usd=llm_cost(model, in_tok, out_tok))
        except (ValueError, TypeError) as e:
            err = f"invalid JSON: {e}"
    return DecisionRun(model=model, answers=[], latency_ms=latency,
                       input_tokens=in_tok or None, output_tokens=out_tok or None,
                       cost_usd=llm_cost(model, in_tok, out_tok) if in_tok else None, error=err)

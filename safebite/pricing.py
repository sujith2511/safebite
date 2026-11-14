"""Price constants in USD per 1M tokens. Fill in current prices before the benchmark.

A price of 0 means "unknown" and triggers a warning; it is never silently used as free.
"""
import warnings

# Drex: billed on input tokens only (output is free). Value from the build spec;
# confirm on console.nace.ai/docs/drex/reference/pricing (the price table did not
# render when we read the docs).
DREX_INPUT_PER_M = 0.03

# Frontier LLMs: (input $/1M, output $/1M)
LLM_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic list prices (first-party API)
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-2": (8.00, 40.00),
    # DeepInfra list prices (api.deepinfra.com/models/list, 2026-11-14)
    "Qwen/Qwen3.8-27B": (0.20, 2.50),
    "Qwen/Qwen3.8-2.4T-A95B": (2.00, 6.00),
    "Qwen/Qwen3.8-Max": (1.65, 4.951),
    # OpenAI / Google: fill in the model you use, e.g. "gpt-...": (in, out)
}


def drex_cost(input_tokens: int | None) -> float | None:
    if input_tokens is None:
        return None
    return input_tokens * DREX_INPUT_PER_M / 1_000_000


def llm_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    price_in, price_out = LLM_PRICES.get(model, (0.0, 0.0))
    if price_in == 0 or price_out == 0:
        warnings.warn(f"pricing.py: no price set for LLM model {model!r}; cost will read 0", stacklevel=2)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

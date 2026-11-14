"""Run Drex and the frontier LLM(s) on the same products and compute metrics (spec 8.2-8.4).

Usage:
  .venv/bin/python bench/run_bench.py [--limit N] [--llm-model M ...] [--no-cache]
Outputs bench/results.json (raw runs + metrics) and bench/results.md (the table).
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from safebite import cache, config, drex_client, llm_client  # noqa: E402
from safebite.decide import verdict  # noqa: E402
from safebite.models import ALLERGENS, DecisionRun  # noqa: E402
from safebite.state import build_state  # noqa: E402
from safebite.synonyms import mentions  # noqa: E402

DATA = ROOT / "data" / "off_products.json"
OUT_JSON = ROOT / "bench" / "results.json"
OUT_MD = ROOT / "bench" / "results.md"
MAX_IN_FLIGHT = 8
SEVERITIES = {a: "severe" for a in ALLERGENS}  # severity does not change safe vs not-safe


async def run_one(name: str, product: dict, state: dict, fn, use_cache: bool) -> DecisionRun:
    # Claude runs use structured outputs (v2); keep them apart from earlier cached answers.
    k = cache.key(name + ("|json_schema" if name.startswith("claude-") else ""), state, ALLERGENS)
    if use_cache and (hit := cache.get(k)) and not hit.get("error"):
        return DecisionRun(**hit)
    result = await fn(state)
    if not result.error:
        cache.put(k, result.model_dump())
    return result


async def run_all(products: list[dict], models: dict, use_cache: bool) -> dict[str, list[DecisionRun]]:
    # Drex account limit is 8 in flight; the LLM is slow (reasoning), so allow more.
    sems = {m: asyncio.Semaphore(MAX_IN_FLIGHT if m == config.DREX_MODEL else 16) for m in models}
    done = {"n": 0}
    total = len(products) * len(models)

    async def task(m, i, p, st):
        async with sems[m]:
            r = await run_one(m, p, st, models[m], use_cache)
        done["n"] += 1
        if done["n"] % 50 == 0 or done["n"] == total:
            print(f"  {done['n']}/{total} calls done", flush=True)
        return m, i, r

    states = [build_state(p["product_name"], p["ingredients"], p["contains_statement"],
                          p["may_contain_statement"]) for p in products]
    jobs = [task(m, i, p, states[i]) for m in models for i, p in enumerate(products)]
    results = {m: [None] * len(products) for m in models}
    for m, i, r in await asyncio.gather(*jobs):
        results[m][i] = r
    return results


# --- metrics -------------------------------------------------------------------

def _pct(x):
    return None if x is None else round(100 * x, 1)


def metrics(products: list[dict], runs: list[DecisionRun]) -> dict:
    pos = pos_flagged = neg = neg_safe = hidden = hidden_flagged = correct = pairs = 0
    vis = vis_flagged = 0
    per_label = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in ("contains", "may_contain", "not_found")}
    conf_correct = []  # (confidence of argmax, correct?) for calibration
    for p, r in zip(products, runs):
        v = verdict(r, SEVERITIES)
        preds = {a.allergen: a for a in r.answers} if not r.error else {}
        for a in ALLERGENS:
            truth = p["ground_truth"][a]
            status = v.per_allergen[a]
            ans = preds.get(a)
            label = ans.label if ans else "cant_tell"
            pairs += 1
            correct += label == truth
            if ans:
                conf_correct.append((ans.probs[label], label == truth))
            for lab, c in per_label.items():
                c["tp"] += (label == lab and truth == lab)
                c["fp"] += (label == lab and truth != lab)
                c["fn"] += (label != lab and truth == lab)
            if truth in ("contains", "may_contain"):
                pos += 1
                pos_flagged += status != "safe"
                if mentions(p["ingredients_text_en"], a):  # allergen visible on the label text
                    vis += 1
                    vis_flagged += status != "safe"
                if a in p.get("hidden_name_allergens", []):
                    hidden += 1
                    hidden_flagged += status != "safe"
            else:
                neg += 1
                neg_safe += status == "safe"
    f1s = []
    for c in per_label.values():
        if c["tp"] + c["fp"] + c["fn"] == 0:
            continue  # label absent from both truth and predictions
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    # expected calibration error, 10 bins
    ece, bins = 0.0, [[] for _ in range(10)]
    for conf, ok in conf_correct:
        bins[min(int(conf * 10), 9)].append((conf, ok))
    for b in bins:
        if b:
            ece += len(b) / len(conf_correct) * abs(
                sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b))
    ok_runs = [r for r in runs if not r.error]
    lat = [r.latency_ms for r in ok_runs]
    costs = [r.cost_usd for r in ok_runs if r.cost_usd is not None]
    return {
        "products": len(products), "pairs": pairs, "errors": len(runs) - len(ok_runs),
        "wrong_safe_rate": _pct((pos - pos_flagged) / pos if pos else None),
        "wrong_safe_count": pos - pos_flagged, "positive_pairs": pos,
        "wrong_safe_visible_rate": _pct((vis - vis_flagged) / vis if vis else None),
        "wrong_safe_visible_count": vis - vis_flagged, "visible_positive_pairs": vis,
        "recall": _pct(pos_flagged / pos if pos else None),
        "safe_food_cleared": _pct(neg_safe / neg if neg else None),
        "accuracy_4way": _pct(correct / pairs if pairs else None),
        "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else None,
        "hidden_name_recall": _pct(hidden_flagged / hidden if hidden else None),
        "hidden_name_pairs": hidden,
        "median_latency_ms": round(statistics.median(lat)) if lat else None,
        "p95_latency_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))]) if lat else None,
        "cost_per_1000_usd": round(1000 * statistics.mean(costs), 4) if costs else None,
        "ece": round(ece, 3) if conf_correct else None,
        "latencies_ms": lat,
    }


def _ratio(llm, drex, higher_is_better: bool):
    if llm is None or drex is None:
        return "n/a"
    if higher_is_better:
        return f"{drex - llm:+.1f} pts"
    if drex == 0:
        return "Drex 0" if llm else "same"
    return f"{llm / drex:.1f}x lower" if llm >= drex else f"{drex / llm:.1f}x higher"


def write_markdown(all_metrics: dict, n: int) -> str:
    d = all_metrics[config.DREX_MODEL]
    llms = [m for m in all_metrics if m != config.DREX_MODEL]
    rows = [
        ("Wrong-SAFE rate (%) - lower is better", "wrong_safe_rate", False),
        ("Wrong-SAFE when the allergen is named on the label (%)", "wrong_safe_visible_rate", False),
        ("Recall, allergen present (%)", "recall", True),
        ("Safe foods correctly cleared (%)", "safe_food_cleared", True),
        ("4-way accuracy (%)", "accuracy_4way", True),
        ("Macro F1", "macro_f1", True),
        ("Hidden-name recall (%)", "hidden_name_recall", True),
        ("Median latency (ms)", "median_latency_ms", False),
        ("p95 latency (ms)", "p95_latency_ms", False),
        ("Cost per 1,000 scans ($)", "cost_per_1000_usd", False),
        ("Calibration error (ECE) - lower is better", "ece", False),
        ("Errors (calls)", "errors", False),
    ]
    head = ("| Metric | " + " | ".join(llms) + " | Drex Max v1.1 | "
            + " | ".join(f"Drex vs {m}" for m in llms) + " |")
    lines = [f"# SafeBite benchmark: Drex vs frontier LLMs\n",
             f"{n} Open Food Facts products x {len(ALLERGENS)} allergens = {n * len(ALLERGENS)} "
             f"allergen decisions per model. Identical input to every model; identical safety rules.\n",
             head, "|" + "---|" * (2 * len(llms) + 2)]
    for title, k, hib in rows:
        vals = " | ".join(str(all_metrics[m][k]) for m in llms)
        imps = " | ".join(_ratio(all_metrics[m][k], d[k], hib) for m in llms)
        lines.append(f"| {title} | {vals} | {d[k]} | {imps} |")
    lines.append(f"\nWrong-SAFE: {d['wrong_safe_count']} of {d['positive_pairs']} allergen-present pairs "
                 f"for Drex; " + "; ".join(f"{all_metrics[m]['wrong_safe_count']} for {m}" for m in llms)
                 + f". Hidden-name pairs: {d['hidden_name_pairs']}.\n\n"
                 "Every wrong-SAFE case we inspected is an allergen that Open Food Facts tags but that "
                 "appears nowhere in the product's ingredient text (for example a 'may contain' warning "
                 "printed elsewhere on the pack), so no model could see it. On allergens named on the "
                 f"label: {d['wrong_safe_visible_count']} of {d['visible_positive_pairs']} wrong-SAFE for "
                 "Drex; " + "; ".join(f"{all_metrics[m]['wrong_safe_visible_count']} for {m}" for m in llms)
                 + ".\n\nCosts: Drex input tokens x $0.03/1M (spec price; confirm on the Nace console). "
                 "LLMs: list prices (DeepInfra, Anthropic) x reported tokens. Latency: wall clock per "
                 "product call with up to 8 (Drex) / 16 (LLM) calls in flight. Claude runs at effort "
                 "'low' with structured JSON output (current Claude models do not accept temperature).")
    return "\n".join(lines) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--llm-model", action="append", help="repeatable; default LLM_MODEL from .env")
    ap.add_argument("--provider", help="override LLM_PROVIDER")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if args.provider:
        config.LLM_PROVIDER = args.provider
    products = json.loads(DATA.read_text())[: args.limit]
    llm_models = args.llm_model or [llm_client.model_id()]

    http = httpx.AsyncClient(timeout=config.HTTP_TIMEOUT_S)
    models = {config.DREX_MODEL: lambda st: drex_client.decide(st, ALLERGENS, client=http)}
    for m in llm_models:
        models[m] = (lambda mm: (lambda st: llm_client.decide(st, ALLERGENS, model=mm)))(m)

    print(f"Running {len(products)} products on {list(models)} ...", flush=True)
    t = time.time()
    runs = await run_all(products, models, use_cache=not args.no_cache)
    await http.aclose()
    print(f"done in {time.time() - t:.0f}s")

    all_metrics = {m: metrics(products, rs) for m, rs in runs.items()}
    OUT_JSON.write_text(json.dumps({
        "n_products": len(products), "allergens": ALLERGENS, "metrics": all_metrics,
        "runs": {m: [r.model_dump() for r in rs] for m, rs in runs.items()},
        "codes": [p["code"] for p in products],
    }, indent=1))
    md = write_markdown(all_metrics, len(products))
    OUT_MD.write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    asyncio.run(main())

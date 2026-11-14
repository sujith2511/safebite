"""PNG charts from bench/results.json (spec 8.4): latency, cost, safety, calibration.

Usage: .venv/bin/python bench/charts.py
Palette: validated categorical slots in fixed order: Drex = blue, then each LLM in the
order it appears in results.json (orange, aqua, yellow). Every bar carries a value label.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safebite import config  # noqa: E402

OUT = ROOT / "charts"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DREX_C = "#2a78d6"
LLM_COLORS = ["#eb6834", "#1baf7a", "#eda100"]  # categorical slots 2-4, fixed order

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def short(m: str) -> str:
    if m == config.DREX_MODEL:
        return "Drex Max v1.1"
    if m.startswith("claude-"):
        return "Claude " + m.removeprefix("claude-").replace("-", " ", 1).replace("-", ".").title()
    return m.split("/")[-1]


def _models(R):
    llms = [m for m in R["metrics"] if m != config.DREX_MODEL]
    return [config.DREX_MODEL] + llms, dict(zip([config.DREX_MODEL] + llms, [DREX_C] + LLM_COLORS))


def _best_ratio(R, key, llms):
    d = R["metrics"][config.DREX_MODEL][key]
    return min(R["metrics"][m][key] for m in llms) / d


def latency(R):
    models, colors = _models(R)
    fig, ax = plt.subplots(figsize=(8.5, 1.6 + 1.2 * len(models)))
    data = [R["metrics"][m]["latencies_ms"] for m in models]
    ys = list(range(len(models), 0, -1))
    bp = ax.boxplot(data, positions=ys, orientation="horizontal", widths=0.45, patch_artist=True,
                    medianprops={"color": INK, "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.4})
    for patch, m in zip(bp["boxes"], models):
        patch.set(facecolor=colors[m], edgecolor=colors[m], alpha=0.9)
    ax.set_xscale("log")
    ax.set_yticks(ys, [short(m) for m in models])
    ax.grid(axis="y", visible=False)
    for y, m in zip(ys, models):
        med = R["metrics"][m]["median_latency_ms"]
        ax.annotate(f"median {med:,} ms", (med, y + 0.3), ha="center", color=INK2, fontsize=11)
    ratio = _best_ratio(R, "median_latency_ms", models[1:])
    ax.set_title(f"Drex answers {ratio:.0f}x faster than the fastest LLM (median)", loc="left", fontsize=14)
    ax.set_xlabel("Time per product decision (ms, log scale)")
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=160)
    plt.close(fig)


def cost(R):
    models, colors = _models(R)
    fig, ax = plt.subplots(figsize=(8.5, 1.4 + 0.9 * len(models)))
    vals = [R["metrics"][m]["cost_per_1000_usd"] for m in models]
    ys = list(range(len(models) - 1, -1, -1))
    ax.barh(ys, vals, color=[colors[m] for m in models], height=0.55, edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks(ys, [short(m) for m in models])
    ax.grid(axis="y", visible=False)
    for y, v in zip(ys, vals):
        ax.text(v * 1.15, y, f"${v:,.3f}", va="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlim(min(vals) / 3, max(vals) * 8)
    ax.set_title(f"Drex is {_best_ratio(R, 'cost_per_1000_usd', models[1:]):.0f}x cheaper than the "
                 f"cheapest LLM", loc="left", fontsize=14)
    ax.set_xlabel("Cost per 1,000 scans (USD, log scale)")
    fig.tight_layout()
    fig.savefig(OUT / "cost.png", dpi=160)
    plt.close(fig)


def safety(R):
    models, colors = _models(R)
    metrics = [("wrong_safe_visible_rate", "Wrong-SAFE\n(allergen on label)\nlower is better"),
               ("recall", "Allergens\ncaught"),
               ("hidden_name_recall", "Hidden names\ncaught"),
               ("safe_food_cleared", "Safe foods\ncleared"),
               ("accuracy_4way", "Accuracy")]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        vals = [R["metrics"][m][k] or 0 for k, _ in metrics]
        xs = [i + (j - (len(models) - 1) / 2) * w for i in range(len(metrics))]
        ax.bar(xs, vals, width=w - 0.03, color=colors[m], label=short(m))
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f}" if v == int(v) else f"{v:.1f}", ha="center",
                    color=INK, fontsize=9)
    ax.set_xticks(range(len(metrics)), [t for _, t in metrics], fontsize=10.5)
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("%")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncols=len(models))
    ax.set_title("Safety and usefulness on the same 240 products", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "safety.png", dpi=160)
    plt.close(fig)


def calibration(R, products):
    models, colors = _models(R)
    truth = {p["code"]: p["ground_truth"] for p in products}
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--", label="Perfectly calibrated")
    for m in models:
        bins = [[0, 0] for _ in range(10)]
        for code, run in zip(R["codes"], R["runs"][m]):
            for a in run["answers"]:
                conf = a["probs"][a["label"]]
                b = bins[min(int(conf * 10), 9)]
                b[0] += 1
                b[1] += a["label"] == truth[code][a["allergen"]]
        pts = [((i + 0.5) / 10, b[1] / b[0]) for i, b in enumerate(bins) if b[0] >= 5]
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=colors[m], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=f"{short(m)} (ECE {R['metrics'][m]['ece']})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Actually correct")
    ax.set_title("Is the stated confidence honest?", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=160)
    plt.close(fig)


def main():
    R = json.loads((ROOT / "bench" / "results.json").read_text())
    products = json.loads((ROOT / "data" / "off_products.json").read_text())
    OUT.mkdir(exist_ok=True)
    latency(R)
    cost(R)
    safety(R)
    calibration(R, products)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()

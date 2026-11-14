# SafeBite benchmark: Drex vs frontier LLMs

240 Open Food Facts products x 9 allergens = 2160 allergen decisions per model. Identical input to every model; identical safety rules.

| Metric | Qwen/Qwen3.8-27B | claude-sonnet-5-5 | Drex Max v1.1 | Drex vs Qwen/Qwen3.8-27B | Drex vs claude-sonnet-5-5 |
|---|---|---|---|---|---|
| Wrong-SAFE rate (%) - lower is better | 5.4 | 0.3 | 5.4 | 1.0x lower | 18.0x higher |
| Wrong-SAFE when the allergen is named on the label (%) | 0.0 | 0.0 | 0.0 | same | same |
| Recall, allergen present (%) | 94.6 | 99.7 | 94.6 | +0.0 pts | -5.1 pts |
| Safe foods correctly cleared (%) | 63.8 | 14.2 | 81.5 | +17.7 pts | +67.3 pts |
| 4-way accuracy (%) | 96.4 | 91.8 | 97.5 | +1.1 pts | +5.7 pts |
| Macro F1 | 0.746 | 0.736 | 0.751 | +0.0 pts | +0.0 pts |
| Hidden-name recall (%) | 100.0 | 100.0 | 100.0 | +0.0 pts | +0.0 pts |
| Median latency (ms) | 16865 | 2103 | 357 | 47.2x lower | 5.9x lower |
| p95 latency (ms) | 93578 | 2406 | 628 | 149.0x lower | 3.8x lower |
| Cost per 1,000 scans ($) | 3.3038 | 6.5961 | 0.0364 | 90.8x lower | 181.2x lower |
| Calibration error (ECE) - lower is better | 0.049 | 0.104 | 0.011 | 4.5x lower | 9.5x lower |
| Errors (calls) | 0 | 0 | 0 | same | same |

Wrong-SAFE: 21 of 390 allergen-present pairs for Drex; 21 for Qwen/Qwen3.8-27B; 1 for claude-sonnet-5-5. Hidden-name pairs: 48.

Every wrong-SAFE case we inspected is an allergen that Open Food Facts tags but that appears nowhere in the product's ingredient text (for example a 'may contain' warning printed elsewhere on the pack), so no model could see it. On allergens named on the label: 0 of 352 wrong-SAFE for Drex; 0 for Qwen/Qwen3.8-27B; 0 for claude-sonnet-5-5.

Costs: Drex input tokens x $0.03/1M (spec price; confirm on the Nace console). LLMs: list prices (DeepInfra, Anthropic) x reported tokens. Latency: wall clock per product call with up to 8 (Drex) / 16 (LLM) calls in flight. Claude runs at effort 'low' with structured JSON output (current Claude models do not accept temperature).

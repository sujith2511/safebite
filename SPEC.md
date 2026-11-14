# SafeBite: build spec (text copy of ../LabelSafe_Track1_Spec.pdf)

Track 1 (Apex Predator) + NDI bonus. Nace Drexathon, Nov 14, 2026. Submit by 2:45 PM (hard deadline 3:00 PM).

Rules: build milestones (section 10) in order; after each, run its check, report, stop. Ask before changing a
schema (section 5) or a safety rule (section 7). Keep every NDI, Drex and LLM request inside its own client file.
Never write a code path that returns SAFE on an error, timeout or missing data. Keep code simple.

## 1. Pitch
SafeBite reads the ingredient label on a food packet and tells a person with allergies whether it is safe.
Track 1 result: a measured head-to-head benchmark. The same allergen decisions are made by a frontier LLM and by
Drex; compare accuracy, dangerous mistakes, speed and cost. Never claim a number before it is measured.

## 2. Scope
- Must have: NDI label reading with boxes; synonym dictionary; Drex client; LLM client; benchmark script +
  results table + charts; Streamlit app with Scan and Benchmark tabs; safety rules; disclaimer; cached demo.
- Nice to have: embedding fuzzy matcher; read-aloud; calibration chart.
- Cut: detective agent; family profiles; allergist-report import; crop tool.

## 3. Workflow
Scan path (live demo): photo -> NDI (extract + boxes) -> ingredients -> synonym dictionary -> evidence ->
Drex (one request, one choice question per allergen -> probabilities) and LLM (one request, same questions,
JSON answer -> labels + confidence) -> safety rules applied to BOTH -> UI: verdict, red/yellow boxes on photo,
Drex vs LLM side by side with latency.

Benchmark path: Open Food Facts products (ingredient text + allergen tags = ground truth) -> synonym dictionary
-> evidence -> Drex and LLM on identical state -> predictions + latency + tokens -> metrics -> results.json,
results.md, charts/*.png.

Scan steps: 1 user picks allergens + severity (severe/mild) each. 2 upload/camera photo. 3 NDI extracts each
ingredient plus "Contains:" and "May contain:" statements with page + box + score. 4 readability gate: Drex
question `readable` (yes/partly/no) on the photo; unless P(yes) >= 0.85 show "Please retake the photo" and stop.
5 synonym evidence (whey -> milk). 6 identical state to Drex and LLM. 7 safety rules. 8 result: big verdict
(from Drex), boxes on photo, comparison card (Drex answer + ms vs LLM answer + ms + cost).

## 4. Files
```
app.py                  Streamlit: tabs Scan | Benchmark
data/allergen_synonyms.json
data/off_products.json  benchmark set (built by bench/fetch_off.py)
data/demo_photos/       4-6 real packet photos
safebite/models.py     pydantic schemas (section 5)  [DONE]
safebite/ndi_client.py read_label(image_bytes) -> list[Ingredient]
safebite/drex_client.py decide(state, allergens) -> DecisionRun
safebite/llm_client.py decide(state, allergens) -> DecisionRun (same interface)
safebite/synonyms.py   dictionary evidence
safebite/decide.py     safety rules -> Verdict
safebite/draw.py       boxes on photo
safebite/cache.py      disk cache keyed by sha256 of input
safebite/pricing.py    price constants  [DONE]
bench/fetch_off.py      build data/off_products.json
bench/run_bench.py      run both models, compute metrics
bench/charts.py         PNG charts
tests/test_decide.py
```

## 5. Data models
See `safebite/models.py` (already written; do not change without asking). Allergens: milk, eggs, peanuts,
tree_nuts, gluten, soy, sesame, fish, crustaceans. Labels: contains, may_contain, not_found, cant_tell.
Open Food Facts tags: en:milk, en:eggs, en:peanuts, en:nuts (tree_nuts), en:gluten, en:soybeans (soy),
en:sesame-seeds (sesame), en:fish, en:crustaceans.

## 6. Model integration
Shared state (identical for both models):
```json
{"product": "Choco Crunch Bar",
 "ingredients": ["sugar", "wheat flour", "cocoa butter", "whey powder", "soy lecithin"],
 "contains_statement": "Contains: wheat, soy",
 "may_contain_statement": "May contain traces of peanuts",
 "evidence": {"milk": ["whey powder = milk (dictionary)"],
              "peanuts": ["may-contain statement mentions peanuts"]}}
```
Drex: ONE request per product with every allergen question (`drex_client.allergen_question`). Readability gate
on the Scan path only. Latency with time.perf_counter around the HTTP call. Cost = input_tokens x
pricing.DREX_INPUT_PER_M / 1M (output not billed). If no token count is reported, estimate len(text)/4.

LLM: same state and allergens, ONE request per product, temperature 0, system prompt in llm_client.SYSTEM_PROMPT.
Expected JSON: {"milk": {"label": "contains", "confidence": 0.95}, ...}. Parse strictly; on invalid JSON retry
once, then record error (counts as cant_tell, never safe). Cost from reported tokens x pricing.py; warn if price 0.

### 6.4 Synonym dictionary (whole-word, lowercase; "butternut" must NOT match "butter")
- milk: whey, whey protein, casein, caseinate, sodium caseinate, lactose, lactalbumin, lactoglobulin, ghee,
  butter, buttermilk, curds, milk solids, cream, paneer
- eggs: albumin, ovalbumin, lysozyme, mayonnaise, meringue, ovomucoid
- gluten: wheat, semolina, spelt, durum, farina, kamut, einkorn, couscous, seitan, bulgur, barley, rye, malt
- soy: soya, soybean, soy lecithin, edamame, tofu, tempeh, miso, textured vegetable protein
- peanuts: peanut, groundnut, arachis oil
- tree_nuts: almond, cashew, walnut, pecan, pistachio, hazelnut, macadamia, praline, marzipan
- sesame: sesame, tahini, sesamol, benne, gingelly
- fish: anchovy, fish sauce, surimi, cod, salmon, tuna
- crustaceans: shrimp, prawn, crab, lobster, krill
The dictionary is shared by both models, so it does not bias the comparison.

## 7. Safety rules (decide.py), identical for both models; thresholds are constants at the top of the file
1. Readability gate fails (Scan path) -> overall = retake, no allergen verdicts.
2. Model error or timeout for an allergen -> check. Never safe.
3. P(contains) >= 0.30 -> not_safe.
4. P(may_contain) >= 0.30 -> not_safe if severity is severe, else check.
5. P(not_found) >= 0.95 -> safe.
6. Otherwise -> check.
For the LLM, P(label) = its stated confidence; the remainder is spread evenly over the other options.
Overall verdict = worst per-allergen status (not_safe > check > safe). tests/test_decide.py covers every rule
plus the error case. Every results screen shows: "SafeBite is an assistant, not a guarantee. Always
double-check the label and follow your doctor's advice."

## 8. Benchmark
### 8.1 bench/fetch_off.py
- Pull 150-300 products from the Open Food Facts public search API, fields: code, product_name,
  ingredients_text_en, allergens_tags, traces_tags. English products with non-empty ingredients_text_en.
  Verify the current API URL in the Open Food Facts API docs (https://openfoodfacts.github.io/openfoodfacts-server/api/).
- Ground truth per allergen: tag in allergens_tags -> contains; else in traces_tags -> may_contain; else not_found.
- Balance: at least 40% of products contain at least one allergen; include products whose allergen appears only
  under a hidden name (whey, casein, semolina, tahini).
- Split ingredient text into a list (split on commas, keep parentheses content as separate items). Parse
  "contains" and "may contain" sentences if present.
- Save data/off_products.json. Spot-check 20 products by hand; drop products whose tags are clearly wrong and
  note the count in the README.
### 8.2 bench/run_bench.py
For each product: build shared state, call Drex and the LLM, store both DecisionRuns. asyncio, max 8 in flight,
disk cache for every response. Latency per call: report median and p95. Flags: --limit N, --provider, --no-cache.
### 8.3 Metrics
Wrong-SAFE rate (truth contains/may_contain but verdict safe; headline metric); recall (truth
contains/may_contain flagged not_safe or check); 4-way accuracy; macro F1 over the three truth labels;
hidden-name subset recall; median and p95 latency; cost per 1,000 scans; optional calibration (ECE).
### 8.4 Outputs
bench/results.json (raw runs + metrics), bench/results.md (table below). charts/latency.png (box plot),
charts/cost.png (bar, log scale if needed), charts/safety.png (wrong-SAFE and recall bars), optional
charts/calibration.png. Report improvements as ratios from medians ("Drex was 12x faster").

| Metric | Frontier LLM (<model>) | Drex Max v1.1 | Improvement |
|---|---|---|---|
| Wrong-SAFE rate | | | |
| Recall (allergen present) | | | |
| 4-way accuracy | | | |
| Hidden-name recall | | | |
| Median latency (ms) | | | |
| Cost per 1,000 scans ($) | | | |

## 9. Streamlit app (app.py)
Scan tab: sidebar profile (multiselect allergens + severe/mild toggle each); camera or upload; "Check label"
button with stage status (Reading label, Checking, Deciding); verdict banner from Drex: SAFE (green), CHECK
CAREFULLY (amber), NOT SAFE (red), RETAKE PHOTO (grey), always text + color; annotated photo (red boxes =
contains, yellow = may contain); comparison card, two columns Drex vs LLM with per-allergen answer, latency ms,
cost, disagreements highlighted; "Why" panel with evidence; disclaimer at the bottom.
Benchmark tab: loads bench/results.json, shows the table and PNG charts, short note on dataset size and how
ground truth was built. Never run the benchmark live; load cached results.

## 10. Milestones
| Time | Milestone | Check |
|---|---|---|
| 10:05-10:35 | M1 setup, models.py, .env, three clients making one raw call each | DONE: scripts/m1_smoke.py passes |
| 10:35-11:15 | M2 bench/fetch_off.py, safebite/synonyms.py | off_products.json has 150+ products; 20 spot-checked |
| 11:15-12:15 | M3 drex_client + llm_client decide(), decide.py, run_bench.py, tests | pytest passes; results.md with real numbers on 100+ products |
| 12:30-1:45 | M4 ndi_client.read_label, draw.py, Scan tab, Benchmark tab | 4 demo photos work end to end; blurry photo shows RETAKE |
| 1:45-2:15 | M5 charts + fixes | charts render in the Benchmark tab |
| 2:15-2:45 | M6 cache demo responses, 2-3 min video, README, slides | video recorded, repo pushed, submitted |

## 11. Demo script
1 Problem: real packet, tiny text, hidden names. 2 Claim: LLMs can do this but are slow and expensive.
3 Live scan of a whey/casein packet -> NOT SAFE, word boxed red; Drex ms vs LLM ms. 4 "May contain" packet:
severe vs mild changes the verdict. 5 Blurry photo -> RETAKE. 6 Benchmark tab, state the ratios.
7 Close: NDI brings the label in; Drex makes the safety call faster and cheaper.

## 12. Risks
NDI misreads -> ask to retake; the benchmark uses text input, so NDI is not on its critical path. OFF API slow ->
fetch once, save to disk; 100 products is enough. Schema differs -> change only the client file. Results do not
favour Drex on a metric -> report honestly. Wifi down in demo -> cached responses + recorded video.

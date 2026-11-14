# SafeBite: agent instructions

SafeBite reads the ingredient label on a food packet and says whether it is safe for a person with
allergies. For Drexathon Track 1 the main result is a measured benchmark: the same allergen decisions
are made by Drex and by a frontier LLM, and we compare accuracy, wrong-SAFE rate, latency and cost.

The full build spec is `SPEC.md` in this folder (a text copy of `../LabelSafe_Track1_Spec.pdf`). Read it before writing code. Submission
deadline is 3:00 PM on Nov 14, 2026; aim to submit by 2:45 PM.

## Workflow
- Build the milestones in spec section 10 in order. After each one, run its check, report the result, and stop.
- Ask before changing a schema (`safebite/models.py`, spec section 5) or a safety rule (spec section 7).
- Keep every NDI, Drex and LLM request inside its own client file (`ndi_client.py`, `drex_client.py`, `llm_client.py`).
- Never write a code path that returns SAFE on an error, timeout or missing data.
- Keep code simple. Python venv is `.venv/` (Python 3.12): run things with `.venv/bin/python`.
- Secrets live in `.env` (gitignored). Never print or commit keys.

## Status
- M1 (setup) is DONE and passing: `.venv/bin/python scripts/m1_smoke.py` gets real replies from NDI,
  Drex (drex-max-v1.1, ~315 ms) and DeepInfra (Qwen/Qwen3.8-27B, ~5.3 s). Keys are in `.env`.
- M2 is DONE. `safebite/synonyms.py` (whole-word, plurals, exclusions such as cocoa butter) and
  `bench/fetch_off.py`, which read the OFF dump on Hugging Face. Raw rows are cached in `data/.off_raw_rows.json`,
  so re-runs are offline. `data/off_products.json` has 240 products: 65% with an allergen, 20 may-contain,
  48 hidden-name (field `hidden_name_allergens`). 361 were dropped because their tags contradicted their own
  text, and HTML markup was stripped.
- Intern Detective is DONE: `safebite/intern_client.py` `investigate(product, allergen, severity, verdict, answers)`
  runs `intern -p --json` (System 1 drex-latest, System 2 deepinfra/Qwen/Qwen3.8-27B) in a temporary folder
  with product.json and allergen_reference.md. It returns the answer, turns, system1_checks and tool_calls.
  The answer is an explanation only and never changes a verdict. Try it with `scripts/try_intern.py`
  (about 11 s, 3 tool calls, 4 Drex checks). The Scan tab button "Ask Intern to investigate" comes in M4.
- M3 code is DONE: `safebite/state.py` (the shared state), `drex_client.decide()` (one request, retries 429/5xx),
  `llm_client.decide()` (strict JSON, one retry), `safebite/decide.py` (section 7 rules), `safebite/cache.py`,
  `bench/run_bench.py` (`--limit`, `--llm-model` repeatable, `--provider`, `--no-cache`), and
  `tests/test_decide.py` (13 tests passing). Results go to `bench/results.json` and `bench/results.md`.
  Qwen3.8-27B is a reasoning model, so it takes about 35 s per call. Drex takes about 0.5 s.
- M4 is DONE: `ndi_client.read_label` (box estimated within NDI's paragraph box; statements searched in every
  block), `safebite/pipeline.py` (scan path, all cached), `draw.py`, and `app.py` (Scan tab with Drex vs LLM,
  the Intern button and the Why panel; Benchmark tab). Run it with `.venv/bin/streamlit run app.py`.
  The readability gate runs on a crop of NDI's ingredient block. Real can photos with glare score 0.56-0.78
  and show RETAKE: retake them flat and close.
- M5 is DONE: `bench/charts.py` writes charts/latency.png, cost.png, safety.png and calibration.png
  (validated blue/orange palette). The Benchmark tab shows them.
- Claude Sonnet 5.5 added as a second baseline. `claude-*` models route to Anthropic via ANTHROPIC_API_KEY
  (`config.provider_for`), with effort low and structured JSON output. The 3-model results are in bench/results.md.
  Run with `--llm-model Qwen/Qwen3.8-27B --llm-model claude-sonnet-5-5`. Sonnet is very cautious: 1 wrong-SAFE
  overall, but it clears only 14% of safe foods. The app's sidebar lets you pick the live comparison model
  (default Sonnet).
- Safety fix (14:00): the may-contain parser missed "PRODUCED ON EQUIPMENT THAT ALSO PROCESSES PEANUTS…",
  which caused a wrong SAFE on a croissant photo. `safebite/label_text.py` now catches produced/made/processed
  on/in equipment/facility/line, "also processes" and shared lines, and skips "…free facility".
  `tests/test_label_text.py` covers this. Scans also pass `full_label_text` in the state. Scan caches were
  bumped to v2. The benchmark dataset is unchanged (0 statements differ), so the results stand.
- Next: M6 ship: README, cached demo, video, slides, push, submit.

## API facts already checked against the docs
- Drex: `POST https://console.nace.ai/v1/systemone`, `Authorization: Bearer $NACE_API_KEY`,
  body `{"model": "drex-max-v1.1", "state": ..., "questions": {id: {"type": "choice", "instructions": ...,
  "criteria": {label: description}}}, "images"?: ["data:image/...;base64,..."]}`.
  The answer is in `answers[id].probabilities` (label -> p), along with `usage.input_tokens` (the only billed field)
  and `evaluation_time_ms`. Use a client timeout of 60 s or more (Drex returns 529 after 55 s).
  Errors come back as `{"error": {"type", "message"}}`. 429/529 carry `retry-after-ms`.
- NDI: `POST /v1/documents/upload-grants {"path": name}` -> `upload_url` + `token`; then multipart POST the file
  to `upload_url` with `X-Upload-Token`; then `POST /v1/documents/parse?wait_seconds=60` with
  `{"source": {"type": "workspace_file", "workspace_id", "file_id"}}`; poll `GET /v1/documents/jobs/{id}`.
  Confirmed response shape: `result.document.text`, and `result.document.blocks[]` each with `content`
  and `location: {"kind": "page_region", "page": 1, "polygons": [[[x,y], [x,y], [x,y], [x,y]]]}`, where
  x and y are normalized 0-1 (multiply by image width/height for pixels). Blocks are whole LINES, not single
  ingredients: to box one ingredient, use `/v1/documents/ground` with targets, or split the line box
  in proportion to character position. Parse costs $0.0035 per page.
- LLM baseline: `LLM_PROVIDER` in openai | anthropic | google | deepinfra. DeepInfra uses the OpenAI SDK with
  `base_url="https://api.deepinfra.com/v1/openai"`.

## Open Food Facts: USE THE HUGGING FACE COPY (tested 11:50, 2026-11-14)
The live OFF APIs now return 429 (rate-limited) for us. Use the official OFF dump on Hugging Face instead,
read remotely with DuckDB (`duckdb` is in requirements). It needs no key and has no rate limit: 3000 rows took 17 s.
```python
import duckdb
U = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/refs%2Fconvert%2Fparquet/default/food/0000.parquet"
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
rows = con.execute(f"""
SELECT code,
  list_filter(product_name, lambda x: x.lang IN ('en','main'))[1].text AS product_name,
  list_filter(ingredients_text, lambda x: x.lang='en')[1].text AS ingredients_text_en,
  allergens_tags, traces_tags
FROM read_parquet('{U}')
WHERE list_contains(countries_tags,'en:united-states')
  AND len(list_filter(ingredients_text, lambda x: x.lang='en' AND length(x.text)>20))>0
LIMIT 3000""").fetchall()
```
Use the new lambda syntax (`lambda x: ...`); DuckDB 2 rejects `x -> ...`. `product_name` and
`ingredients_text` are lists of `{lang, text}`. Products with traces are rare (about 1%), so run a second
query with `len(traces_tags) > 0` to get enough may_contain cases. Sample from the result to balance the set.

## Open Food Facts live API (now rate-limited; kept for reference)
- The old search (`world.openfoodfacts.org/api/v2/search` and `/cgi/search.pl`) returns HTTP 503. Do NOT use it.
- Step 1, get barcodes: `GET https://search.openfoodfacts.org/search` with params
  `q` (Lucene query, e.g. `countries_tags:"en:united-states" AND allergens_tags:"en:milk"`, or
  `ingredients_tags:"en:whey"` for hidden-name products), `page_size` (max 100), `page`, `langs=en`,
  `fields=code,product_name,allergens_tags,traces_tags`. Results are in `hits`.
  This index has NO ingredient text, so it only gives barcodes.
- Step 2, get ingredients: `GET https://world.openfoodfacts.org/api/v2/product/{code}?fields=product_name,ingredients_text_en,allergens_tags,traces_tags`
  -> `{"product": {...}, "status": 1}`. Keep only products with non-empty `ingredients_text_en`.
  Rate limit about 100 product reads per minute: sleep 0.7 s between calls and send a User-Agent header
  like `SafeBite-hackathon/0.1`.
- Mix queries so that at least 40% of products have an allergen, some have none, and some have an allergen
  only under a hidden name (whey, casein, semolina, tahini). Fetch about 400 barcodes to end with 150-300 products.

## Open decisions (ask the user)
- Drex price: the spec says $0.03 per 1M input tokens, but Nace marketing says $0.03 per 1k decisions. Confirm before quoting cost.
- Adding `tokens_estimated: bool = False` to `DecisionRun` (a schema change, so it needs approval).

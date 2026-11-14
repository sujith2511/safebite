"""NDI (Nace document intelligence / Perception) client. All NDI HTTP lives here.

Flow (console.nace.ai/docs/perception/*):
  1. POST /v1/documents/upload-grants {"path": name} -> workspace_id, upload_url, token
  2. POST upload_url (multipart: file, metadata) with X-Upload-Token -> result.file.file_id
  3. POST /v1/documents/{parse|ground}?wait_seconds=N {"source": {"type":"workspace_file",...}, ...}
     -> 200 with result, or 202 {job_id, status:"queued"}
  4. GET /v1/documents/jobs/{job_id} until status in succeeded|failed|cancelled
Bounding boxes come back as normalized (0-1) polygons on `page_region` locations; the
exact field name was not in the docs, so M1 prints the raw parse result to find it.
"""
import asyncio
import json
import math
import re
import time
from typing import Any

import httpx

from . import config
from .http_retry import send
from .label_text import ingredient_span, parse_statements
from .models import Ingredient

DONE = {"succeeded", "failed", "cancelled"}


def _headers() -> dict[str, str]:
    if not config.NACE_API_KEY:
        raise RuntimeError("NACE_API_KEY is not set (.env)")
    return {"Authorization": f"Bearer {config.NACE_API_KEY}"}


def _dig(d: Any, *keys: str) -> Any:
    """Return the first matching key found at the top level or under 'result'/'file'/'data'."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    for wrapper in ("result", "file", "data", "grant"):
        if isinstance(d.get(wrapper), dict):
            v = _dig(d[wrapper], *keys)
            if v is not None:
                return v
    return None


async def upload(client: httpx.AsyncClient, image_bytes: bytes, filename: str) -> dict:
    """Upload bytes to the workspace. Returns a workspace_file source dict."""
    r = await send(client, "POST", f"{config.NACE_BASE_URL}/v1/documents/upload-grants",
                          json={"path": filename}, headers=_headers())
    if r.status_code >= 300:
        raise RuntimeError(f"NDI upload-grant HTTP {r.status_code}: {r.text[:500]}")
    grant = r.json()
    workspace_id = _dig(grant, "workspace_id")
    upload_url, token = _dig(grant, "upload_url"), _dig(grant, "token")
    if not (workspace_id and upload_url and token):
        raise RuntimeError(f"NDI upload-grant: unexpected shape {json.dumps(grant)[:500]}")
    r = await send(client, "POST", 
        upload_url,
        headers={"X-Upload-Token": token},
        files={"file": (filename, image_bytes)},
        data={"metadata": json.dumps({"path": filename})},
    )
    if r.status_code >= 300:
        raise RuntimeError(f"NDI upload HTTP {r.status_code}: {r.text[:500]}")
    file_id = _dig(r.json(), "file_id")
    if not file_id:
        raise RuntimeError(f"NDI upload: no file_id in {r.text[:500]}")
    return {"type": "workspace_file", "workspace_id": workspace_id, "file_id": file_id}


async def run_tool(client: httpx.AsyncClient, tool: str, body: dict,
                   wait_seconds: int = 60, poll_timeout_s: float = 120) -> dict:
    """Create a documents job and wait for it. Returns the finished job JSON."""
    r = await send(client, "POST", f"{config.NACE_BASE_URL}/v1/documents/{tool}",
                          params={"wait_seconds": wait_seconds}, json=body, headers=_headers())
    if r.status_code not in (200, 202):
        raise RuntimeError(f"NDI {tool} HTTP {r.status_code}: {r.text[:500]}")
    job = r.json()
    job_id = job.get("job_id") or job.get("id")
    deadline = asyncio.get_running_loop().time() + poll_timeout_s
    delay = 1.0
    while job.get("status") not in DONE and job_id:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"NDI {tool} job {job_id} did not finish")
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 5.0)
        r = await send(client, "GET", f"{config.NACE_BASE_URL}/v1/documents/jobs/{job_id}", headers=_headers())
        if r.status_code != 200:
            raise RuntimeError(f"NDI job read HTTP {r.status_code}: {r.text[:500]}")
        job = r.json()
    if job.get("status") not in (None, "succeeded"):
        raise RuntimeError(f"NDI {tool} job ended {job.get('status')}: {json.dumps(job)[:500]}")
    return job


async def raw_parse(image_bytes: bytes, filename: str = "label.jpg") -> dict:
    """M1 smoke call: upload a photo and parse it into text + blocks (with page regions)."""
    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT_S + 10) as client:
        source = await upload(client, image_bytes, filename)
        return await run_tool(client, "parse", {
            "source": source,
            "output": {"formats": ["text", "blocks"]},
        })

def _bbox(location: dict) -> list[float] | None:
    polys = (location or {}).get("polygons") or []
    pts = [pt for poly in polys for pt in poly]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]  # normalized 0-1


def _pick_ingredient_block(blocks: list[dict]) -> dict | None:
    text_blocks = [b for b in blocks if b.get("type") == "text" and b.get("content")]
    for b in text_blocks:
        if re.search(r"\bingredients?\b", b["content"], re.IGNORECASE):
            return b
    # no header (e.g. soda cans): take the block that looks most like a list
    return max(text_blocks, key=lambda b: b["content"].count(","), default=None)


def _span_box(block_box: list[float], text: str, start: int, end: int,
              img_w: int, img_h: int) -> list[float]:
    """Estimate the pixel box of text[start:end] inside a wrapped paragraph box.

    NDI gives one box per paragraph (no line breaks), so the number of lines is estimated
    from the box shape (line height ~ 2 x average character width), and the span is placed
    by character position. It is an approximation, drawn as a highlight.
    """
    x1, y1, x2, y2 = block_box[0] * img_w, block_box[1] * img_h, block_box[2] * img_w, block_box[3] * img_h
    w, h, n_chars = max(x2 - x1, 1), max(y2 - y1, 1), max(len(text), 1)
    lines = max(1, round(math.sqrt(h * n_chars / (2.0 * w))))
    per_line = n_chars / lines
    line_h = h / lines
    row = min(int(start // per_line), lines - 1)
    col0 = (start - row * per_line) / per_line
    col1 = min((end - row * per_line) / per_line, 1.0)
    pad = 0.6 / per_line
    return [x1 + max(col0 - pad, 0) * w, y1 + row * line_h,
            x1 + min(col1 + pad, 1) * w, y1 + (row + 1) * line_h]


async def read_label(image_bytes: bytes, img_w: int, img_h: int,
                     filename: str = "label.jpg") -> dict:
    """Read a packet photo with NDI.

    Returns {"ingredients": [Ingredient], "text", "ingredient_text", "contains_statement",
             "may_contain_statement", "block_box" (pixels), "ms"}.
    """
    t0 = time.perf_counter()
    job = await raw_parse(image_bytes, filename)
    doc = (job.get("result") or {}).get("document") or {}
    blocks = doc.get("blocks") or []
    block = _pick_ingredient_block(blocks)
    ms = (time.perf_counter() - t0) * 1000
    if not block:
        return {"ingredients": [], "text": doc.get("text") or "", "ingredient_text": "",
                "contains_statement": None, "may_contain_statement": None, "block_box": None, "ms": ms}

    text = block["content"]
    box = _bbox(block.get("location"))
    con, may = parse_statements(text)
    # Statements are often printed as separate paragraphs: search every text block too.
    stmt_blocks: dict[str, dict] = {}
    for b in blocks:
        if b is block or b.get("type") != "text" or not b.get("content"):
            continue
        c2, m2 = parse_statements(b["content"])
        if c2 and not con:
            con, stmt_blocks["contains_statement"] = c2, b
        if m2 and not may:
            may, stmt_blocks["may_contain_statement"] = m2, b
    span = ingredient_span(text)
    items: list[Ingredient] = []
    if span:
        s0, s1 = span
        pos = s0
        for part in re.split(r"([,;()\[\]])", text[s0:s1]):
            if part and part not in ",;()[]":
                clean = part.strip(" .-:/*\n\t")
                if clean:
                    start = text.index(clean, pos)
                    items.append(Ingredient(
                        text=clean.lower(), kind="ingredient", page=1,
                        box=_span_box(box, text, start, start + len(clean), img_w, img_h) if box else None,
                        score=None))
            pos += len(part)
    for stmt, kind in ((con, "contains_statement"), (may, "may_contain_statement")):
        if not stmt:
            continue
        if kind in stmt_blocks:  # statement has its own paragraph: box the whole paragraph
            sb = _bbox(stmt_blocks[kind].get("location"))
            sbox = [sb[0] * img_w, sb[1] * img_h, sb[2] * img_w, sb[3] * img_h] if sb else None
        else:
            start = text.find(stmt)
            sbox = (_span_box(box, text, start, start + len(stmt), img_w, img_h)
                    if box and start >= 0 else None)
        items.append(Ingredient(text=stmt.lower(), kind=kind, page=1, box=sbox, score=None))
    return {
        "ingredients": items,
        "text": doc.get("text") or "",
        "ingredient_text": text,
        "contains_statement": con,
        "may_contain_statement": may,
        "block_box": [box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h] if box else None,
        "ms": ms,
    }

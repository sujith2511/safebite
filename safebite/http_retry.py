"""Polite retries for Nace calls (organizer guidance): back off, respect retry-after.

Retries 429 / 503 / 529 and the venue firewall's 403 "Vercel Security Checkpoint" page.
Other errors return immediately so the caller can handle them (never as "safe").
"""
import asyncio
import random

import httpx

RETRY_STATUS = {429, 500, 502, 503, 529}
MAX_TRIES = 4


def _wait_s(resp: httpx.Response, attempt: int) -> float:
    if ms := resp.headers.get("retry-after-ms"):
        return float(ms) / 1000
    if s := resp.headers.get("retry-after"):
        try:
            return float(s)
        except ValueError:
            pass
    base = 8.0 if resp.status_code == 403 else 1.5  # firewall blocks clear in ~30 s
    return base * (2 ** attempt) + random.uniform(0, 1)


def _retryable(resp: httpx.Response) -> bool:
    if resp.status_code in RETRY_STATUS:
        return True
    return resp.status_code == 403 and "Vercel Security Checkpoint" in resp.text


async def send(client: httpx.AsyncClient, method: str, url: str, **kw) -> httpx.Response:
    for attempt in range(MAX_TRIES):
        resp = await client.request(method, url, **kw)
        if not _retryable(resp) or attempt == MAX_TRIES - 1:
            return resp
        await asyncio.sleep(min(_wait_s(resp, attempt), 40))
    return resp

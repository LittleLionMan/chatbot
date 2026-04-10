from __future__ import annotations
import logging
import httpx
from bot import config

logger = logging.getLogger(__name__)

_SEARXNG_URL = f"{config.SEARXNG_BASE_URL}/search"
_MAX_RESULTS = 8
_MAX_SNIPPET_CHARS = 300

_VALID_TIME_RANGES = {"day", "week", "month", "year"}


def _format_results(results: list[dict]) -> str:
    if not results:
        return "Keine Suchergebnisse gefunden."
    lines: list[str] = []
    for i, r in enumerate(results[:_MAX_RESULTS], 1):
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        snippet = r.get("content", "").strip()
        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[:_MAX_SNIPPET_CHARS] + "…"
        lines.append(f"[{i}] {title}\n{url}\n{snippet}")
    return "\n\n".join(lines)


async def search(query: str, language: str = "de-DE", time_range: str | None = None) -> str:
    try:
        params: dict[str, str] = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": "0",
            "categories": "general",
        }
        if time_range and time_range in _VALID_TIME_RANGES:
            params["time_range"] = time_range

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_SEARXNG_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            results: list[dict] = data.get("results", [])
            logger.info("SearXNG query '%s' (time_range=%s): %d results", query, time_range or "none", len(results))
            return _format_results(results)
    except httpx.ConnectError:
        logger.warning("SearXNG not reachable")
        return ""
    except Exception as e:
        logger.warning("SearXNG search failed for query '%s': %s", query, e)
        return ""


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{config.SEARXNG_BASE_URL}/healthz")
            return resp.status_code == 200
    except Exception:
        return False
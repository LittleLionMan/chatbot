from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_httpx, listing

logger = logging.getLogger(__name__)

_SUBREDDITS: dict[str, list[str]] = {
    "gpu": ["hardwareswap", "buildapcsales", "hardware_de"],
    "apartment": ["wohnen_de"],
    "job": ["forhire"],
    "default": ["hardwareswap", "buildapcsales"],
}

_SEARCH_URL = "https://old.reddit.com/r/{subreddit}/search"


def _parse_price_from_title(title: str) -> tuple[float | None, str | None]:
    match = re.search(r"[\$€£]?\s*(\d[\d,.]+)\s*[\$€£]?", title)
    if not match:
        return None, None
    raw = match.group(1).replace(",", "")
    try:
        price = float(raw)
    except ValueError:
        return None, None
    currency = None
    if "$" in title:
        currency = "USD"
    elif "€" in title:
        currency = "EUR"
    elif "£" in title:
        currency = "GBP"
    return price, currency


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    subreddits = _SUBREDDITS.get(category, _SUBREDDITS["default"])
    results: list[dict] = []

    for subreddit in subreddits:
        url = _SEARCH_URL.format(subreddit=subreddit)
        try:
            html = await fetch_with_httpx(url, params={
                "q": query,
                "restrict_sr": "on",
                "sort": "new",
                "t": "week",
            })
        except Exception as e:
            logger.warning("Reddit fetch failed for r/%s: %s", subreddit, e)
            continue

        soup = BeautifulSoup(html, "lxml")
        for item in soup.select(".search-result-link")[:20]:
            try:
                href = item.get("href", "")
                if not href.startswith("http"):
                    href = "https://old.reddit.com" + href
                ext_id_match = re.search(r"/comments/([a-z0-9]+)/", href)
                if not ext_id_match:
                    continue
                ext_id = f"{subreddit}_{ext_id_match.group(1)}"
                title_el = item.select_one(".search-title")
                title = title_el.get_text(strip=True) if title_el else ""
                if not title:
                    continue
                price, currency = _parse_price_from_title(title)
                results.append(listing(
                    external_id=ext_id,
                    url=href,
                    title=title,
                    price=price,
                    currency=currency,
                    attributes={"subreddit": subreddit, "source_query": query},
                ))
            except Exception as e:
                logger.debug("Failed to parse Reddit item: %s", e)

    logger.info("Reddit: %d listings for %r", len(results), query)
    return results

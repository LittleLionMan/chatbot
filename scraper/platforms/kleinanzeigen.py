from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_playwright, listing

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.kleinanzeigen.de/s-anzeige:angebote/preis:{price_min}:{price_max}/{query}/k0"
_SEARCH_URL = "https://www.kleinanzeigen.de/s-anzeige:angebote/{query}/k0"


def _parse_price(text: str) -> float | None:
    cleaned = re.sub(r"[^\d,.]", "", text).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _condition_map(text: str) -> str | None:
    text = text.lower()
    if "neuwertig" in text or "sehr gut" in text:
        return "very_good"
    if "gut" in text:
        return "good"
    if "befriedigend" in text or "akzeptabel" in text:
        return "acceptable"
    return None


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    price_min = filters.get("price_min", "")
    price_max = filters.get("price_max", "")
    encoded_query = query.replace(" ", "-").lower()

    if price_min or price_max:
        url = _BASE_URL.format(
            price_min=price_min or "",
            price_max=price_max or "",
            query=encoded_query,
        )
    else:
        url = _SEARCH_URL.format(query=encoded_query)

    logger.info("Kleinanzeigen scraping: %s", url)
    try:
        html = await fetch_with_playwright(url, wait_selector=".aditem")
    except Exception as e:
        logger.warning("Kleinanzeigen fetch failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for item in soup.select(".aditem")[:40]:
        try:
            link_el = item.select_one(".aditem-main--top a, a.ellipsis")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                href = "https://www.kleinanzeigen.de" + href
            external_id = re.search(r"/s-anzeige/[^/]+/(\d+)", href)
            if not external_id:
                continue
            ext_id = external_id.group(1)
            title_el = item.select_one(".aditem-main--top--left, .text-module-begin")
            title = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
            price_el = item.select_one(".aditem-main--top--right")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(price_text) if "€" in price_text else None
            location_el = item.select_one(".aditem-main--top--left .text-module-end, .aditem-details")
            location = location_el.get_text(strip=True)[:60] if location_el else None
            desc_el = item.select_one(".aditem-main--middle--description")
            raw_text = desc_el.get_text(strip=True)[:500] if desc_el else None
            tag_els = item.select(".simpletag")
            condition = None
            for tag in tag_els:
                mapped = _condition_map(tag.get_text(strip=True))
                if mapped:
                    condition = mapped
                    break
            results.append(listing(
                external_id=ext_id,
                url=href,
                title=title,
                price=price,
                currency="EUR" if price else None,
                location=location,
                condition=condition,
                raw_text=raw_text,
                attributes={"source_query": query},
            ))
        except Exception as e:
            logger.debug("Failed to parse Kleinanzeigen item: %s", e)

    logger.info("Kleinanzeigen: %d listings for %r", len(results), query)
    return results

from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_playwright, listing

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.immobilienscout24.de/Suche/de/{city}/wohnung-mieten"


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    city = filters.get("city", query.lower().replace(" ", "-"))
    url = _SEARCH_URL.format(city=city)
    params = []
    if filters.get("price_max"):
        params.append(f"price=-{filters['price_max']}.0")
    if filters.get("rooms_min"):
        params.append(f"numberofrooms={filters['rooms_min']}.")
    if filters.get("sqm_min"):
        params.append(f"livingspace={filters['sqm_min']}.")
    if params:
        url += "?" + "&".join(params)

    logger.info("Immoscout scraping: %s", url)
    try:
        html = await fetch_with_playwright(url, wait_selector="[data-testid='result-list-entry']")
    except Exception as e:
        logger.warning("Immoscout fetch failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for item in soup.select("[data-testid='result-list-entry']")[:30]:
        try:
            link_el = item.select_one("a[href*='/expose/']")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                href = "https://www.immobilienscout24.de" + href
            ext_id_match = re.search(r"/expose/(\d+)", href)
            if not ext_id_match:
                continue
            ext_id = ext_id_match.group(1)
            title_el = item.select_one("[data-testid='title']")
            title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue
            price_el = item.select_one("[data-testid='price']")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price_match = re.search(r"[\d.,]+", price_text.replace(".", "").replace(",", "."))
            price = float(price_match.group()) if price_match else None
            attrs: dict = {"source_query": query}
            rooms_el = item.select_one("[data-testid='area']")
            if rooms_el:
                rooms_text = rooms_el.get_text(strip=True)
                rooms_match = re.search(r"(\d+(?:[,.]\d+)?)\s*Zi", rooms_text)
                sqm_match = re.search(r"(\d+(?:[,.]\d+)?)\s*m²", rooms_text)
                if rooms_match:
                    attrs["rooms"] = float(rooms_match.group(1).replace(",", "."))
                if sqm_match:
                    attrs["sqm"] = float(sqm_match.group(1).replace(",", "."))
            location_el = item.select_one("[data-testid='address']")
            location = location_el.get_text(strip=True)[:80] if location_el else None
            results.append(listing(
                external_id=ext_id,
                url=href,
                title=title,
                price=price,
                currency="EUR" if price else None,
                location=location,
                attributes=attrs,
            ))
        except Exception as e:
            logger.debug("Failed to parse Immoscout item: %s", e)

    logger.info("Immoscout: %d listings for %r", len(results), query)
    return results

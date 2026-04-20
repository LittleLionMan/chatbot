from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_playwright, listing

logger = logging.getLogger(__name__)


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    city_id = filters.get("city_id", "8")
    url = f"https://www.wg-gesucht.de/wg-zimmer-in-{query.lower().replace(' ', '-')}.{city_id}.0.1.0.html"
    logger.info("WG-Gesucht scraping: %s", url)
    try:
        html = await fetch_with_playwright(url, wait_selector=".wgg_card")
    except Exception as e:
        logger.warning("WG-Gesucht fetch failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for item in soup.select(".wgg_card")[:30]:
        try:
            link_el = item.select_one("a[href*='/wg-zimmer-']")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                href = "https://www.wg-gesucht.de" + href
            ext_id_match = re.search(r"\.(\d+)\.html", href)
            if not ext_id_match:
                continue
            ext_id = ext_id_match.group(1)
            title_el = item.select_one(".truncate_title")
            title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue
            price_el = item.select_one(".detail-size-price-wrapper b")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price_match = re.search(r"(\d+)", price_text)
            price = float(price_match.group(1)) if price_match else None
            location_el = item.select_one(".col-xs-11 span")
            location = location_el.get_text(strip=True)[:80] if location_el else None
            results.append(listing(
                external_id=ext_id,
                url=href,
                title=title,
                price=price,
                currency="EUR" if price else None,
                location=location,
                attributes={"source_query": query},
            ))
        except Exception as e:
            logger.debug("Failed to parse WG-Gesucht item: %s", e)

    logger.info("WG-Gesucht: %d listings for %r", len(results), query)
    return results

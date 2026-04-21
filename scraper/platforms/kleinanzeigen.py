from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_playwright, listing

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.kleinanzeigen.de/s-anzeige:angebote/{query}/k0"


def _build_url(query: str, filters: dict) -> str:
    encoded = query.strip().replace(" ", "+")
    url = _SEARCH_URL.format(query=encoded)
    price_min = filters.get("price_min", "")
    price_max = filters.get("price_max", "")
    if price_min or price_max:
        url = f"https://www.kleinanzeigen.de/s-anzeige:angebote/preis:{price_min}:{price_max}/{encoded}/k0"
    return url


def _parse_price(text: str) -> float | None:
    text = re.sub(r"[^\d.,]", "", text.strip())
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    elif "." in text:
        parts = text.split(".")
        if len(parts) == 2 and len(parts[1]) == 3:
            text = text.replace(".", "")
    try:
        return float(text) if text else None
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
    url = _build_url(query, filters)
    logger.info("Kleinanzeigen scraping: %s", url)
    try:
        html = await fetch_with_playwright(url, wait_selector=".aditem")
    except Exception as e:
        logger.warning("Kleinanzeigen fetch failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for item in soup.select("article.aditem")[:40]:
        try:
            href = item.get("data-href", "")
            if not href:
                link_el = item.select_one("a[href*='/s-anzeige/']")
                href = link_el.get("href", "") if link_el else ""
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.kleinanzeigen.de" + href
            ext_id = item.get("data-adid", "")
            if not ext_id:
                ext_id_match = re.search(r"/(\d+)-\d+-\d+$", href)
                if not ext_id_match:
                    continue
                ext_id = ext_id_match.group(1)
            title_el = item.select_one("h2.text-module-begin, .aditem-main h2")
            title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                title_el = item.select_one("a.ellipsis")
                title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue
            price_el = item.select_one("p.aditem-main--middle--price-shipping--price")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(price_text) if "€" in price_text else None
            location_el = item.select_one(".aditem-main--top--left")
            location = location_el.get_text(strip=True)[:60] if location_el else None
            desc_el = item.select_one(".aditem-main--middle--description, p.aditem-main--middle--description")
            raw_text = desc_el.get_text(strip=True)[:500] if desc_el else None
            tag_els = item.select(".simpletag, .badge")
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

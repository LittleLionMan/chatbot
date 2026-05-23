from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_playwright, listing

logger = logging.getLogger(__name__)

_CITY_SLUGS: dict[str, str] = {
    "münster": "muenster",
    "muenster": "muenster",
    "münchen": "muenchen",
    "münchen": "muenchen",
    "köln": "koeln",
    "düsseldorf": "duesseldorf",
    "nürnberg": "nuernberg",
}

_CATEGORY_PATHS: dict[str, str] = {
    "apartment": "wohnungen-in",
    "room": "wg-zimmer-in",
    "default": "wohnungen-in",
}


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    city_raw = filters.get("city", query.lower().replace(" ", "-"))
    city = _CITY_SLUGS.get(city_raw.lower(), city_raw.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace(" ", "-"))
    city_id = filters.get("city_id", "8")
    path = _CATEGORY_PATHS.get(category, _CATEGORY_PATHS["default"])
    url = f"https://www.wg-gesucht.de/{path}-{city}.{city_id}.2.1.0.html"

    price_max = filters.get("price_max")
    rooms_min = filters.get("rooms_min")
    params = []
    if price_max:
        params.append(f"rent_types[]=2&rent_types[]=0&rent_types[]=4&maxRent={price_max}")
    if rooms_min:
        params.append(f"min_rooms={rooms_min}")
    if params:
        url += "?" + "&".join(params)

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
            link_el = item.select_one("a[href]")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                href = "https://www.wg-gesucht.de" + href
            ext_id_match = re.search(r"\.(\d+)\.html", href)
            if not ext_id_match:
                continue
            ext_id = ext_id_match.group(1)

            title_el = item.select_one(".truncate_title, h3.truncate_title, .card-title")
            title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue

            price_el = item.select_one(".detail-size-price-wrapper b, .rent_type b, .basic_facts_top b")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price_match = re.search(r"(\d+)", price_text.replace(".", ""))
            price = float(price_match.group(1)) if price_match else None

            location_el = item.select_one(".col-xs-11 span, .address_city")
            location = location_el.get_text(strip=True)[:80] if location_el else None

            desc_el = item.select_one(".card-description, .truncate_description")
            raw_text = desc_el.get_text(strip=True)[:500] if desc_el else None

            results.append(listing(
                external_id=ext_id,
                url=href,
                title=title,
                price=price,
                currency="EUR" if price else None,
                location=location,
                raw_text=raw_text,
                attributes={"source_query": query},
            ))
        except Exception as e:
            logger.debug("Failed to parse WG-Gesucht item: %s", e)

    logger.info("WG-Gesucht: %d listings for %r", len(results), query)
    return results

from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_httpx, listing

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ebay.com/sch/i.html"


def _parse_price(text: str) -> float | None:
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _detect_currency(text: str) -> str | None:
    if "€" in text:
        return "EUR"
    if "$" in text:
        return "USD"
    if "£" in text:
        return "GBP"
    return None


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    params = {
        "_nkw": query,
        "_sop": "10",
        "LH_ItemCondition": "3000",
        "_ipg": "60",
    }
    price_min = filters.get("price_min")
    price_max = filters.get("price_max")
    if price_min:
        params["_udlo"] = str(price_min)
    if price_max:
        params["_udhi"] = str(price_max)
    if filters.get("location") == "DE":
        params["_sacat"] = "0"
        params["LH_PrefLoc"] = "3"

    url = _SEARCH_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    logger.info("eBay scraping: %s", url)

    try:
        html = await fetch_with_httpx(url)
    except Exception as e:
        logger.warning("eBay fetch failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for item in soup.select(".s-item")[:40]:
        try:
            link_el = item.select_one(".s-item__link")
            if not link_el:
                continue
            href = link_el.get("href", "")
            ext_id_match = re.search(r"/itm/(\d+)", href)
            if not ext_id_match:
                continue
            ext_id = ext_id_match.group(1)
            title_el = item.select_one(".s-item__title")
            title = title_el.get_text(strip=True) if title_el else ""
            if not title or title.lower() == "shop on ebay":
                continue
            price_el = item.select_one(".s-item__price")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(price_text)
            currency = _detect_currency(price_text)
            location_el = item.select_one(".s-item__location")
            location = location_el.get_text(strip=True).replace("From ", "")[:60] if location_el else None
            condition_el = item.select_one(".SECONDARY_INFO")
            condition_text = condition_el.get_text(strip=True).lower() if condition_el else ""
            condition = None
            if "very good" in condition_text or "sehr gut" in condition_text:
                condition = "very_good"
            elif "good" in condition_text or "gut" in condition_text:
                condition = "good"
            elif "acceptable" in condition_text or "akzeptabel" in condition_text:
                condition = "acceptable"
            seller_el = item.select_one(".s-item__seller-info-text")
            seller_name = seller_el.get_text(strip=True)[:60] if seller_el else None
            results.append(listing(
                external_id=ext_id,
                url=href.split("?")[0],
                title=title,
                price=price,
                currency=currency,
                location=location,
                condition=condition,
                seller_name=seller_name,
                attributes={"source_query": query},
            ))
        except Exception as e:
            logger.debug("Failed to parse eBay item: %s", e)

    logger.info("eBay: %d listings for %r", len(results), query)
    return results

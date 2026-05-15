from __future__ import annotations
import logging
import os
import httpx
from platforms.base import listing

logger = logging.getLogger(__name__)

_FINDING_URL = "https://svcs.ebay.com/services/search/FindingService/v1"
_APP_ID = os.environ.get("EBAY_APP_ID", "")

_CONDITION_MAP = {
    "2000": "very_good",
    "2500": "very_good",
    "3000": "good",
    "4000": "acceptable",
    "5000": "acceptable",
    "6000": "acceptable",
}


def _parse_price(item: dict) -> tuple[float | None, str | None]:
    try:
        price_data = item["sellingStatus"][0]["currentPrice"][0]
        return float(price_data["__value__"]), price_data["@currencyId"]
    except Exception:
        return None, None


def _build_query_string(query: str, filters: dict) -> str:
    from urllib.parse import quote

    parts = [
        f"OPERATION-NAME=findItemsByKeywords",
        f"SERVICE-VERSION=1.0.0",
        f"SECURITY-APPNAME={quote(_APP_ID)}",
        f"RESPONSE-DATA-FORMAT=JSON",
        f"keywords={quote(query)}",
        f"sortOrder=StartTimeNewest",
        f"paginationInput.entriesPerPage=100",
        f"itemFilter(0).name=Condition",
        f"itemFilter(0).value=Used",
        f"itemFilter(1).name=ListingType",
        f"itemFilter(1).value(0)=FixedPrice",
        f"itemFilter(1).value(1)=Auction",
    ]

    idx = 2
    price_min = filters.get("price_min")
    price_max = filters.get("price_max")

    if price_min:
        parts += [
            f"itemFilter({idx}).name=MinPrice",
            f"itemFilter({idx}).value={price_min}",
            f"itemFilter({idx}).paramName=Currency",
            f"itemFilter({idx}).paramValue=EUR",
        ]
        idx += 1

    if price_max:
        parts += [
            f"itemFilter({idx}).name=MaxPrice",
            f"itemFilter({idx}).value={price_max}",
            f"itemFilter({idx}).paramName=Currency",
            f"itemFilter({idx}).paramValue=EUR",
        ]

    return "&".join(parts)


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    if not _APP_ID:
        logger.warning("eBay: EBAY_APP_ID not set, skipping")
        return []

    qs = _build_query_string(query, filters)
    url = f"{_FINDING_URL}?{qs}"
    logger.info("eBay API: %s", url)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("eBay API request failed: %s", e)
        return []

    try:
        search_result = data["findItemsByKeywordsResponse"][0]
        if search_result.get("ack", [None])[0] != "Success":
            logger.warning("eBay API ack not Success: %s", search_result.get("ack"))
            return []
        items = search_result.get("searchResult", [{}])[0].get("item", [])
    except Exception as e:
        logger.warning("eBay API response parse failed: %s", e)
        return []

    results: list[dict] = []
    for item in items:
        try:
            ext_id = item["itemId"][0]
            title = item["title"][0]
            url = item["viewItemURL"][0]
            price, currency = _parse_price(item)

            condition_id = item.get("condition", [{}])[0].get("conditionId", [None])[0]
            condition = _CONDITION_MAP.get(condition_id or "")

            location = item.get("location", [None])[0]
            country = item.get("country", [None])[0]
            if location and country and country not in location:
                location = f"{location}, {country}"

            results.append(listing(
                external_id=ext_id,
                url=url,
                title=title,
                price=price,
                currency=currency,
                location=location[:60] if location else None,
                condition=condition,
                attributes={"source_query": query, "country": country},
            ))
        except Exception as e:
            logger.debug("eBay API: failed to parse item: %s", e)

    logger.info("eBay API: %d listings for %r", len(results), query)
    return results

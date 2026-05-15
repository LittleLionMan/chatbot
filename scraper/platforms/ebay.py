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


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    if not _APP_ID:
        logger.warning("eBay: EBAY_APP_ID not set, skipping")
        return []

    params: dict[str, str] = {
        "OPERATION-NAME": "findItemsByKeywords",
        "SERVICE-VERSION": "1.0.0",
        "SECURITY-APPNAME": _APP_ID,
        "RESPONSE-DATA-FORMAT": "JSON",
        "keywords": query,
        "sortOrder": "StartTimeNewest",
        "paginationInput.entriesPerPage": "100",
        "itemFilter(0).name": "Condition",
        "itemFilter(0).value": "Used",
        "itemFilter(1).name": "ListingType",
        "itemFilter(1).value(0)": "FixedPrice",
        "itemFilter(1).value(1)": "Auction",
    }

    price_min = filters.get("price_min")
    price_max = filters.get("price_max")
    if price_min:
        params["itemFilter(2).name"] = "MinPrice"
        params["itemFilter(2).value"] = str(price_min)
        params["itemFilter(2).paramName"] = "Currency"
        params["itemFilter(2).paramValue"] = "EUR"
    if price_max:
        idx = "3" if price_min else "2"
        params[f"itemFilter({idx}).name"] = "MaxPrice"
        params[f"itemFilter({idx}).value"] = str(price_max)
        params[f"itemFilter({idx}).paramName"] = "Currency"
        params[f"itemFilter({idx}).paramValue"] = "EUR"

    logger.info("eBay API: query=%r", query)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_FINDING_URL, params=params)
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

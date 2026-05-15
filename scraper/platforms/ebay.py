from __future__ import annotations
import base64
import logging
import os
import time
import httpx
from platforms.base import listing

logger = logging.getLogger(__name__)

_APP_ID = os.environ.get("EBAY_APP_ID", "")
_CERT_ID = os.environ.get("EBAY_CERT_ID", "")
_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
_BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
_SCOPE = "https://api.ebay.com/oauth/api_scope"

_token_cache: dict = {"token": None, "expires_at": 0.0}

_CONDITION_MAP = {
    "USED_EXCELLENT": "very_good",
    "USED_VERY_GOOD": "very_good",
    "USED_GOOD": "good",
    "USED_ACCEPTABLE": "acceptable",
    "FOR_PARTS_OR_NOT_WORKING": "acceptable",
}


async def _get_token() -> str | None:
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    if not _APP_ID or not _CERT_ID:
        logger.warning("eBay: EBAY_APP_ID or EBAY_CERT_ID not set")
        return None

    credentials = base64.b64encode(f"{_APP_ID}:{_CERT_ID}".encode()).decode()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _TOKEN_URL,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                content=f"grant_type=client_credentials&scope={_SCOPE}",
            )
            resp.raise_for_status()
            data = resp.json()
            token = data["access_token"]
            expires_in = int(data.get("expires_in", 7200))
            _token_cache["token"] = token
            _token_cache["expires_at"] = time.time() + expires_in
            logger.info("eBay: token acquired, expires in %ds", expires_in)
            return token
    except Exception as e:
        logger.warning("eBay: token request failed: %s", e)
        return None


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    token = await _get_token()
    if not token:
        return []

    params: dict[str, str] = {
        "q": query,
        "filter": "conditions:{USED}",
        "sort": "newlyListed",
        "limit": "100",
        "fieldgroups": "MATCHING_ITEMS",
    }

    price_min = filters.get("price_min")
    price_max = filters.get("price_max")
    if price_min and price_max:
        params["filter"] += f",price:[{price_min}..{price_max}],priceCurrency:EUR"
    elif price_min:
        params["filter"] += f",price:[{price_min}..],priceCurrency:EUR"
    elif price_max:
        params["filter"] += f",price:[..{price_max}],priceCurrency:EUR"

    logger.info("eBay Browse API: query=%r", query)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                _BROWSE_URL,
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("eBay Browse API request failed: %s", e)
        return []

    items = data.get("itemSummaries", [])
    results: list[dict] = []

    for item in items:
        try:
            ext_id = item["itemId"].split("|")[-1]
            title = item.get("title", "")
            url = item.get("itemWebUrl", "")

            price = None
            currency = None
            price_data = item.get("price", {})
            if price_data:
                try:
                    price = float(price_data.get("value", 0))
                    currency = price_data.get("currency")
                except (ValueError, TypeError):
                    pass

            condition_id = item.get("conditionId", "")
            condition = _CONDITION_MAP.get(condition_id)

            location_data = item.get("itemLocation", {})
            location_parts = [
                location_data.get("city"),
                location_data.get("stateOrProvince"),
                location_data.get("country"),
            ]
            location = ", ".join(p for p in location_parts if p) or None
            country = location_data.get("country")

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
            logger.debug("eBay Browse API: failed to parse item: %s", e)

    logger.info("eBay Browse API: %d listings for %r", len(results), query)
    return results

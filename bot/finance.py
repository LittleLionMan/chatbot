from __future__ import annotations
import logging
import httpx

logger = logging.getLogger(__name__)

_FINANCE_BASE_URL = "http://finance:8003"


async def get_quote_summary(ticker: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{_FINANCE_BASE_URL}/quote/{ticker}/summary")
            if resp.status_code == 404:
                logger.warning("Finance service: ticker %s not found", ticker)
                return ""
            resp.raise_for_status()
            data = resp.json()
            summary = data.get("summary", "")
            logger.info("Finance quote fetched for %s (%d chars)", ticker, len(summary))
            return summary
    except httpx.ConnectError:
        logger.warning("Finance service not reachable")
        return ""
    except Exception as e:
        logger.warning("Finance service error for %s: %s", ticker, e)
        return ""


async def get_quote_json(ticker: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{_FINANCE_BASE_URL}/quote/{ticker}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Finance service JSON error for %s: %s", ticker, e)
        return None


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{_FINANCE_BASE_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False

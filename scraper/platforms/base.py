from __future__ import annotations
import asyncio
import logging
import random
from typing import Protocol

import httpx
from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
]

_browser: Browser | None = None
_playwright_instance = None


async def get_browser() -> Browser:
    global _browser, _playwright_instance
    if _browser is None or not _browser.is_connected():
        _playwright_instance = await async_playwright().start()
        _browser = await _playwright_instance.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
    return _browser


async def new_context() -> BrowserContext:
    browser = await get_browser()
    context = await browser.new_context(
        user_agent=random.choice(_USER_AGENTS),
        viewport={"width": random.randint(1280, 1920), "height": random.randint(800, 1080)},
        locale="de-DE",
        timezone_id="Europe/Berlin",
        java_script_enabled=True,
    )
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
    """)
    return context


async def fetch_with_playwright(url: str, wait_selector: str | None = None) -> str:
    context = await new_context()
    try:
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        if wait_selector:
            try:
                await page.wait_for_selector(wait_selector, timeout=8000)
            except Exception:
                pass
        await asyncio.sleep(random.uniform(1.5, 3.5))
        return await page.content()
    finally:
        await context.close()


async def fetch_with_httpx(url: str, params: dict | None = None) -> str:
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    }
    await asyncio.sleep(random.uniform(2.0, 6.0))
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.text


class Listing(Protocol):
    platform: str
    category: str
    external_id: str
    url: str
    title: str
    price: float | None
    currency: str | None
    location: str | None
    condition: str | None
    seller_name: str | None
    seller_rating: float | None
    attributes: dict
    raw_text: str | None


def listing(
    external_id: str,
    url: str,
    title: str,
    price: float | None = None,
    currency: str | None = None,
    location: str | None = None,
    condition: str | None = None,
    seller_name: str | None = None,
    seller_rating: float | None = None,
    attributes: dict | None = None,
    raw_text: str | None = None,
) -> dict:
    return {
        "external_id": external_id,
        "url": url,
        "title": title,
        "price": price,
        "currency": currency,
        "location": location,
        "condition": condition,
        "seller_name": seller_name,
        "seller_rating": seller_rating,
        "attributes": attributes or {},
        "raw_text": raw_text,
    }

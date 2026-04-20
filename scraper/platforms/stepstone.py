from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_playwright, listing

logger = logging.getLogger(__name__)


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    location = filters.get("location", "deutschland")
    url = f"https://www.stepstone.de/jobs/{query.lower().replace(' ', '-')}/in-{location.lower().replace(' ', '-')}"
    logger.info("StepStone scraping: %s", url)
    try:
        html = await fetch_with_playwright(url, wait_selector="[data-testid='job-item']")
    except Exception as e:
        logger.warning("StepStone fetch failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for item in soup.select("[data-testid='job-item']")[:30]:
        try:
            link_el = item.select_one("a[href]")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                href = "https://www.stepstone.de" + href
            ext_id_match = re.search(r"/(\d+)-", href)
            if not ext_id_match:
                continue
            ext_id = ext_id_match.group(1)
            title_el = item.select_one("[data-testid='job-title']")
            title = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
            if not title:
                continue
            company_el = item.select_one("[data-testid='company-name']")
            seller_name = company_el.get_text(strip=True)[:80] if company_el else None
            location_el = item.select_one("[data-testid='job-location']")
            location_text = location_el.get_text(strip=True)[:80] if location_el else None
            results.append(listing(
                external_id=ext_id,
                url=href,
                title=title,
                seller_name=seller_name,
                location=location_text,
                attributes={"source_query": query},
            ))
        except Exception as e:
            logger.debug("Failed to parse StepStone item: %s", e)

    logger.info("StepStone: %d listings for %r", len(results), query)
    return results

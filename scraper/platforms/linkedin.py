from __future__ import annotations
import logging
import re
from bs4 import BeautifulSoup
from platforms.base import fetch_with_httpx, listing

logger = logging.getLogger(__name__)


async def scrape(query: str, category: str, filters: dict) -> list[dict]:
    location = filters.get("location", "Germany")
    url = "https://www.linkedin.com/jobs/search/"
    params = {
        "keywords": query,
        "location": location,
        "sortBy": "DD",
        "f_TPR": "r86400",
    }
    logger.info("LinkedIn scraping: %s query=%r", url, query)
    try:
        html = await fetch_with_httpx(url, params=params)
    except Exception as e:
        logger.warning("LinkedIn fetch failed: %s", e)
        return []

    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for item in soup.select(".job-search-card")[:30]:
        try:
            link_el = item.select_one("a.job-search-card__title-link")
            if not link_el:
                continue
            href = link_el.get("href", "").split("?")[0]
            ext_id_match = re.search(r"/jobs/view/(\d+)", href)
            if not ext_id_match:
                continue
            ext_id = ext_id_match.group(1)
            title = link_el.get_text(strip=True)
            if not title:
                continue
            company_el = item.select_one(".job-search-card__company-name")
            seller_name = company_el.get_text(strip=True)[:80] if company_el else None
            location_el = item.select_one(".job-search-card__location")
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
            logger.debug("Failed to parse LinkedIn item: %s", e)

    logger.info("LinkedIn: %d listings for %r", len(results), query)
    return results

from __future__ import annotations
from typing import Callable, Awaitable

from .kleinanzeigen import scrape as scrape_kleinanzeigen
from .ebay import scrape as scrape_ebay
from .reddit import scrape as scrape_reddit
from .immoscout import scrape as scrape_immoscout
from .wggesucht import scrape as scrape_wggesucht
from .stepstone import scrape as scrape_stepstone
from .linkedin import scrape as scrape_linkedin

ScraperFn = Callable[..., Awaitable[list[dict]]]

SCRAPERS: dict[str, ScraperFn] = {
    "kleinanzeigen": scrape_kleinanzeigen,
    "ebay": scrape_ebay,
    "reddit": scrape_reddit,
    "immoscout": scrape_immoscout,
    "wggesucht": scrape_wggesucht,
    "stepstone": scrape_stepstone,
    "linkedin": scrape_linkedin,
}

SUPPORTED_PLATFORMS = list(SCRAPERS.keys())

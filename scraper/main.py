from __future__ import annotations
import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone

import asyncpg

from platforms import SCRAPERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", "")

_MIN_POLL_INTERVAL = 900


async def _get_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(DB_URL, min_size=1, max_size=3)


async def _get_due_configs(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, platform, category, query, filters, target_agent, poll_interval_seconds, last_scraped_at
        FROM scraper_configs
        WHERE is_active = TRUE
          AND (last_scraped_at IS NULL OR last_scraped_at + (poll_interval_seconds * INTERVAL '1 second') <= NOW())
        ORDER BY last_scraped_at ASC NULLS FIRST
        """
    )
    return [dict(r) for r in rows]


async def _upsert_listing(pool: asyncpg.Pool, listing: dict) -> tuple[bool, int]:
    row = await pool.fetchrow(
        """
        INSERT INTO listings (platform, category, external_id, url, title, price, currency,
                              location, condition, seller_name, seller_rating, attributes, raw_text)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (platform, external_id) DO UPDATE
            SET last_seen_at = NOW(),
                price = EXCLUDED.price,
                title = EXCLUDED.title
        RETURNING id, (xmax = 0) AS is_new
        """,
        listing["platform"],
        listing["category"],
        listing["external_id"],
        listing["url"],
        listing["title"],
        listing.get("price"),
        listing.get("currency"),
        listing.get("location"),
        listing.get("condition"),
        listing.get("seller_name"),
        listing.get("seller_rating"),
        json.dumps(listing.get("attributes", {})),
        listing.get("raw_text"),
    )
    return bool(row["is_new"]), row["id"]


async def _enqueue_trigger(pool: asyncpg.Pool, target_agent: str, payload: dict) -> None:
    await pool.execute(
        """
        INSERT INTO agent_trigger_queue (source_agent_id, target_agent_name, payload, scheduled_for)
        VALUES (NULL, $1, $2, NOW())
        """,
        target_agent, json.dumps(payload),
    )
    logger.info("Trigger → %s: [%s] %s", target_agent, payload.get("title", "")[:40], payload.get("url", ""))


async def _mark_scraped(pool: asyncpg.Pool, config_id: int) -> None:
    await pool.execute(
        "UPDATE scraper_configs SET last_scraped_at = NOW() WHERE id = $1",
        config_id,
    )


async def _run_config(pool: asyncpg.Pool, config: dict) -> None:
    platform: str = config["platform"]
    category: str = config["category"]
    query: str = config["query"]
    filters: dict = config["filters"] if isinstance(config["filters"], dict) else json.loads(config["filters"] or "{}")
    target_agent: str = config["target_agent"]
    config_id: int = config["id"]

    scraper = SCRAPERS.get(platform)
    if scraper is None:
        logger.warning("No scraper for platform: %s", platform)
        return

    logger.info("Scraping %s — query=%r category=%s", platform, query, category)
    try:
        listings = await scraper(query=query, category=category, filters=filters)
    except Exception as e:
        logger.error("Scraper %s failed for query %r: %s", platform, query, e)
        await _mark_scraped(pool, config_id)
        return

    new_count = 0
    for listing in listings:
        listing["platform"] = platform
        listing["category"] = category
        try:
            is_new, listing_id = await _upsert_listing(pool, listing)
            if is_new:
                new_count += 1
                payload = {
                    "listing_id": listing_id,
                    "platform": platform,
                    "category": category,
                    "title": listing["title"],
                    "price": listing.get("price"),
                    "currency": listing.get("currency"),
                    "url": listing["url"],
                    "location": listing.get("location"),
                    "condition": listing.get("condition"),
                    "seller_rating": listing.get("seller_rating"),
                    "attributes": listing.get("attributes", {}),
                    "raw_text": (listing.get("raw_text") or "")[:2000],
                }
                await _enqueue_trigger(pool, target_agent, payload)
        except Exception as e:
            logger.error("Failed to upsert listing %r: %s", listing.get("url"), e)

    await _mark_scraped(pool, config_id)
    logger.info("Scraper %s done: %d listings, %d new → triggered %s", platform, len(listings), new_count, target_agent)
    await asyncio.sleep(random.uniform(3.0, 8.0))


async def _run_cycle(pool: asyncpg.Pool) -> None:
    configs = await _get_due_configs(pool)
    if not configs:
        return
    logger.info("%d scraper config(s) due", len(configs))
    for config in configs:
        await _run_config(pool, config)


async def main() -> None:
    logger.info("Scraper service starting")
    pool = await _get_pool()
    while True:
        try:
            await _run_cycle(pool)
        except Exception as e:
            logger.error("Scraper cycle failed: %s", e)
        await asyncio.sleep(_MIN_POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())

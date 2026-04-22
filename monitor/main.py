from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import asyncpg
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", "")
ARTICLE_FETCH_TIMEOUT = 8.0
ARTICLE_MAX_CHARS = 8000


def _fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_pub_date(pub: str) -> datetime | None:
    if not pub:
        return None
    try:
        return parsedate_to_datetime(pub).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_since_date(since: str) -> datetime | None:
    if not since:
        return None
    try:
        return datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _matches_keywords(title: str, text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    combined = (title + " " + text).lower()
    return any(kw.lower() in combined for kw in keywords)


async def _fetch_article_text(client: httpx.AsyncClient, url: str) -> str:
    import re
    try:
        resp = await client.get(url, timeout=ARTICLE_FETCH_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<aside[^>]*>.*?</aside>", "", html, flags=re.DOTALL | re.IGNORECASE)
        for pattern in [
            r"<article[^>]*>(.*?)</article>",
            r'<main[^>]*>(.*?)</main>',
            r'<div[^>]*role=["\']main["\'][^>]*>(.*?)</div>',
            r'<div[^>]*class=["\'][^"\']*(?:article|content|story|post|body)[^"\']*["\'][^>]*>(.*?)</div>',
        ]:
            match = re.search(pattern, html, flags=re.DOTALL | re.IGNORECASE)
            if match:
                html = match.group(1)
                break
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:ARTICLE_MAX_CHARS]
    except Exception as e:
        logger.debug("Article fetch failed for %s: %s", url[:80], e)
        return ""


async def _get_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(DB_URL, min_size=1, max_size=3)


async def _get_active_configs(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch("SELECT * FROM monitor_configs WHERE is_active = TRUE ORDER BY id")
    return [dict(r) for r in rows]


async def _resolve_watchlist(pool: asyncpg.Pool, config: dict) -> dict[str, tuple[str, str]]:
    row = await pool.fetchrow(
        """
        SELECT as2.value FROM agents a
        JOIN agent_state as2 ON as2.agent_id = a.id AND as2.key = $1
        WHERE LOWER(a.name) = LOWER($2) AND a.is_active = TRUE LIMIT 1
        """,
        config["source_state_key"], config["source_agent"],
    )
    if not row or not row["value"]:
        return {}

    raw = row["value"]
    result: dict[str, tuple[str, str]] = {}
    fmt = config["source_format"]

    if fmt == "pipe_delimited_overview":
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                result[parts[0]] = (parts[1], parts[5])
    elif fmt == "comma_list":
        for key in raw.split(","):
            key = key.strip()
            if key:
                result[key] = (key, "")
    elif fmt == "pipe_name_map":
        for line in raw.strip().splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                result[parts[0].strip()] = (parts[1].strip(), "")

    return result


async def _get_seen(pool: asyncpg.Pool, config_id: int) -> set[str]:
    rows = await pool.fetch("SELECT fingerprint FROM monitor_seen WHERE config_id = $1", config_id)
    return {r["fingerprint"] for r in rows}


async def _mark_seen(pool: asyncpg.Pool, config_id: int, fingerprint: str) -> None:
    await pool.execute(
        "INSERT INTO monitor_seen (config_id, fingerprint) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        config_id, fingerprint,
    )


async def _enqueue_trigger(pool: asyncpg.Pool, target_agent: str, payload: dict) -> None:
    await pool.execute(
        """
        INSERT INTO agent_trigger_queue (source_agent_id, target_agent_name, payload, scheduled_for)
        VALUES (NULL, $1, $2, NOW())
        """,
        target_agent, json.dumps(payload),
    )
    logger.info("Trigger → %s: [%s] %s", target_agent, payload.get("key", ""), payload.get("reason", "")[:60])


async def _fetch_feed(client: httpx.AsyncClient, url: str) -> list[dict]:
    try:
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items: list[dict] = []
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            pub = item.findtext("pubDate", "").strip()
            source_el = item.find("source")
            source = source_el.text.strip() if source_el is not None and source_el.text else ""
            if title and link:
                items.append({"title": title, "url": link, "published": pub, "source": source})
        return items
    except Exception as e:
        logger.debug("Feed fetch failed: %s", e)
        return []


async def _process_feed_items(
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    config: dict,
    feed_url: str,
    item_key: str,
    item_name: str,
    item_date: str,
    seen: set[str],
    keywords: list[str],
) -> None:
    config_id: int = config["id"]
    target_agent: str = config["target_agent"]
    extra_raw = config.get("extra_config") or {}
    extra: dict = extra_raw if isinstance(extra_raw, dict) else {}

    since_dt = _parse_since_date(item_date)
    articles = await _fetch_feed(client, feed_url)

    for article in articles:
        if not isinstance(article, dict):
            continue

        pub_dt = _parse_pub_date(article["published"])
        if since_dt and pub_dt and pub_dt < since_dt:
            fp = _fingerprint(article["url"])
            seen.add(fp)
            await _mark_seen(pool, config_id, fp)
            continue

        fp = _fingerprint(article["url"])
        if fp in seen:
            continue

        if not _matches_keywords(article["title"], "", keywords):
            seen.add(fp)
            await _mark_seen(pool, config_id, fp)
            logger.debug("Skipping non-matching article: %s", article["title"][:60])
            continue

        seen.add(fp)
        await _mark_seen(pool, config_id, fp)

        article_text = await _fetch_article_text(client, article["url"])

        if article_text and not _matches_keywords(article["title"], article_text, keywords):
            logger.debug("Skipping after full-text check: %s", article["title"][:60])
            continue

        payload: dict = {
            "key": item_key,
            "name": item_name,
            "reason": f"{article['title']} ({article['source']})",
            "article_text": article_text,
            "article_url": article["url"],
            "published": article["published"],
            "since_date": item_date,
        }
        payload.update(extra.get("payload_extra", {}))
        await _enqueue_trigger(pool, target_agent, payload)


async def _run_rss_config(pool: asyncpg.Pool, config: dict) -> None:
    config_id: int = config["id"]
    source: str = config.get("source", "agent")
    feed_templates_raw = config["feed_templates"]
    feed_templates: list[str] = list(feed_templates_raw) if not isinstance(feed_templates_raw, str) else [feed_templates_raw]
    keywords: list[str] = list(config.get("keywords") or [])

    seen = await _get_seen(pool, config_id)

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (compatible; BobMonitor/1.0)"}) as client:

        if source == "static":
            logger.info("Config %d (static): %d feeds, %d keywords, %d seen",
                        config_id, len(feed_templates), len(keywords), len(seen))
            for feed_url in feed_templates:
                await _process_feed_items(
                    pool, client, config,
                    feed_url=feed_url,
                    item_key="static",
                    item_name="static",
                    item_date="",
                    seen=seen,
                    keywords=keywords,
                )
                await asyncio.sleep(1)

        else:
            watchlist = await _resolve_watchlist(pool, config)
            if not watchlist:
                logger.info("Config %d: empty watchlist", config_id)
                return

            logger.info("Config %d (agent): %d items, %d keywords, %d seen",
                        config_id, len(watchlist), len(keywords), len(seen))

            for item_key, (item_name, item_date) in watchlist.items():
                query = f"{item_key} {item_name}".strip().replace(" ", "+")
                for template in feed_templates:
                    feed_url = template.format(query=query, key=item_key, name=item_name)
                    await _process_feed_items(
                        pool, client, config,
                        feed_url=feed_url,
                        item_key=item_key,
                        item_name=item_name,
                        item_date=item_date,
                        seen=seen,
                        keywords=keywords,
                    )
                await asyncio.sleep(1)


async def _run_cycle(pool: asyncpg.Pool) -> None:
    configs = await _get_active_configs(pool)
    if not configs:
        logger.info("No active monitor configs")
        return
    for config in configs:
        try:
            if config["monitor_type"] == "rss":
                await _run_rss_config(pool, config)
            else:
                logger.warning("Unknown monitor_type: %s", config["monitor_type"])
        except Exception as e:
            import traceback
            logger.error("Config %d failed: %s\n%s", config["id"], e, traceback.format_exc())


async def main() -> None:
    logger.info("Monitor service starting")
    pool = await _get_pool()
    while True:
        configs = await _get_active_configs(pool)
        interval = min((c["poll_interval_seconds"] for c in configs), default=900)
        try:
            await _run_cycle(pool)
        except Exception as e:
            logger.error("Cycle failed: %s", e)
        logger.info("Next cycle in %ds", interval)
        await asyncio.sleep(interval)


if __name__ == "__main__":
    asyncio.run(main())

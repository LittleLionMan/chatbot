from __future__ import annotations
import json
import asyncpg
from datetime import datetime
from bot import config


async def get_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        database=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
    )


async def upsert_user(pool: asyncpg.Pool, telegram_id: int, username: str | None, first_name: str | None, last_name: str | None) -> None:
    await pool.execute(
        """
        INSERT INTO users (telegram_id, username, first_name, last_name, timezone, last_seen_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (telegram_id) DO UPDATE
        SET username = EXCLUDED.username,
            first_name = EXCLUDED.first_name,
            last_name = EXCLUDED.last_name,
            last_seen_at = NOW()
        """,
        telegram_id, username, first_name, last_name, config.BOT_DEFAULT_TIMEZONE,
    )


async def get_user_timezone(pool: asyncpg.Pool, user_id: int) -> str:
    row = await pool.fetchrow("SELECT timezone FROM users WHERE telegram_id = $1", user_id)
    if not row:
        return config.BOT_DEFAULT_TIMEZONE
    return row["timezone"] or config.BOT_DEFAULT_TIMEZONE


async def set_user_timezone(pool: asyncpg.Pool, user_id: int, timezone: str) -> None:
    await pool.execute(
        "UPDATE users SET timezone = $1 WHERE telegram_id = $2",
        timezone, user_id,
    )


async def upsert_group(pool: asyncpg.Pool, telegram_id: int, title: str | None) -> None:
    await pool.execute(
        """
        INSERT INTO groups (telegram_id, title)
        VALUES ($1, $2)
        ON CONFLICT (telegram_id) DO UPDATE SET title = EXCLUDED.title
        """,
        telegram_id, title,
    )


async def save_message(pool: asyncpg.Pool, chat_id: int, user_id: int | None, role: str, content: str) -> None:
    await pool.execute(
        "INSERT INTO messages (chat_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
        chat_id, user_id, role, content,
    )


async def get_recent_messages(pool: asyncpg.Pool, chat_id: int, limit: int = 20) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT role, content, user_id, created_at
        FROM messages
        WHERE chat_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        chat_id, limit,
    )
    return [dict(r) for r in reversed(rows)]


async def add_memory(pool: asyncpg.Pool, subject_type: str, subject_id: int, content: str) -> None:
    await pool.execute(
        "INSERT INTO memories (subject_type, subject_id, content) VALUES ($1, $2, $3)",
        subject_type, subject_id, content,
    )


async def get_memories(pool: asyncpg.Pool, subject_type: str, subject_id: int, limit: int = 10) -> list[str]:
    rows = await pool.fetch(
        """
        SELECT content FROM memories
        WHERE subject_type = $1 AND subject_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        subject_type, subject_id, limit,
    )
    return [r["content"] for r in rows]


async def get_reflection_memories(
    pool: asyncpg.Pool,
    group_id: int,
    user_id: int,
    limit: int = 8,
) -> list[str]:
    rows = await pool.fetch(
        """
        SELECT content FROM memories
        WHERE subject_type = 'reflection'
          AND subject_id IN ($1, $2)
        ORDER BY created_at DESC
        LIMIT $3
        """,
        group_id, user_id, limit,
    )
    return [r["content"] for r in rows]


async def get_all_group_ids(pool: asyncpg.Pool) -> list[int]:
    rows = await pool.fetch("SELECT telegram_id FROM groups")
    return [r["telegram_id"] for r in rows]


async def touch_session_message(pool: asyncpg.Pool, group_id: int) -> None:
    await pool.execute(
        """
        INSERT INTO session_extractions (group_id, last_extracted_at, last_message_at)
        VALUES ($1, '1970-01-01', NOW())
        ON CONFLICT (group_id) DO UPDATE SET last_message_at = NOW()
        """,
        group_id,
    )


async def get_sessions_due_for_extraction(
    pool: asyncpg.Pool,
    session_timeout_seconds: int,
) -> list[int]:
    rows = await pool.fetch(
        """
        SELECT group_id FROM session_extractions
        WHERE last_message_at > last_extracted_at
          AND EXTRACT(EPOCH FROM (NOW() - last_message_at)) > $1
        """,
        session_timeout_seconds,
    )
    return [r["group_id"] for r in rows]


async def mark_session_extracted(pool: asyncpg.Pool, group_id: int) -> None:
    await pool.execute(
        """
        UPDATE session_extractions SET last_extracted_at = NOW()
        WHERE group_id = $1
        """,
        group_id,
    )


async def get_session_messages(
    pool: asyncpg.Pool,
    group_id: int,
    since: datetime,
) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT role, content, user_id, created_at
        FROM messages
        WHERE chat_id = $1 AND created_at > $2
        ORDER BY created_at ASC
        """,
        group_id, since,
    )
    return [dict(r) for r in rows]


async def get_last_extracted_at(pool: asyncpg.Pool, group_id: int) -> datetime:
    row = await pool.fetchrow(
        "SELECT last_extracted_at FROM session_extractions WHERE group_id = $1",
        group_id,
    )
    if not row:
        return datetime.min
    return row["last_extracted_at"].replace(tzinfo=None)


async def get_cooldown_seconds_since_last_spontaneous(pool: asyncpg.Pool, group_id: int) -> float:
    row = await pool.fetchrow(
        "SELECT last_spontaneous_at FROM group_cooldowns WHERE group_id = $1",
        group_id,
    )
    if not row or row["last_spontaneous_at"] is None:
        return float("inf")
    delta = datetime.utcnow() - row["last_spontaneous_at"].replace(tzinfo=None)
    return delta.total_seconds()


async def update_spontaneous_timestamp(pool: asyncpg.Pool, group_id: int) -> None:
    await pool.execute(
        """
        INSERT INTO group_cooldowns (group_id, last_spontaneous_at)
        VALUES ($1, NOW())
        ON CONFLICT (group_id) DO UPDATE SET last_spontaneous_at = NOW()
        """,
        group_id,
    )


async def create_task(
    pool: asyncpg.Pool,
    user_id: int,
    source_chat_id: int,
    target_chat_id: int,
    description: str,
    schedule: str,
    next_run_at: datetime,
) -> int:
    row = await pool.fetchrow(
        """
        INSERT INTO tasks (user_id, source_chat_id, target_chat_id, description, schedule, next_run_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        user_id, source_chat_id, target_chat_id, description, schedule, next_run_at,
    )
    return row["id"]


async def get_active_tasks_for_user(pool: asyncpg.Pool, user_id: int) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, description, schedule, target_chat_id, next_run_at, last_run_at
        FROM tasks
        WHERE user_id = $1 AND is_active = TRUE
        ORDER BY created_at ASC
        """,
        user_id,
    )
    return [dict(r) for r in rows]


async def get_due_tasks(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, user_id, source_chat_id, target_chat_id, description, schedule
        FROM tasks
        WHERE is_active = TRUE AND next_run_at <= NOW()
        ORDER BY next_run_at ASC
        """,
    )
    return [dict(r) for r in rows]


async def update_task_run(pool: asyncpg.Pool, task_id: int, next_run_at: datetime) -> None:
    await pool.execute(
        """
        UPDATE tasks SET last_run_at = NOW(), next_run_at = $1 WHERE id = $2
        """,
        next_run_at, task_id,
    )


async def deactivate_task(pool: asyncpg.Pool, task_id: int) -> None:
    await pool.execute("UPDATE tasks SET is_active = FALSE WHERE id = $1", task_id)


async def deactivate_tasks_by_description(
    pool: asyncpg.Pool,
    user_id: int,
    task_ids: list[int],
) -> int:
    result = await pool.execute(
        """
        UPDATE tasks SET is_active = FALSE
        WHERE user_id = $1 AND id = ANY($2::int[]) AND is_active = TRUE
        """,
        user_id, task_ids,
    )
    return int(result.split()[-1])


async def create_agent(
    pool: asyncpg.Pool,
    user_id: int,
    target_chat_id: int,
    name: str,
    config_json: dict,
    schedule: str,
    next_run_at: datetime,
) -> int:
    row = await pool.fetchrow(
        """
        INSERT INTO agents (user_id, target_chat_id, name, config, schedule, next_run_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        user_id, target_chat_id, name, json.dumps(config_json), schedule, next_run_at,
    )
    return row["id"]


async def get_active_agents_for_user(pool: asyncpg.Pool, user_id: int) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, name, config, schedule, target_chat_id, next_run_at, last_run_at
        FROM agents
        WHERE user_id = $1 AND is_active = TRUE
        ORDER BY created_at ASC
        """,
        user_id,
    )
    return [dict(r) for r in rows]


async def get_due_agents(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, user_id, target_chat_id, name, config, schedule
        FROM agents
        WHERE is_active = TRUE AND next_run_at <= NOW()
        ORDER BY next_run_at ASC
        """,
    )
    return [dict(r) for r in rows]


async def get_agent_by_name(pool: asyncpg.Pool, user_id: int, name: str) -> dict | None:
    row = await pool.fetchrow(
        """
        SELECT id, name, config, schedule, target_chat_id, next_run_at, last_run_at, is_active
        FROM agents
        WHERE user_id = $1 AND LOWER(name) = LOWER($2)
        """,
        user_id, name,
    )
    return dict(row) if row else None


async def update_agent_run(pool: asyncpg.Pool, agent_id: int, next_run_at: datetime) -> None:
    await pool.execute(
        "UPDATE agents SET last_run_at = NOW(), next_run_at = $1 WHERE id = $2",
        next_run_at, agent_id,
    )


async def deactivate_agent(pool: asyncpg.Pool, agent_id: int) -> None:
    await pool.execute("UPDATE agents SET is_active = FALSE WHERE id = $1", agent_id)


async def rename_agent(pool: asyncpg.Pool, agent_id: int, new_name: str) -> None:
    await pool.execute("UPDATE agents SET name = $1 WHERE id = $2", new_name, agent_id)


async def update_agent_config(pool: asyncpg.Pool, agent_id: int, config_json: dict) -> None:
    await pool.execute(
        "UPDATE agents SET config = $1 WHERE id = $2",
        json.dumps(config_json), agent_id,
    )


async def get_agent_state(pool: asyncpg.Pool, agent_id: int) -> dict[str, str]:
    rows = await pool.fetch(
        "SELECT key, value FROM agent_state WHERE agent_id = $1",
        agent_id,
    )
    return {r["key"]: r["value"] for r in rows}


async def get_agent_state_by_name(pool: asyncpg.Pool, name: str) -> dict[str, str] | None:
    row = await pool.fetchrow(
        "SELECT id FROM agents WHERE LOWER(name) = LOWER($1) AND is_active = TRUE LIMIT 1",
        name,
    )
    if not row:
        return None
    rows = await pool.fetch(
        "SELECT key, value FROM agent_state WHERE agent_id = $1",
        row["id"],
    )
    return {r["key"]: r["value"] for r in rows}


async def set_agent_state(pool: asyncpg.Pool, agent_id: int, state: dict[str, str]) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            for key, value in state.items():
                await conn.execute(
                    """
                    INSERT INTO agent_state (agent_id, key, value, updated_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (agent_id, key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    agent_id, key, value,
                )


async def get_agent_memories(pool: asyncpg.Pool, agent_id: int, limit: int = 20) -> list[str]:
    rows = await pool.fetch(
        """
        SELECT content FROM memories
        WHERE subject_type = 'agent' AND subject_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        agent_id, limit,
    )
    return [r["content"] for r in rows]


async def write_agent_data(
    pool: asyncpg.Pool,
    agent_id: int,
    namespace: str,
    key: str,
    value: str,
) -> None:
    await pool.execute(
        """
        INSERT INTO agent_data (agent_id, namespace, key, value, updated_at)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (agent_id, namespace, key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = NOW()
        """,
        agent_id, namespace, key, value,
    )


async def read_agent_data(
    pool: asyncpg.Pool,
    agent_id: int,
    namespace: str,
    key: str,
) -> str | None:
    row = await pool.fetchrow(
        "SELECT value FROM agent_data WHERE agent_id = $1 AND namespace = $2 AND key = $3",
        agent_id, namespace, key,
    )
    return row["value"] if row else None


async def query_agent_data(
    pool: asyncpg.Pool,
    namespace: str,
    agent_id: int | None = None,
    limit: int = 50,
) -> list[dict]:
    if agent_id is not None:
        rows = await pool.fetch(
            """
            SELECT agent_id, key, value, updated_at
            FROM agent_data
            WHERE namespace = $1 AND agent_id = $2
            ORDER BY updated_at DESC
            LIMIT $3
            """,
            namespace, agent_id, limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT agent_id, key, value, updated_at
            FROM agent_data
            WHERE namespace = $1
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            namespace, limit,
        )
    return [dict(r) for r in rows]


async def get_all_agent_data(
    pool: asyncpg.Pool,
    agent_id: int,
    limit: int = 100,
) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT namespace, key, value, updated_at
        FROM agent_data
        WHERE agent_id = $1
        ORDER BY namespace, key
        LIMIT $2
        """,
        agent_id, limit,
    )
    return [dict(r) for r in rows]


async def enqueue_agent_trigger(
    pool: asyncpg.Pool,
    source_agent_id: int | None,
    target_agent_name: str,
    payload: dict,
    delay_minutes: int = 0,
) -> None:
    await pool.execute(
        """
        INSERT INTO agent_trigger_queue (source_agent_id, target_agent_name, payload, scheduled_for)
        VALUES ($1, $2, $3, NOW() + ($4 * INTERVAL '1 minute'))
        """,
        source_agent_id, target_agent_name, json.dumps(payload), delay_minutes,
    )


async def get_pending_triggers(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, source_agent_id, target_agent_name, payload
        FROM agent_trigger_queue
        WHERE processed_at IS NULL
        ORDER BY created_at ASC
        """,
    )
    return [dict(r) for r in rows]


async def mark_trigger_processed(pool: asyncpg.Pool, trigger_id: int) -> None:
    await pool.execute(
        "UPDATE agent_trigger_queue SET processed_at = NOW() WHERE id = $1",
        trigger_id,
    )


async def get_agent_by_name_global(pool: asyncpg.Pool, name: str) -> dict | None:
    row = await pool.fetchrow(
        """
        SELECT id, user_id, name, config, schedule, target_chat_id, next_run_at, is_active
        FROM agents
        WHERE LOWER(name) = LOWER($1) AND is_active = TRUE
        LIMIT 1
        """,
        name,
    )
    return dict(row) if row else None


async def get_agent_id_by_name(pool: asyncpg.Pool, name: str) -> int | None:
    row = await pool.fetchrow(
        """
        SELECT id FROM agents
        WHERE LOWER(name) = LOWER($1) AND is_active = TRUE
        LIMIT 1
        """,
        name,
    )
    return row["id"] if row else None


async def log_llm_usage(
    pool: asyncpg.Pool,
    caller: str,
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
) -> None:
    await pool.execute(
        "INSERT INTO llm_usage (caller, model, input_tokens, output_tokens) VALUES ($1, $2, $3, $4)",
        caller, model, input_tokens, output_tokens,
    )


async def create_monitor_config(
    pool: asyncpg.Pool,
    monitor_type: str,
    name: str,
    source_agent: str,
    source_state_key: str,
    source_format: str,
    target_agent: str,
    feed_templates: list[str],
    poll_interval_seconds: int = 900,
    extra_config: dict | None = None,
) -> int:
    row = await pool.fetchrow(
        """
        INSERT INTO monitor_configs
            (monitor_type, name, source_agent, source_state_key, source_format,
             target_agent, feed_templates, poll_interval_seconds, extra_config)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """,
        monitor_type, name, source_agent, source_state_key, source_format,
        target_agent, feed_templates, poll_interval_seconds,
        json.dumps(extra_config or {}),
    )
    return row["id"]


async def get_monitor_configs(pool: asyncpg.Pool) -> list[dict]:
    rows = await pool.fetch(
        "SELECT * FROM monitor_configs ORDER BY id"
    )
    return [dict(r) for r in rows]


async def create_scraper_config(
    pool: asyncpg.Pool,
    platform: str,
    category: str,
    query: str,
    target_agent: str,
    filters: dict | None = None,
    poll_interval_seconds: int = 3600,
) -> int:
    row = await pool.fetchrow(
        """
        INSERT INTO scraper_configs
            (platform, category, query, filters, target_agent, poll_interval_seconds)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        platform, category, query,
        json.dumps(filters or {}),
        target_agent, poll_interval_seconds,
    )
    return row["id"]


async def get_scraper_configs(pool: asyncpg.Pool, active_only: bool = True) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, platform, category, query, filters, target_agent,
               poll_interval_seconds, is_active, last_scraped_at, created_at
        FROM scraper_configs
        WHERE ($1 = FALSE OR is_active = TRUE)
        ORDER BY created_at DESC
        """,
        active_only,
    )
    return [dict(r) for r in rows]


async def deactivate_scraper_config(pool: asyncpg.Pool, config_id: int) -> None:
    await pool.execute(
        "UPDATE scraper_configs SET is_active = FALSE WHERE id = $1",
        config_id,
    )


async def get_listings(
    pool: asyncpg.Pool,
    category: str | None = None,
    platform: str | None = None,
    limit: int = 50,
) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, platform, category, external_id, url, title, price, currency,
               location, condition, seller_name, seller_rating, attributes, raw_text,
               first_seen_at, last_seen_at
        FROM listings
        WHERE ($1::text IS NULL OR category = $1)
          AND ($2::text IS NULL OR platform = $2)
        ORDER BY first_seen_at DESC
        LIMIT $3
        """,
        category, platform, limit,
    )
    return [dict(r) for r in rows]


async def get_listing_by_id(pool: asyncpg.Pool, listing_id: int) -> dict | None:
    row = await pool.fetchrow(
        """
        SELECT id, platform, category, external_id, url, title, price, currency,
               location, condition, seller_name, seller_rating, attributes, raw_text,
               first_seen_at, last_seen_at
        FROM listings WHERE id = $1
        """,
        listing_id,
    )
    return dict(row) if row else None


async def delete_listing(pool: asyncpg.Pool, listing_id: int) -> None:
    await pool.execute("DELETE FROM listings WHERE id = $1", listing_id)

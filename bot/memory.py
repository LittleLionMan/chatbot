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
        INSERT INTO users (telegram_id, username, first_name, last_name, last_seen_at)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (telegram_id) DO UPDATE
        SET username = EXCLUDED.username,
            first_name = EXCLUDED.first_name,
            last_name = EXCLUDED.last_name,
            last_seen_at = NOW()
        """,
        telegram_id, username, first_name, last_name,
    )


async def get_user_timezone(pool: asyncpg.Pool, user_id: int) -> str:
    from bot import config
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

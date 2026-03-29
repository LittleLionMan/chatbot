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


async def get_cooldown_seconds_since_last_spontaneous(pool: asyncpg.Pool, group_id: int) -> float:
    row = await pool.fetchrow(
        "SELECT last_spontaneous_at FROM group_cooldowns WHERE group_id = $1",
        group_id,
    )
    if not row:
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

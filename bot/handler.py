import asyncio
import logging
import anthropic
import asyncpg
from telegram import Update
from telegram.ext import ContextTypes
from bot import brain, memory, decider, config, ratelimit, extractor, greeter

logger = logging.getLogger(__name__)


def _display_name(user) -> str:
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.first_name or user.username or str(user.id)


def _build_snippet(history: list[dict], current_user_turn: str, display: str) -> str:
    lines = []
    for entry in history[-6:]:
        prefix = "Bot" if entry["role"] == "assistant" else display
        lines.append(f"{prefix}: {entry['content']}")
    lines.append(f"{display}: {current_user_turn}")
    return "\n".join(lines)


def _last_bot_response(history: list[dict]) -> str | None:
    for entry in reversed(history):
        if entry["role"] == "assistant":
            return entry["content"]
    return None


async def _reply(update: Update, pool: asyncpg.Pool, triggered_by_mention: bool) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat

    if not message or not message.text or not user:
        return

    is_group = chat.type in ("group", "supergroup")
    group_title = chat.title if is_group else None
    group_id = chat.id if is_group else None

    await memory.upsert_user(pool, user.id, user.username, user.first_name, user.last_name)
    if is_group:
        await memory.upsert_group(pool, chat.id, group_title)

    text = message.text.strip()
    display = _display_name(user)

    explicit_reply = await extractor.handle_explicit_memory(pool, user.id, group_id, text)
    if explicit_reply is not None:
        await message.reply_text(explicit_reply)
        return

    user_memories = await memory.get_memories(pool, "user", user.id)
    group_memories = await memory.get_memories(pool, "group", chat.id) if is_group else []
    bot_memories = await memory.get_memories(pool, "bot", chat.id) if is_group else []
    history = await memory.get_recent_messages(pool, chat.id)

    system = brain.build_system_prompt(user_memories, group_memories, bot_memories, display, group_title)
    llm_messages = brain.history_to_llm_messages(history)

    if is_group and not triggered_by_mention:
        user_turn = f"{display}: {text}"
    else:
        user_turn = text

    llm_messages.append({"role": "user", "content": user_turn})

    try:
        response = await brain.chat(system=system, messages=llm_messages)
    except anthropic.RateLimitError:
        if triggered_by_mention:
            await message.reply_text(ratelimit.rate_limit_message())
        return
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        if triggered_by_mention:
            await message.reply_text(ratelimit.rate_limit_message())
        return

    await memory.save_message(pool, chat.id, user.id, "user", user_turn)
    await memory.save_message(pool, chat.id, None, "assistant", response)

    if not triggered_by_mention and is_group:
        await memory.update_spontaneous_timestamp(pool, chat.id)

    await message.reply_text(response)

    snippet = _build_snippet(history, text, display)
    asyncio.create_task(
        extractor.extract_and_store_automatic(pool, user.id, display, snippet)
    )

    if is_group and triggered_by_mention:
        prev_bot = _last_bot_response(history)
        if prev_bot:
            asyncio.create_task(
                extractor.extract_reaction_about_bot(pool, chat.id, prev_bot, text)
            )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    message = update.effective_message
    chat = update.effective_chat

    if not message or not message.text:
        return

    bot_username = context.bot.username
    is_group = chat.type in ("group", "supergroup")
    text = message.text.strip()

    if is_group and greeter.is_greeting(text):
        await message.reply_text(greeter.introduction_text())
        return

    is_mention = bot_username and f"@{bot_username}".lower() in text.lower()
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )

    if is_group:
        if is_mention or is_reply_to_bot:
            if ratelimit.is_rate_limited():
                await message.reply_text(ratelimit.rate_limit_message())
                return
            await _reply(update, pool, triggered_by_mention=True)
        else:
            if ratelimit.is_rate_limited():
                return
            should = await decider.should_respond_spontaneously(
                pool=pool,
                group_id=chat.id,
                message_text=text,
                bot_character=config.BOT_CHARACTER,
            )
            if should:
                await _reply(update, pool, triggered_by_mention=False)
    else:
        if ratelimit.is_rate_limited():
            await message.reply_text(ratelimit.rate_limit_message())
            return
        await _reply(update, pool, triggered_by_mention=True)

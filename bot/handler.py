import asyncio
import logging
import anthropic
import asyncpg
from telegram import Update
from telegram.ext import ContextTypes
from bot import brain, memory, decider, config, ratelimit, extractor, greeter, voice, task_parser, agent_parser, agent_runner

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


async def _send_response(
    update: Update,
    response_text: str,
    use_voice: bool,
    detected_language: str = "de",
) -> None:
    message = update.effective_message
    if not use_voice:
        await message.reply_text(response_text)
        return
    try:
        audio_bytes = await voice.synthesize(response_text, language=detected_language)
        await message.reply_voice(voice=audio_bytes)
    except Exception as e:
        logger.warning("TTS failed, falling back to text: %s", e)
        await message.reply_text(response_text)


async def _reply(
    update: Update,
    pool: asyncpg.Pool,
    triggered_by_mention: bool,
    transcribed_text: str | None = None,
    detected_language: str = "de",
    force_voice: bool = False,
) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat

    if not message or not user:
        return

    logger.warning("_reply called: text=%r triggered_by_mention=%s",
                   (message.text or "")[:60], triggered_by_mention)

    is_group = chat.type in ("group", "supergroup")
    group_title = chat.title if is_group else None
    group_id = chat.id if is_group else None

    await memory.upsert_user(pool, user.id, user.username, user.first_name, user.last_name)
    if is_group:
        await memory.upsert_group(pool, chat.id, group_title)

    text = transcribed_text if transcribed_text is not None else (message.text or "").strip()
    if not text:
        return

    display = _display_name(user)

    explicit_reply = await extractor.handle_explicit_memory(pool, user.id, group_id, text)
    if explicit_reply is not None:
        await message.reply_text(explicit_reply)
        return

    active_agents = await memory.get_active_agents_for_user(pool, user.id)

    if await agent_parser.is_agent_list_request(text):
        if not active_agents:
            await message.reply_text("Du hast keine aktiven Agenten.")
        else:
            lines = [f"{a['name']} — {a['config'].get('instruction', '')[:60]}… ({a['schedule']})" for a in active_agents]
            await message.reply_text("Deine aktiven Agenten:\n" + "\n".join(lines))
        return

    if active_agents and await agent_parser.is_agent_stop_request(text):
        target_agent = await agent_parser.resolve_agent_by_text(text, active_agents)
        if target_agent:
            await memory.deactivate_agent(pool, target_agent["id"])
            await message.reply_text(f"{target_agent['name']} wurde gestoppt.")
        else:
            names = ", ".join(a["name"] for a in active_agents)
            await message.reply_text(f"Ich bin nicht sicher welchen Agenten du meinst. Aktive Agenten: {names}")
        return

    if active_agents and await agent_parser.is_agent_rename_request(text):
        target_agent = await agent_parser.resolve_agent_by_text(text, active_agents)
        new_name = await agent_parser.parse_rename_request(text)
        if target_agent and new_name:
            old_name = target_agent["name"]
            await memory.rename_agent(pool, target_agent["id"], new_name)
            await message.reply_text(f"{old_name} heißt jetzt {new_name}.")
        else:
            await message.reply_text("Ich konnte den Agenten oder den neuen Namen nicht eindeutig erkennen.")
        return

    if active_agents and await agent_parser.is_agent_talk(text):
        target_agent = await agent_parser.resolve_agent_by_text(text, active_agents)
        if target_agent:
            state = await memory.get_agent_state(pool, target_agent["id"])
            agent_memories = await memory.get_agent_memories(pool, target_agent["id"])
            response, new_config = await agent_parser.handle_agent_talk(text, target_agent, state, agent_memories)
            if new_config is not None:
                await memory.update_agent_config(pool, target_agent["id"], new_config)
            await message.reply_text(response)
        else:
            names = ", ".join(a["name"] for a in active_agents)
            await message.reply_text(f"Ich bin nicht sicher welchen Agenten du meinst. Aktive Agenten: {names}")
        return

    if await agent_parser.is_agent_creation(text):
        parsed_agent = await agent_parser.parse_agent_creation(text, user.id, chat.id, pool)
        if parsed_agent:
            suggested = parsed_agent.get("suggested_name")
            name = suggested if suggested else agent_parser._pick_name_for_topic(parsed_agent["config"]["type"])
            await memory.create_agent(
                pool,
                user_id=user.id,
                target_chat_id=parsed_agent["target_chat_id"],
                name=name,
                config_json=parsed_agent["config"],
                schedule=parsed_agent["schedule"],
                next_run_at=parsed_agent["next_run_at"],
            )
            instruction_preview = parsed_agent["config"].get("instruction", "")[:80]
            next_display = parsed_agent["next_run_display"].strftime("%d.%m.%Y %H:%M")
            if parsed_agent.get("wants_name") and not suggested:
                await message.reply_text(
                    f"Agent angelegt: {name} — {instruction_preview}… — ab {next_display}.\n"
                    f"Soll er einen anderen Namen bekommen?"
                )
            else:
                await message.reply_text(
                    f"Agent angelegt: {name} — {instruction_preview}… — ab {next_display}."
                )
        else:
            await message.reply_text("Ich konnte keinen sinnvollen Beobachtungsauftrag erkennen. Versuch's konkreter.")
        return

    if await task_parser.is_task_list_request(text):
        active_tasks = await memory.get_active_tasks_for_user(pool, user.id)
        if not active_tasks:
            await message.reply_text("Du hast keine aktiven Aufgaben.")
        else:
            lines = [f"{t['id']}. {t['description']} — {t['schedule']}" for t in active_tasks]
            await message.reply_text("Deine aktiven Aufgaben:\n" + "\n".join(lines))
        return

    active_tasks = await memory.get_active_tasks_for_user(pool, user.id)
    if await task_parser.is_task_stop_request(text):
        quoted_text = (
            message.reply_to_message.text
            if message.reply_to_message and message.reply_to_message.text
            else None
        )
        stop_context = f"{text}\n\nZitierte Nachricht: {quoted_text}" if quoted_text else text
        stop_ids = await task_parser.parse_stop_request(stop_context, active_tasks)
        if stop_ids:
            count = await memory.deactivate_tasks_by_description(pool, user.id, stop_ids)
            await message.reply_text(f"{count} Aufgabe(n) gestoppt.")
        else:
            if active_tasks:
                lines = [f"{t['id']}. {t['description']}" for t in active_tasks]
                await message.reply_text(
                    "Ich bin nicht sicher welche Aufgabe du meinst. Deine aktiven Aufgaben:\n"
                    + "\n".join(lines)
                )
            else:
                await message.reply_text("Du hast keine aktiven Aufgaben.")
        return

    if await task_parser.is_task_creation(text):
        parsed = await task_parser.parse_task(text, user.id, chat.id, pool)
        if parsed:
            await memory.create_task(
                pool,
                user_id=user.id,
                source_chat_id=chat.id,
                target_chat_id=parsed["target_chat_id"],
                description=parsed["description"],
                schedule=parsed["schedule"],
                next_run_at=parsed["next_run_at"],
            )
            target_note = " (per DM)" if parsed["target_chat_id"] == user.id else ""
            await message.reply_text(
                f"Aufgabe gespeichert{target_note}: {parsed['description']}\n"
                f"Zeitplan: {parsed['schedule']}\n"
                f"Nächste Ausführung: {parsed['next_run_display'].strftime('%d.%m.%Y %H:%M')}"
            )
        else:
            await message.reply_text("Ich konnte keinen gültigen Zeitplan erkennen. Versuch's nochmal konkreter.")
        return

    voice_request = await voice.parse_voice_request(text)
    use_voice = force_voice or voice_request

    user_memories = await memory.get_memories(pool, "user", user.id)
    group_memories = await memory.get_memories(pool, "group", chat.id) if is_group else []
    bot_memories = await memory.get_memories(pool, "bot", chat.id) if is_group else []
    reflection_memories = await memory.get_reflection_memories(pool, chat.id, user.id)
    history = await memory.get_recent_messages(pool, chat.id)

    system = brain.build_system_prompt(
        user_memories,
        group_memories,
        bot_memories,
        reflection_memories,
        display,
        group_title,
        active_agents=active_agents,
    )
    llm_messages = brain.history_to_llm_messages(history)

    if is_group and not triggered_by_mention:
        user_turn = f"{display}: {text}"
    else:
        user_turn = text

    llm_messages.append({"role": "user", "content": user_turn})

    try:
        response = await brain.chat(system=system, messages=llm_messages, use_web_search=True)
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

    if is_group:
        await memory.touch_session_message(pool, chat.id)

    if not triggered_by_mention and is_group:
        await memory.update_spontaneous_timestamp(pool, chat.id)

    await _send_response(update, response, use_voice, detected_language)

    snippet = _build_snippet(history, text, display)
    asyncio.create_task(
        extractor.extract_and_store_automatic(pool, user.id, display, snippet)
    )
    asyncio.create_task(
        extractor.extract_and_store_reflection(pool, chat.id, user.id, snippet)
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("handle_voice triggered")
    pool: asyncpg.Pool = context.bot_data["pool"]
    message = update.effective_message
    chat = update.effective_chat

    if not message or not message.voice:
        return

    bot_username = context.bot.username
    is_group = chat.type in ("group", "supergroup")

    if ratelimit.is_rate_limited():
        await message.reply_text(ratelimit.rate_limit_message())
        return

    try:
        voice_file = await message.voice.get_file()
        audio_bytes = await voice_file.download_as_bytearray()
        transcribed, lang = await voice.transcribe(bytes(audio_bytes))
    except Exception as e:
        logger.error("STT failed: %s", e)
        await message.reply_text("Sprachnachricht konnte nicht transkribiert werden.")
        return

    if not transcribed.strip():
        logger.info("STT returned empty transcript")
        return

    logger.info("STT transcript: %s (lang: %s)", transcribed, lang)

    is_mention = (
        (bot_username and f"@{bot_username}".lower() in transcribed.lower())
        or config.BOT_NAME.lower() in transcribed.lower()
    )
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )

    if is_group:
        if is_mention or is_reply_to_bot:
            await _reply(
                update, pool,
                triggered_by_mention=True,
                transcribed_text=transcribed,
                detected_language=lang,
                force_voice=False,
            )
        else:
            should = await decider.should_respond_spontaneously(
                pool=pool,
                group_id=chat.id,
                message_text=transcribed,
            )
            if should:
                await _reply(
                    update, pool,
                    triggered_by_mention=False,
                    transcribed_text=transcribed,
                    detected_language=lang,
                    force_voice=False,
                )
    else:
        await _reply(
            update, pool,
            triggered_by_mention=True,
            transcribed_text=transcribed,
            detected_language=lang,
            force_voice=False,
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

    is_mention = (
        (bot_username and f"@{bot_username}".lower() in text.lower())
        or config.BOT_NAME.lower() in text.lower()
    )
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
            )
            if should:
                await _reply(update, pool, triggered_by_mention=False)
    else:
        if ratelimit.is_rate_limited():
            await message.reply_text(ratelimit.rate_limit_message())
            return
        await _reply(update, pool, triggered_by_mention=True)

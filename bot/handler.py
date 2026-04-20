from __future__ import annotations
import asyncio
import base64
import logging
import asyncpg
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from bot import brain, memory, decider, config, ratelimit, extractor, greeter, voice, task_parser, agent_parser, agent_runner, intent_classifier, agent_system_parser
from bot.brain import ProviderRateLimitError, ProviderAuthError
from bot.models import CAPABILITY_CHAT, CAPABILITY_MULTIMODAL
from bot.agent_parser import _classify_work_capability, _generate_pipeline
from bot.utils import parse_agent_config

logger = logging.getLogger(__name__)

_pending_agent_systems: dict[int, dict] = {}


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


def _quoted_text(message) -> str | None:
    if message.reply_to_message is None:
        return None
    quoted = message.reply_to_message
    if quoted.text:
        return quoted.text
    if quoted.caption:
        return quoted.caption
    return None


def _agent_keyboard(agent_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Status", callback_data=f"agent:status:{agent_id}"),
            InlineKeyboardButton("Stoppen", callback_data=f"agent:stop:{agent_id}"),
        ],
        [
            InlineKeyboardButton("Umbenennen", callback_data=f"agent:rename:{agent_id}"),
        ],
    ])


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


async def _handle_file_content(
    update: Update,
    pool: asyncpg.Pool,
    file_bytes: bytes,
    media_type: str,
    caption: str | None,
    triggered_by_mention: bool,
    detected_language: str = "de",
) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if not message or not user:
        return
    is_group = chat.type in ("group", "supergroup")
    display = _display_name(user)
    group_title = chat.title if is_group else None
    await memory.upsert_user(pool, user.id, user.username, user.first_name, user.last_name)
    if is_group:
        await memory.upsert_group(pool, chat.id, group_title)
    user_memories = await memory.get_memories(pool, "user", user.id)
    group_memories = await memory.get_memories(pool, "group", chat.id) if is_group else []
    bot_memories = await memory.get_memories(pool, "bot", chat.id) if is_group else []
    reflection_memories = await memory.get_reflection_memories(pool, chat.id, user.id)
    active_agents = await memory.get_active_agents_for_user(pool, user.id)
    history = await memory.get_recent_messages(pool, chat.id)
    system = brain.build_system_prompt(
        user_memories, group_memories, bot_memories, reflection_memories,
        display, group_title, active_agents=active_agents,
    )
    llm_messages = brain.history_to_llm_messages(history)
    user_text = caption if caption else "Was siehst du hier?"
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    content: list[dict] = [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
        {"type": "text", "text": user_text},
    ]
    llm_messages.append({"role": "user", "content": content})
    try:
        response = await brain.chat(
            system=system,
            messages=llm_messages,
            capability=CAPABILITY_MULTIMODAL,
            caller="handler_file",
            pool=pool,
        )
    except (ProviderRateLimitError, ProviderAuthError) as e:
        if triggered_by_mention:
            await message.reply_text(ratelimit.rate_limit_message(e.provider))
        return
    text_turn = f"{display}: [Datei] {user_text}"
    await memory.save_message(pool, chat.id, user.id, "user", text_turn)
    await memory.save_message(pool, chat.id, None, "assistant", response)
    if is_group:
        await memory.touch_session_message(pool, chat.id)
    if not triggered_by_mention and is_group:
        await memory.update_spontaneous_timestamp(pool, chat.id)
    await message.reply_text(response)


async def _handle_pending_confirmation(
    update: Update,
    pool: asyncpg.Pool,
    user_id: int,
    text: str,
) -> bool:
    pending_plan = _pending_agent_systems.get(user_id)
    if not pending_plan:
        return False
    message = update.effective_message
    normalized = text.strip().lower()
    if normalized in ("ja", "yes", "ok", "anlegen", "mach es", "los"):
        for agent_cfg in pending_plan["agents"]:
            await memory.create_agent(
                pool,
                user_id=user_id,
                target_chat_id=agent_cfg["target_chat_id"],
                name=agent_cfg["name"],
                config_json=agent_cfg["config"],
                schedule=agent_cfg["schedule"],
                next_run_at=agent_cfg["next_run_at"],
            )
        names = ", ".join(a["name"] for a in pending_plan["agents"])
        _pending_agent_systems.pop(user_id, None)
        await message.reply_text(f"Angelegt: {names}. Die Agents starten zu ihren geplanten Zeiten.")
        return True
    if normalized in ("nein", "no", "abbruch", "cancel", "stopp"):
        _pending_agent_systems.pop(user_id, None)
        await message.reply_text("Abgebrochen. Kein Agent wurde angelegt.")
        return True
    return False


async def _handle_agent_intent(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    intent: str,
    user_id: int,
    chat_id: int,
    active_agents: list[dict],
) -> None:
    message = update.effective_message

    if intent == "agent_list":
        if not active_agents:
            await message.reply_text("Du hast keine aktiven Agenten.")
        else:
            for agent in active_agents:
                instruction = parse_agent_config(agent["config"]).get("instruction", "")[:80]
                line = f"{agent['name']} — {instruction}… ({agent['schedule']})"
                await message.reply_text(line, reply_markup=_agent_keyboard(agent["id"]))
        return

    if intent == "scraper_create":
        extracted = await intent_classifier.extract_scraper_create_params(text, pool)
        platforms: list[str] = extracted.get("platforms", [])
        target_agent: str = extracted.get("target_agent", "")
        if not platforms or not target_agent:
            await message.reply_text(
                "Ich konnte die Scraper-Parameter nicht erkennen. Beispiel: "
                "'Scrape Kleinanzeigen und eBay nach RTX 4090 und triggere Linus'."
            )
            return
        created_ids: list[int] = []
        for platform in platforms:
            config_id = await memory.create_scraper_config(
                pool,
                platform=platform,
                category=extracted.get("category", "general"),
                query=extracted.get("query", ""),
                target_agent=target_agent,
                filters=extracted.get("filters", {}),
                poll_interval_seconds=extracted.get("poll_interval_seconds", 3600),
            )
            created_ids.append(config_id)
        interval_min = extracted.get("poll_interval_seconds", 3600) // 60
        await message.reply_text(
            f"Scraper eingerichtet für {', '.join(platforms)} — "
            f"sucht nach '{extracted.get('query')}' und triggert {target_agent} "
            f"alle {interval_min} Minuten bei neuen Listings."
        )
        return

    if intent == "agent_system":
        parsed_system = await agent_system_parser.parse_agent_system(text, user_id, chat_id, pool)
        if parsed_system:
            _pending_agent_systems[user_id] = parsed_system
            await message.reply_text(parsed_system["description"])
        else:
            await message.reply_text("Ich konnte kein sinnvolles Agent-System erkennen. Versuch's konkreter.")
        return

    if intent == "agent_create":
        parsed_agent = await agent_parser.parse_agent_creation(text, user_id, chat_id, pool)
        if parsed_agent:
            suggested = parsed_agent.get("suggested_name")
            name = suggested if suggested else agent_parser._pick_name_for_topic(parsed_agent["config"]["type"])
            agent_id = await memory.create_agent(
                pool,
                user_id=user_id,
                target_chat_id=parsed_agent["target_chat_id"],
                name=name,
                config_json=parsed_agent["config"],
                schedule=parsed_agent["schedule"],
                next_run_at=parsed_agent["next_run_at"],
            )
            instruction_preview = parsed_agent["config"].get("instruction", "")[:80]
            next_display = parsed_agent["next_run_display"].strftime("%d.%m.%Y %H:%M")
            await message.reply_text(
                f"Agent angelegt: {name} — {instruction_preview}… — ab {next_display}.\n"
                f"Soll er einen anderen Namen bekommen?",
                reply_markup=_agent_keyboard(agent_id),
            )
            if parsed_agent.get("wants_monitor"):
                monitor_params = await intent_classifier.extract_monitor_create_params(text, pool)
                if monitor_params and monitor_params.get("target_agent"):
                    monitor_id = await memory.create_monitor_config(
                        pool,
                        monitor_type=monitor_params.get("monitor_type", "rss"),
                        name=monitor_params.get("name", f"Monitor für {name}"),
                        source_agent=monitor_params.get("source_agent", name),
                        source_state_key=monitor_params.get("source_state_key", ""),
                        source_format=monitor_params.get("source_format", "comma_list"),
                        target_agent=monitor_params["target_agent"],
                        feed_templates=[
                            "https://news.google.com/rss/search?q={query}+stock&hl=en&gl=US&ceid=US:en",
                            "https://news.google.com/rss/search?q={query}&hl=de&gl=DE&ceid=DE:de",
                        ],
                        poll_interval_seconds=monitor_params.get("poll_interval_seconds", 900),
                    )
                    await message.reply_text(
                        f"RSS-Monitor eingerichtet (ID: {monitor_id}): überwacht relevante Quellen und triggert {name} bei neuen Artikeln."
                    )
            if parsed_agent.get("wants_scraper"):
                scraper_params = await intent_classifier.extract_scraper_create_params(text, pool)
                platforms: list[str] = scraper_params.get("platforms", [])
                if platforms and scraper_params.get("query"):
                    created_ids: list[int] = []
                    for platform in platforms:
                        config_id = await memory.create_scraper_config(
                            pool,
                            platform=platform,
                            category=scraper_params.get("category", "general"),
                            query=scraper_params["query"],
                            target_agent=name,
                            filters=scraper_params.get("filters", {}),
                            poll_interval_seconds=scraper_params.get("poll_interval_seconds", 3600),
                        )
                        created_ids.append(config_id)
                    platform_list = ", ".join(platforms)
                    interval_display = f"{scraper_params.get('poll_interval_seconds', 3600) // 60} Minuten"
                    await message.reply_text(
                        f"Scraper eingerichtet: suche nach '{scraper_params['query']}' auf {platform_list} "
                        f"alle {interval_display} und triggere {name} bei neuen Listings."
                    )
        else:
            await message.reply_text("Ich konnte keinen sinnvollen Beobachtungsauftrag erkennen. Versuch's konkreter.")
        return

    if not active_agents:
        await message.reply_text("Du hast keine aktiven Agenten.")
        return

    if intent == "agent_trigger":
        extracted = await intent_classifier.extract_trigger_payload(text, pool)
        agent_name: str = extracted.get("agent_name", "")
        action: str = extracted.get("action", "run")
        payload: dict = extracted.get("payload", {})
        target_agent = await agent_parser.resolve_agent_by_text(agent_name or text, active_agents)
        if not target_agent:
            names = ", ".join(a["name"] for a in active_agents)
            await message.reply_text(f"Ich bin nicht sicher welchen Agenten du meinst. Aktive Agenten: {names}")
            return
        if action == "stop":
            await memory.deactivate_agent(pool, target_agent["id"])
            await message.reply_text(f"{target_agent['name']} wurde gestoppt.")
        else:
            await memory.enqueue_agent_trigger(pool, None, target_agent["name"], payload)
            payload_desc = f" mit Payload: {payload}" if payload else ""
            await message.reply_text(
                f"{target_agent['name']} wird beim nächsten Scheduler-Lauf ausgeführt{payload_desc}."
            )
        return

    if intent == "agent_talk":
        extracted = await intent_classifier.extract_agent_talk(text, pool)
        agent_name = extracted.get("agent_name", "")
        talk_type: str = extracted.get("talk_type", "query")
        target_agent = await agent_parser.resolve_agent_by_text(agent_name or text, active_agents)
        if not target_agent:
            names = ", ".join(a["name"] for a in active_agents)
            await message.reply_text(f"Ich bin nicht sicher welchen Agenten du meinst. Aktive Agenten: {names}")
            return

        if talk_type == "config_technical":
            current_config = dict(parse_agent_config(target_agent["config"]))
            response_parts: list[str] = []
            set_capability: str | None = extracted.get("set_capability")
            if set_capability:
                current_config["work_capability"] = set_capability
                response_parts.append(f"work_capability auf {set_capability} gesetzt.")
            if extracted.get("reclassify_capability") and not set_capability:
                new_cap = await _classify_work_capability(current_config.get("instruction", ""))
                current_config["work_capability"] = new_cap
                response_parts.append(f"Capability neu klassifiziert: {new_cap}.")
            if extracted.get("regenerate_pipeline"):
                new_pipeline = await _generate_pipeline(
                    current_config.get("instruction", ""),
                    current_config.get("work_capability", "chat"),
                    current_config.get("state_keys", ["last_run_summary"]),
                )
                current_config.pop("pipeline", None)
                current_config.pop("pipeline_template", None)
                current_config.pop("pipeline_after_template", None)
                if new_pipeline:
                    if new_pipeline.get("pipeline"):
                        current_config["pipeline"] = new_pipeline["pipeline"]
                    if new_pipeline.get("pipeline_template"):
                        current_config["pipeline_template"] = new_pipeline["pipeline_template"]
                    if new_pipeline.get("pipeline_after_template"):
                        current_config["pipeline_after_template"] = new_pipeline["pipeline_after_template"]
                    total = len(new_pipeline.get("pipeline", [])) + len(new_pipeline.get("pipeline_after_template", []))
                    has_tmpl = bool(new_pipeline.get("pipeline_template"))
                    response_parts.append(f"Pipeline generiert: {total} feste Steps, Template={'ja' if has_tmpl else 'nein'}.")
                else:
                    response_parts.append("Für diese Instruction wird keine Pipeline benötigt.")
            await memory.update_agent_config(pool, target_agent["id"], current_config)
            reply = f"{target_agent['name']}: " + " ".join(response_parts) if response_parts else f"Keine Änderungen für {target_agent['name']}."
            await message.reply_text(reply)
        else:
            state = await memory.get_agent_state(pool, target_agent["id"])
            agent_memories = await memory.get_agent_memories(pool, target_agent["id"])
            response, new_config, new_name = await agent_parser.handle_agent_talk(
                text, target_agent, state, agent_memories, pool=pool
            )
            if new_config is not None:
                await memory.update_agent_config(pool, target_agent["id"], new_config)
            if new_name is not None:
                await memory.rename_agent(pool, target_agent["id"], new_name)
            await message.reply_text(response)


async def _handle_task_intent(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    intent: str,
    user_id: int,
    chat_id: int,
    active_tasks: list[dict],
    message,
) -> None:
    if intent == "task_list":
        if not active_tasks:
            await message.reply_text("Du hast keine aktiven Aufgaben.")
        else:
            lines = [f"{t['id']}. {t['description']} — {t['schedule']}" for t in active_tasks]
            await message.reply_text("Deine aktiven Aufgaben:\n" + "\n".join(lines))
        return

    if intent == "task_stop":
        quoted = (
            message.reply_to_message.text
            if message.reply_to_message and message.reply_to_message.text
            else None
        )
        stop_context = f"{text}\n\nZitierte Nachricht: {quoted}" if quoted else text
        stop_ids = await task_parser.parse_stop_request(stop_context, active_tasks)
        if stop_ids:
            count = await memory.deactivate_tasks_by_description(pool, user_id, stop_ids)
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

    if intent == "task_create":
        parsed = await task_parser.parse_task(text, user_id, chat_id, pool)
        if parsed:
            await memory.create_task(
                pool,
                user_id=user_id,
                source_chat_id=chat_id,
                target_chat_id=parsed["target_chat_id"],
                description=parsed["description"],
                schedule=parsed["schedule"],
                next_run_at=parsed["next_run_at"],
            )
            target_note = " (per DM)" if parsed["target_chat_id"] == user_id else ""
            await message.reply_text(
                f"Aufgabe gespeichert{target_note}: {parsed['description']}\n"
                f"Zeitplan: {parsed['schedule']}\n"
                f"Nächste Ausführung: {parsed['next_run_display'].strftime('%d.%m.%Y %H:%M')}"
            )
        else:
            await message.reply_text("Ich konnte keinen gültigen Zeitplan erkennen. Versuch's nochmal konkreter.")


async def _handle_scraper_intent(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    user_id: int,
) -> None:
    message = update.effective_message
    extracted = await intent_classifier.extract_scraper_create_params(text, pool)
    if not extracted or not extracted.get("target_agent") or not extracted.get("platforms"):
        await message.reply_text(
            "Ich konnte die Scraper-Parameter nicht erkennen. Beispiel: "
            "'Richte einen Scraper auf Kleinanzeigen und eBay ein der RTX 4090 sucht und Linus triggert'."
        )
        return
    platforms: list[str] = extracted["platforms"]
    category: str = extracted.get("category", "general")
    query: str = extracted.get("query", "")
    filters: dict = extracted.get("filters", {})
    target_agent: str = extracted["target_agent"]
    poll_interval: int = extracted.get("poll_interval_seconds", 3600)
    created_ids: list[int] = []
    for platform in platforms:
        config_id = await memory.create_scraper_config(
            pool,
            platform=platform,
            category=category,
            query=query,
            filters=filters,
            target_agent=target_agent,
            poll_interval_seconds=poll_interval,
        )
        created_ids.append(config_id)
        logger.info("Scraper config created: id=%d platform=%s query=%r target=%s", config_id, platform, query, target_agent)
    interval_display = f"{poll_interval // 60} Minuten" if poll_interval < 3600 else f"{poll_interval // 3600} Stunde(n)"
    platforms_display = ", ".join(platforms)
    await message.reply_text(
        f"Scraper eingerichtet ({len(created_ids)} Konfig(s)): sucht nach '{query}' auf {platforms_display} "
        f"alle {interval_display} und triggert {target_agent} bei neuen Funden."
    )


async def _handle_chat(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    user_id: int,
    chat_id: int,
    is_group: bool,
    triggered_by_mention: bool,
    needs_search: bool,
    wants_voice: bool,
    detected_language: str,
    active_agents: list[dict],
    display: str,
    group_title: str | None,
) -> None:
    message = update.effective_message
    user_memories = await memory.get_memories(pool, "user", user_id)
    group_memories = await memory.get_memories(pool, "group", chat_id) if is_group else []
    bot_memories = await memory.get_memories(pool, "bot", chat_id) if is_group else []
    reflection_memories = await memory.get_reflection_memories(pool, chat_id, user_id)
    history = await memory.get_recent_messages(pool, chat_id)
    system = brain.build_system_prompt(
        user_memories, group_memories, bot_memories, reflection_memories,
        display, group_title, active_agents=active_agents,
    )
    llm_messages = brain.history_to_llm_messages(history)
    quoted = _quoted_text(message)
    if is_group and not triggered_by_mention:
        user_turn = f"{display}: {text}"
    else:
        user_turn = text
    if quoted:
        user_turn = f"[Zitiert: {quoted}]\n{user_turn}"
    llm_messages.append({"role": "user", "content": user_turn})
    if needs_search:
        from bot import search as _search
        if await _search.is_available():
            await message.reply_text("Moment, ich schaue kurz nach…")
    try:
        response = await brain.chat(
            system=system,
            messages=llm_messages,
            use_web_search=needs_search,
            capability=CAPABILITY_CHAT,
            caller="handler",
            pool=pool,
        )
    except (ProviderRateLimitError, ProviderAuthError) as e:
        if triggered_by_mention:
            await message.reply_text(ratelimit.rate_limit_message(e.provider))
        return
    await memory.save_message(pool, chat_id, user_id, "user", user_turn)
    await memory.save_message(pool, chat_id, None, "assistant", response)
    if is_group:
        await memory.touch_session_message(pool, chat_id)
    if not triggered_by_mention and is_group:
        await memory.update_spontaneous_timestamp(pool, chat_id)
    await _send_response(update, response, wants_voice, detected_language)
    snippet = _build_snippet(history, text, display)
    asyncio.create_task(extractor.extract_and_store_automatic(pool, user_id, display, snippet))
    asyncio.create_task(extractor.extract_and_store_reflection(pool, chat_id, user_id, snippet))


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

    if await _handle_pending_confirmation(update, pool, user.id, text):
        return

    active_agents = await memory.get_active_agents_for_user(pool, user.id)
    active_tasks = await memory.get_active_tasks_for_user(pool, user.id)

    classified = await intent_classifier.classify(
        text, pool,
        has_active_agents=bool(active_agents),
        has_active_tasks=bool(active_tasks),
    )
    intent = classified["intent"]
    needs_search = classified["needs_search"]
    wants_voice = force_voice or classified["wants_voice"]

    logger.debug("_reply intent=%s search=%s voice=%s", intent, needs_search, wants_voice)

    if intent in ("agent_system", "agent_create", "agent_trigger", "agent_talk", "agent_list"):
        await _handle_agent_intent(
            update, pool, text, intent, user.id, chat.id, active_agents
        )
        return

    if intent in ("task_create", "task_stop", "task_list"):
        await _handle_task_intent(
            update, pool, text, intent, user.id, chat.id, active_tasks, message
        )
        return

    if intent == "scraper_create":
        await _handle_scraper_intent(update, pool, text, user.id)
        return

    await _handle_chat(
        update, pool, text, user.id, chat.id, is_group, triggered_by_mention,
        needs_search, wants_voice, detected_language, active_agents, display, group_title,
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
    if ratelimit.is_any_limited():
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
            await _reply(update, pool, triggered_by_mention=True,
                         transcribed_text=transcribed, detected_language=lang)
        else:
            should = await decider.should_respond_spontaneously(
                pool=pool, group_id=chat.id, message_text=transcribed,
            )
            if should:
                await _reply(update, pool, triggered_by_mention=False,
                             transcribed_text=transcribed, detected_language=lang)
    else:
        await _reply(update, pool, triggered_by_mention=True,
                     transcribed_text=transcribed, detected_language=lang, force_voice=True)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not message or not message.text or not user:
        return
    bot_username = context.bot.username
    is_group = chat.type in ("group", "supergroup")
    text = message.text.strip()
    pending_rename: int | None = context.user_data.get("awaiting_rename_agent_id")
    if pending_rename is not None:
        del context.user_data["awaiting_rename_agent_id"]
        active_agents = await memory.get_active_agents_for_user(pool, user.id)
        agent = next((a for a in active_agents if a["id"] == pending_rename), None)
        if not agent:
            await message.reply_text("Dieser Agent existiert nicht mehr.")
        else:
            old_name = agent["name"]
            await memory.rename_agent(pool, pending_rename, text)
            await message.reply_text(f"{old_name} heißt jetzt {text}.")
        return
    agent_names_for_mention: list[str] = []
    if is_group:
        agents_for_mention = await memory.get_active_agents_for_user(pool, user.id)
        agent_names_for_mention = [a["name"].lower() for a in agents_for_mention]
    is_mention = (
        (bot_username and f"@{bot_username}".lower() in text.lower())
        or config.BOT_NAME.lower() in text.lower()
        or any(name in text.lower() for name in agent_names_for_mention)
    )
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )
    if is_group:
        if is_mention or is_reply_to_bot:
            if ratelimit.is_any_limited():
                await message.reply_text(ratelimit.rate_limit_message())
                return
            await _reply(update, pool, triggered_by_mention=True)
        else:
            if ratelimit.is_any_limited():
                return
            should = await decider.should_respond_spontaneously(
                pool=pool, group_id=chat.id, message_text=text,
            )
            if should:
                await _reply(update, pool, triggered_by_mention=False)
    else:
        if ratelimit.is_any_limited():
            await message.reply_text(ratelimit.rate_limit_message())
            return
        await _reply(update, pool, triggered_by_mention=True)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not message or not message.photo or not user:
        return
    bot_username = context.bot.username
    is_group = chat.type in ("group", "supergroup")
    caption = (message.caption or "").strip() or None
    is_mention = bool(
        caption and (
            (bot_username and f"@{bot_username}".lower() in caption.lower())
            or config.BOT_NAME.lower() in caption.lower()
        )
    )
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )
    triggered = not is_group or is_mention or is_reply_to_bot
    if is_group and not triggered:
        return
    if ratelimit.is_any_limited():
        if triggered:
            await message.reply_text(ratelimit.rate_limit_message())
        return
    photo = message.photo[-1]
    photo_file = await photo.get_file()
    file_bytes = await photo_file.download_as_bytearray()
    await _handle_file_content(
        update, pool,
        file_bytes=bytes(file_bytes),
        media_type="image/jpeg",
        caption=caption,
        triggered_by_mention=triggered,
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not message or not message.document or not user:
        return
    doc = message.document
    mime = doc.mime_type or ""
    caption = (message.caption or "").strip() or None
    SUPPORTED_MIME: dict[str, str] = {
        "image/jpeg": "image/jpeg",
        "image/png": "image/png",
        "image/gif": "image/gif",
        "image/webp": "image/webp",
        "application/pdf": "application/pdf",
    }
    if mime not in SUPPORTED_MIME:
        await message.reply_text(f"Dieses Dateiformat ({mime or 'unbekannt'}) kann ich noch nicht lesen.")
        return
    bot_username = context.bot.username
    is_group = chat.type in ("group", "supergroup")
    is_mention = bool(
        caption and (
            (bot_username and f"@{bot_username}".lower() in caption.lower())
            or config.BOT_NAME.lower() in caption.lower()
        )
    )
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )
    triggered = not is_group or is_mention or is_reply_to_bot
    if is_group and not triggered:
        return
    if ratelimit.is_any_limited():
        if triggered:
            await message.reply_text(ratelimit.rate_limit_message())
        return
    doc_file = await doc.get_file()
    file_bytes = await doc_file.download_as_bytearray()
    await _handle_file_content(
        update, pool,
        file_bytes=bytes(file_bytes),
        media_type=SUPPORTED_MIME[mime],
        caption=caption,
        triggered_by_mention=triggered,
    )


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    query = update.callback_query
    user = update.effective_user
    if not query or not user:
        return
    data = query.data or ""
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "agent":
        await query.answer()
        return
    action = parts[1]
    try:
        agent_id = int(parts[2])
    except ValueError:
        await query.answer()
        return
    active_agents = await memory.get_active_agents_for_user(pool, user.id)
    agent = next((a for a in active_agents if a["id"] == agent_id), None)
    if not agent:
        await query.answer("Dieser Agent existiert nicht mehr.")
        return
    if action == "stop":
        await memory.deactivate_agent(pool, agent_id)
        await query.answer("Gestoppt.")
        await query.edit_message_text(f"{agent['name']} wurde gestoppt.")
    elif action == "status":
        await query.answer("Einen Moment…")
        state = await memory.get_agent_state(pool, agent_id)
        agent_memories = await memory.get_agent_memories(pool, agent_id)
        status_text, _, _ = await agent_parser.handle_agent_talk(
            "Was ist dein aktueller Status und was hast du bisher beobachtet?",
            agent, state, agent_memories, pool=pool
        )
        await query.message.reply_text(
            f"{agent['name']} — Status:\n\n{status_text}",
            reply_markup=_agent_keyboard(agent_id),
        )
    elif action == "rename":
        await query.answer()
        context.user_data["awaiting_rename_agent_id"] = agent_id
        await query.message.reply_text(
            f"Wie soll {agent['name']} heißen? Schreib einfach den neuen Namen."
        )


async def handle_command_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(greeter.introduction_text())


async def handle_command_agents(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    user = update.effective_user
    if not user:
        return
    active_agents = await memory.get_active_agents_for_user(pool, user.id)
    if not active_agents:
        await update.effective_message.reply_text("Du hast keine aktiven Agenten.")
        return
    for agent in active_agents:
        instruction = parse_agent_config(agent["config"]).get("instruction", "")[:80]
        line = f"{agent['name']} — {instruction}… ({agent['schedule']})"
        await update.effective_message.reply_text(line, reply_markup=_agent_keyboard(agent["id"]))


async def handle_command_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    user = update.effective_user
    if not user:
        return
    active_tasks = await memory.get_active_tasks_for_user(pool, user.id)
    if not active_tasks:
        await update.effective_message.reply_text("Du hast keine aktiven Aufgaben.")
        return
    lines = [f"{t['id']}. {t['description']} — {t['schedule']}" for t in active_tasks]
    await update.effective_message.reply_text("Deine aktiven Aufgaben:\n" + "\n".join(lines))

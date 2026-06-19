from __future__ import annotations

import asyncio
import base64
import logging
import re

import asyncpg
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from bot import (
    agent_context,
    agent_edits,
    agent_parser,
    agent_planner,
    brain,
    config,
    decider,
    extractor,
    greeter,
    intent_classifier,
    memory,
    observer,
    ratelimit,
    task_parser,
    voice,
)
from bot.brain import NoSearchResultsError, ProviderAuthError, ProviderRateLimitError
from bot.models import CAPABILITY_CHAT, CAPABILITY_MULTIMODAL
from bot.utils import parse_agent_config

logger = logging.getLogger(__name__)

_SESSION_PLAN = "plan"
_SESSION_CLARIFICATION = "clarification"
_SESSION_AGENT_TALK = "agent_talk"
_AGENT_TALK_TTL_MINUTES = 30
_pending_rollbacks: dict[int, dict] = {}


def _display_name(user) -> str:
    if user.first_name and user.last_name:
        return f"{user.first_name} {user.last_name}"
    return user.first_name or user.username or str(user.id)


def _quoted_text(message) -> str | None:
    if message.reply_to_message is None:
        return None
    quoted = message.reply_to_message
    return quoted.text or quoted.caption or None


def _extract_forward_context(message) -> str | None:
    origin = getattr(message, "forward_origin", None)
    if origin is None:
        return None
    origin_type = getattr(origin, "type", None)
    if origin_type == "user":
        sender = getattr(origin, "sender_user", None)
        if sender:
            name = (
                f"{sender.first_name or ''} {sender.last_name or ''}".strip()
                or sender.username
            )
            return f"[Weitergeleitet von: {name}]"
    if origin_type == "hidden_user":
        name = getattr(origin, "sender_user_name", None)
        return f"[Weitergeleitet von: {name}]" if name else "[Weitergeleitet]"
    if origin_type == "chat":
        chat = getattr(origin, "sender_chat", None)
        if chat:
            return f"[Weitergeleitet aus: {chat.title or 'Kanal'}]"
    return "[Weitergeleitet]"


def _agent_keyboard(agent_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Status", callback_data=f"agent:status:{agent_id}"
                ),
                InlineKeyboardButton("Stoppen", callback_data=f"agent:stop:{agent_id}"),
            ],
            [
                InlineKeyboardButton(
                    "Umbenennen", callback_data=f"agent:rename:{agent_id}"
                ),
            ],
        ]
    )


def _confirmation_keyboard(confirmation_id: int) -> InlineKeyboardMarkup:
    return agent_edits.confirmation_keyboard(confirmation_id)


async def _send_response(
    message,
    response_text: str,
    use_voice: bool,
    detected_language: str = "de",
) -> None:
    if not use_voice:
        await message.reply_text(response_text)
        return
    try:
        audio_bytes = await voice.synthesize(response_text, language=detected_language)
        await message.reply_voice(voice=audio_bytes)
    except Exception as e:
        logger.warning("TTS failed, falling back to text: %s", e)
        await message.reply_text(response_text)


def _build_snippet(history: list[dict], current_user_turn: str, display: str) -> str:
    lines = []
    for entry in history[-6:]:
        prefix = "Bot" if entry["role"] == "assistant" else display
        lines.append(f"{prefix}: {entry['content']}")
    lines.append(f"{display}: {current_user_turn}")
    return "\n".join(lines)


def _summarize_history_for_membership(history: list[dict], limit: int = 6) -> str:
    lines = []
    for entry in history[-limit:]:
        prefix = "Bob" if entry["role"] == "assistant" else "User"
        content = entry["content"]
        preview = content[:200] + "…" if len(content) > 200 else content
        lines.append(f"{prefix}: {preview}")
    return "\n".join(lines)


async def _handle_pending_confirmation(
    update: Update,
    pool: asyncpg.Pool,
    user_id: int,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None,
) -> bool:
    pending = await memory.get_pending_confirmation(pool, user_id, chat_id)
    if not pending:
        return False

    message = update.effective_message
    edit_payload = pending["payload"]

    if reply_to_message_id:
        notification = await memory.get_agent_notification(
            pool, reply_to_message_id, chat_id
        )
        if notification and notification.get("notification_type") == "adjust_request":
            edit_payload["user_correction"] = text
            confirmation_msg = agent_edits.format_confirmation_message(edit_payload)
            conf_id = await memory.replace_pending_confirmation(
                pool,
                chat_id,
                user_id,
                edit_payload.get("agent_id", 0),
                edit_payload.get("edit_type", ""),
                confirmation_msg,
                edit_payload,
            )
            sent = await message.reply_text(
                confirmation_msg,
                reply_markup=_confirmation_keyboard(conf_id),
            )
            await memory.save_agent_notification(
                pool,
                sent.message_id,
                chat_id,
                edit_payload.get("agent_id", 0),
                "confirmation",
                {"confirmation_id": conf_id},
            )
            return True

    return False


async def _handle_plan_session(
    update: Update,
    pool: asyncpg.Pool,
    user_id: int,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None,
) -> bool:
    session = await memory.get_session(pool, user_id, chat_id, _SESSION_PLAN)
    if not session:
        return False

    payload = session["payload"]
    bot_message_id: int | None = payload.get("bot_message_id")
    if bot_message_id is not None and reply_to_message_id != bot_message_id:
        return False

    message = update.effective_message
    accumulated = payload["accumulated_context"]
    rounds = payload.get("clarification_rounds", 0)
    current_plan = payload.get("plan", {})

    accumulated += f"\n\nUser: {text}"
    new_plan = await agent_planner.plan(accumulated, pool, rounds)

    if new_plan["status"] == "confirmed":
        if current_plan.get("status") != "ready":
            await message.reply_text(
                "Ich habe noch keinen fertigen Plan — beschreib mir erst was ich bauen soll."
            )
            return True

        await memory.clear_session(pool, user_id, chat_id, _SESSION_PLAN)

        prepared = await agent_planner.finalize(
            plan_result=current_plan,
            accumulated_context=accumulated,
            user_id=user_id,
            source_chat_id=chat_id,
            pool=pool,
        )

        if not prepared:
            await message.reply_text(
                "Beim Anlegen ist etwas schiefgelaufen. Versuch's nochmal."
            )
            return True

        expected = len(current_plan.get("agents", []))
        created = 0
        failed: list[str] = []

        for agent_cfg in prepared:
            try:
                await memory.create_agent(
                    pool,
                    user_id=user_id,
                    target_chat_id=agent_cfg["target_chat_id"],
                    name=agent_cfg["name"],
                    config_json=agent_cfg["config"],
                    schedule=agent_cfg["schedule"],
                    next_run_at=agent_cfg["next_run_at"],
                )
                created += 1
            except Exception as e:
                logger.error("failed to create agent %s: %s", agent_cfg["name"], e)
                failed.append(agent_cfg["name"])

        names = ", ".join(a["name"] for a in prepared if a["name"] not in failed)
        created_monitors = await agent_planner.finalize_monitors(current_plan, pool)
        created_scrapers, unavailable_scrapers = await agent_planner.finalize_scrapers(
            current_plan, pool
        )

        reply_parts: list[str] = []
        if failed:
            reply_parts.append(
                f"Angelegt: {names}.\nFehlgeschlagen: {', '.join(failed)} — schau in die Logs."
            )
        elif created < expected:
            reply_parts.append(
                f"Angelegt: {names}. Hinweis: {expected - created} Agent(en) konnten nicht erstellt werden."
            )
        else:
            reply_parts.append(f"Angelegt: {names}.")

        if created_monitors:
            reply_parts.append(
                f"RSS-Monitor(e) eingerichtet: {', '.join(m['name'] for m in created_monitors)}."
            )
        if created_scrapers:
            scraper_summary = ", ".join(
                f"{s['platform']} → {s['target_agent']}" for s in created_scrapers
            )
            reply_parts.append(f"Scraper eingerichtet: {scraper_summary}.")
        if unavailable_scrapers:
            reply_parts.append(
                f"Hinweis: Scraper für {', '.join(s['platform'] for s in unavailable_scrapers)} sind noch nicht verfügbar."
            )

        await message.reply_text("\n".join(reply_parts))
        return True

    if new_plan["status"] == "needs_clarification":
        question = new_plan.get("question", "Kannst du das konkretisieren?")
        new_accumulated = accumulated + f"\n\nBob: {question}"
        reply = await message.reply_text(question)
        await memory.set_session(
            pool,
            user_id,
            chat_id,
            _SESSION_PLAN,
            {
                "plan": new_plan,
                "accumulated_context": new_accumulated,
                "clarification_rounds": rounds + 1,
                "bot_message_id": reply.message_id,
            },
        )
        return True

    if new_plan["status"] == "ready":
        plan_text = agent_planner.format_plan_message(new_plan)
        full_message = (
            plan_text
            + "\n\nAntworte auf diese Nachricht um den Plan anzupassen oder zu bestätigen."
        )
        new_accumulated = accumulated + f"\n\nBob: {plan_text}"
        reply = await message.reply_text(full_message)
        await memory.set_session(
            pool,
            user_id,
            chat_id,
            _SESSION_PLAN,
            {
                "plan": new_plan,
                "accumulated_context": new_accumulated,
                "clarification_rounds": rounds,
                "bot_message_id": reply.message_id,
            },
        )
        return True

    return True


async def _handle_clarification_session(
    update: Update,
    pool: asyncpg.Pool,
    user_id: int,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None,
    active_agents: list[dict],
) -> bool:
    session = await memory.get_session(pool, user_id, chat_id, _SESSION_CLARIFICATION)
    if not session:
        return False

    payload = session["payload"]
    bot_message_id: int | None = payload.get("bot_message_id")
    if bot_message_id is not None and reply_to_message_id != bot_message_id:
        return False

    message = update.effective_message
    agent_id: int = payload["agent_id"]
    most_likely_key: str = payload["most_likely_key"]

    await memory.clear_session(pool, user_id, chat_id, _SESSION_CLARIFICATION)

    _CONFIRM_SIGNALS = ("ja", "genau", "richtig", "stimmt", "korrekt", "yes", "yep")
    final_key = (
        most_likely_key
        if any(
            re.search(rf"\b{re.escape(s)}\b", text.lower()) for s in _CONFIRM_SIGNALS
        )
        else (text.lower().strip() if "_" in text.lower() else most_likely_key)
    )

    target_agent = next((a for a in active_agents if a["id"] == agent_id), None)
    if not target_agent:
        return True

    original_text = payload.get("original_text", "")
    edit_payload = await agent_edits.prepare_preference(
        pool, target_agent, original_text, state_key=final_key
    )
    if not edit_payload or isinstance(edit_payload, tuple):
        await message.reply_text(
            "Ich konnte keine klare Präferenz aus dem Feedback extrahieren."
        )
        return True

    config_data = parse_agent_config(target_agent.get("config", {}))
    edit_payload["agent_type"] = config_data.get("type", "unknown")

    confirmation_msg = agent_edits.format_confirmation_message(edit_payload)
    conf_id = await memory.replace_pending_confirmation(
        pool,
        chat_id,
        user_id,
        target_agent["id"],
        "preference",
        confirmation_msg,
        edit_payload,
    )
    sent = await message.reply_text(
        confirmation_msg, reply_markup=_confirmation_keyboard(conf_id)
    )
    await memory.save_agent_notification(
        pool,
        sent.message_id,
        chat_id,
        target_agent["id"],
        "confirmation",
        {"confirmation_id": conf_id},
    )
    return True


_AGENT_TALK_SYSTEM_SUFFIX = """
## Gespräch über einen deiner Agenten

Du sprichst gerade über einen deiner laufenden Agenten. Seine Pipeline-Struktur und sein Kontext stehen oben. Du sprichst ÜBER den Agenten in deiner eigenen Stimme — du bist nicht der Agent.

Führe ein normales, offenes Gespräch. Wenn der User ein Problem mit dem Agenten anspricht (eine Fehlbewertung, ein unerwünschtes Verhalten), dann analysiere es ehrlich anhand der Pipeline die du siehst: Welcher Step ist verantwortlich? Warum konnte das passieren? Spekuliere nicht über Steps die du nicht siehst.

Dräng nicht auf eine Änderung. Erst wenn das Gespräch zu einem KONKRETEN, vom User bestätigten Änderungswunsch gekommen ist — ihr seid euch einig was geändert werden soll — signalisierst du das am Ende deiner Antwort mit GENAU EINEM dieser Blöcke:

Für eine neue Präferenz oder ein neues Kriterium:
```edit_preference
<Beschreibung der gewünschten Änderung in einem Satz>
```

Für die Korrektur eines gespeicherten Inhalts:
```edit_data
<Beschreibung was konkret geändert werden soll>
```

Für ein systematisches Verhaltensproblem der Pipeline (z.B. ein llm_decide-Step der etwas übersieht):
```edit_step
<Beschreibung des Problems und was am Step anders sein soll>
```

Setze KEINEN Block wenn:
- Ihr noch in der Diagnose seid oder Ursachen besprecht
- Der User nur reagiert, fragt oder kommentiert
- Unklar ist ob und was genau geändert werden soll

Wenn der User den Agenten UMBENENNEN will: Das geht nicht über das Gespräch. Verweise ihn auf den Umbenennen-Button in der Agentenübersicht (/agents). Versuche nicht selbst umzubenennen.

Setze maximal einen Block pro Antwort, und nur am Ende."""


async def _parse_and_execute_edit_signal(
    response: str,
    pool: asyncpg.Pool,
    target_agent: dict,
    user_id: int,
    chat_id: int,
    message,
) -> tuple[str, bool]:
    signal_pattern = re.compile(
        r"```(edit_preference|edit_data|edit_step)\n(.*?)```",
        re.DOTALL,
    )
    match = signal_pattern.search(response)
    if not match:
        return response, False

    signal_type = match.group(1)
    description = match.group(2).strip()
    clean_response = response[: match.start()].strip()

    config_data = parse_agent_config(target_agent.get("config", {}))

    try:
        if signal_type == "edit_preference":
            pref_result = await agent_edits.prepare_preference(
                pool, target_agent, description
            )
            if isinstance(pref_result, tuple) and pref_result[0] == "clarification":
                _, clarification_text, most_likely_key = pref_result
                await memory.set_session(
                    pool,
                    user_id,
                    chat_id,
                    _SESSION_CLARIFICATION,
                    {
                        "agent_id": target_agent["id"],
                        "most_likely_key": most_likely_key,
                        "original_text": description,
                        "bot_message_id": None,
                    },
                    ttl_minutes=30,
                )
                await message.reply_text(clarification_text)
                return clean_response, True
            edit_payload = pref_result

        elif signal_type == "edit_data":
            ctx = await agent_context.load(pool, target_agent, description)
            loaded_data = ctx.get("loaded_data", {})
            edit_payload = None
            if loaded_data:
                first_path = next(iter(loaded_data))
                parts = first_path.split("/", 1)
                ns, key = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
                if key:
                    edit_payload = await agent_edits.prepare_data_edit(
                        pool, target_agent, description, ns, key
                    )
            if not edit_payload:
                pref_result = await agent_edits.prepare_preference(
                    pool, target_agent, description
                )
                edit_payload = pref_result if isinstance(pref_result, dict) else None

        elif signal_type == "edit_step":
            edit_payload = await agent_edits.prepare_step_patch(
                pool, target_agent, description
            )
            if not edit_payload:
                pref_result = await agent_edits.prepare_preference(
                    pool, target_agent, description
                )
                edit_payload = pref_result if isinstance(pref_result, dict) else None

        else:
            return clean_response, False

        if not edit_payload or not isinstance(edit_payload, dict):
            return clean_response, False

        edit_payload["agent_type"] = config_data.get("type", "unknown")
        confirmation_msg = agent_edits.format_confirmation_message(edit_payload)
        conf_id = await memory.replace_pending_confirmation(
            pool,
            chat_id,
            user_id,
            target_agent["id"],
            edit_payload.get("edit_type", "preference"),
            confirmation_msg,
            edit_payload,
        )
        sent = await message.reply_text(
            confirmation_msg,
            reply_markup=agent_edits.confirmation_keyboard(conf_id),
        )
        await memory.save_agent_notification(
            pool,
            sent.message_id,
            chat_id,
            target_agent["id"],
            "confirmation",
            {"confirmation_id": conf_id},
        )
        return clean_response, True

    except Exception as e:
        logger.warning("edit signal processing failed: %s", e)
        return clean_response, False


async def _run_agent_talk_turn(
    pool: asyncpg.Pool,
    reply_target,
    user_id: int,
    chat_id: int,
    agent: dict,
    user_text: str,
    active_agents: list[dict],
    wants_voice: bool = False,
    detected_language: str = "de",
    quoted_text: str | None = None,
    open_session: bool = True,
) -> None:
    """
    Ein Gesprächs-Turn über genau einen Agenten. Kennt kein Update-Objekt — bekommt
    das Antwortziel (eine Telegram-Message) übergeben. Speist sich aus Notification-Reply,
    Status-Button und Folgenachrichten der laufenden Agent-Talk-Session.
    """
    display = "User"
    user_memories = await memory.get_memories(pool, "user", user_id)
    history = await memory.get_recent_messages(pool, chat_id)

    ctx = await agent_context.load(pool, agent, user_text)
    agent_ctx_text = agent_context.format_for_system_prompt(ctx)

    base_system = brain.build_system_prompt(
        user_memories,
        [],
        [],
        [],
        display,
        None,
        active_agents=active_agents,
        agent_context=agent_ctx_text or None,
    )
    system = base_system + _AGENT_TALK_SYSTEM_SUFFIX

    llm_messages = brain.history_to_llm_messages(history)
    turn_text = user_text
    if quoted_text:
        turn_text = f"[Zitierte Agent-Meldung: {quoted_text}]\n{user_text}"
    llm_messages.append({"role": "user", "content": turn_text})

    try:
        response = await brain.chat(
            system=system,
            messages=llm_messages,
            capability=CAPABILITY_CHAT,
            caller="agent_talk",
            pool=pool,
        )
    except (ProviderRateLimitError, ProviderAuthError) as e:
        await reply_target.reply_text(ratelimit.rate_limit_message(e.provider))
        return

    if not response:
        logger.warning("_run_agent_talk_turn: unexpected empty response")
        return

    clean_response, edit_triggered = await _parse_and_execute_edit_signal(
        response, pool, agent, user_id, chat_id, reply_target
    )

    await memory.save_message(pool, chat_id, user_id, "user", turn_text)
    await memory.save_message(pool, chat_id, None, "assistant", clean_response)

    if clean_response:
        await _send_response(
            reply_target, clean_response, wants_voice, detected_language
        )

    if edit_triggered:
        await memory.clear_session(pool, user_id, chat_id, _SESSION_AGENT_TALK)
    elif open_session:
        new_history = await memory.get_recent_messages(pool, chat_id)
        await memory.set_session(
            pool,
            user_id,
            chat_id,
            _SESSION_AGENT_TALK,
            {
                "agent_id": agent["id"],
                "conversation_summary": _summarize_history_for_membership(new_history),
            },
            ttl_minutes=_AGENT_TALK_TTL_MINUTES,
        )

    snippet = _build_snippet(history, user_text, display)
    asyncio.create_task(
        extractor.extract_and_store_automatic(pool, user_id, display, snippet)
    )


async def _handle_agent_talk_session(
    update: Update,
    pool: asyncpg.Pool,
    user_id: int,
    chat_id: int,
    text: str,
    active_agents: list[dict],
    wants_voice: bool,
    detected_language: str,
) -> bool:
    """
    Prüft ob eine laufende Agent-Talk-Session existiert und ob die Nachricht dazugehört.
    'continue' → Turn läuft, True. 'closed' → Session weg, False (normaler Pfad greift).
    'unrelated' → Session bleibt, False (normaler Pfad greift, Session überlebt).
    """
    session = await memory.get_session(pool, user_id, chat_id, _SESSION_AGENT_TALK)
    if not session:
        return False

    payload = session["payload"]
    agent_id: int = payload.get("agent_id", 0)
    target_agent = next((a for a in active_agents if a["id"] == agent_id), None)
    if not target_agent:
        await memory.clear_session(pool, user_id, chat_id, _SESSION_AGENT_TALK)
        return False

    verdict = await agent_context.classify_talk_membership(
        target_agent["name"],
        payload.get("conversation_summary", ""),
        text,
    )

    if verdict == "closed":
        await memory.clear_session(pool, user_id, chat_id, _SESSION_AGENT_TALK)
        return False

    if verdict == "unrelated":
        return False

    await _run_agent_talk_turn(
        pool,
        update.effective_message,
        user_id,
        chat_id,
        target_agent,
        text,
        active_agents,
        wants_voice=wants_voice,
        detected_language=detected_language,
    )
    return True


async def _handle_agent_notification_reply(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    user_id: int,
    chat_id: int,
    notification: dict,
    active_agents: list[dict],
    wants_voice: bool,
    detected_language: str,
) -> None:
    """
    Pfad A: User antwortet auf eine Agent-Notification. Öffnet/führt die Agent-Talk-Session
    und zieht den zitierten Meldungstext als Kontext mit.
    """
    message = update.effective_message
    agent_id = notification.get("agent_id")
    target_agent = next((a for a in active_agents if a["id"] == agent_id), None)
    if not target_agent:
        return

    quoted = _quoted_text(message)
    await _run_agent_talk_turn(
        pool,
        message,
        user_id,
        chat_id,
        target_agent,
        text,
        active_agents,
        wants_voice=wants_voice,
        detected_language=detected_language,
        quoted_text=quoted,
    )


async def _handle_explicit_intent(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    intent: str,
    user_id: int,
    chat_id: int,
    active_agents: list[dict],
    active_tasks: list[dict],
) -> None:
    message = update.effective_message

    if intent == "agent_create":
        initial_context = f"User: {text}"
        initial_plan = await agent_planner.plan(initial_context, pool, 0)
        plan_text = agent_planner.format_plan_message(initial_plan)

        if initial_plan["status"] == "ready":
            full_message = (
                plan_text
                + "\n\nAntworte auf diese Nachricht um den Plan anzupassen oder zu bestätigen."
            )
            new_accumulated = initial_context + f"\n\nBob: {plan_text}"
        else:
            full_message = plan_text
            new_accumulated = initial_context + f"\n\nBob: {plan_text}"

        reply = await message.reply_text(full_message)
        await memory.set_session(
            pool,
            user_id,
            chat_id,
            _SESSION_PLAN,
            {
                "plan": initial_plan,
                "accumulated_context": new_accumulated,
                "clarification_rounds": 1
                if initial_plan["status"] == "needs_clarification"
                else 0,
                "bot_message_id": reply.message_id,
            },
            ttl_minutes=120,
        )
        return

    if intent == "agent_trigger":
        extracted = await intent_classifier.extract_trigger_payload(text, pool)
        agent_name: str = extracted.get("agent_name", "")
        action: str = extracted.get("action", "run")
        payload: dict = extracted.get("payload", {})
        target_agent = await agent_parser.resolve_agent_by_text(
            agent_name or text, active_agents
        )
        if not target_agent:
            names = ", ".join(a["name"] for a in active_agents)
            await message.reply_text(
                f"Ich bin nicht sicher welchen Agenten du meinst. Aktive Agenten: {names}"
            )
            return
        if action == "stop":
            await memory.deactivate_agent(pool, target_agent["id"])
            await message.reply_text(f"{target_agent['name']} wurde gestoppt.")
        else:
            await memory.enqueue_agent_trigger(
                pool, None, target_agent["name"], payload
            )
            payload_desc = f" mit Payload: {payload}" if payload else ""
            await message.reply_text(
                f"{target_agent['name']} wird beim nächsten Scheduler-Lauf ausgeführt{payload_desc}."
            )
        return

    if intent == "scraper_create":
        extracted = await intent_classifier.extract_scraper_create_params(text, pool)
        platforms: list[str] = extracted.get("platforms", [])
        target_agent_name: str = extracted.get("target_agent", "")
        if not platforms or not target_agent_name:
            await message.reply_text(
                "Ich konnte die Scraper-Parameter nicht erkennen. Beispiel: "
                "'Scrape Kleinanzeigen und eBay nach RTX 4090 und triggere Linus'."
            )
            return
        for platform in platforms:
            await memory.create_scraper_config(
                pool,
                platform=platform,
                category=extracted.get("category", "general"),
                query=extracted.get("query", ""),
                target_agent=target_agent_name,
                filters=extracted.get("filters", {}),
                poll_interval_seconds=extracted.get("poll_interval_seconds", 3600),
            )
        interval_min = extracted.get("poll_interval_seconds", 3600) // 60
        await message.reply_text(
            f"Scraper eingerichtet für {', '.join(platforms)} — "
            f"sucht nach '{extracted.get('query')}' und triggert {target_agent_name} "
            f"alle {interval_min} Minuten."
        )
        return

    if intent == "monitor_create":
        extracted = await intent_classifier.extract_monitor_create_params(text, pool)
        if not extracted or not extracted.get("target_agent"):
            await message.reply_text(
                "Ich konnte die Monitor-Parameter nicht vollständig erkennen. "
                "Beschreib welcher Agent getriggert werden soll und was überwacht werden soll."
            )
            return
        source: str = extracted.get("source", "agent")
        monitor_id = await memory.create_monitor_config(
            pool,
            monitor_type=extracted.get("monitor_type", "rss"),
            name=extracted.get("name", f"Monitor für {extracted['target_agent']}"),
            source=source,
            target_agent=extracted["target_agent"],
            feed_templates=extracted.get("feed_urls", [])
            if source == "static"
            else extracted.get("feed_templates", []),
            poll_interval_seconds=extracted.get("poll_interval_seconds", 3600),
            source_agent=extracted.get("source_agent", ""),
            source_state_key=extracted.get("source_state_key", ""),
            source_format=extracted.get("source_format", "comma_list"),
            keywords=extracted.get("keywords", []),
        )
        await message.reply_text(
            f"RSS-Monitor eingerichtet (ID: {monitor_id}) → {extracted['target_agent']}."
        )
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
            await message.reply_text(
                "Ich konnte keinen gültigen Zeitplan erkennen. Versuch's nochmal konkreter."
            )
        return

    if intent == "task_stop":
        quoted = _quoted_text(update.effective_message)
        stop_context = f"{text}\n\nZitierte Nachricht: {quoted}" if quoted else text
        stop_ids = await task_parser.parse_stop_request(stop_context, active_tasks)
        if stop_ids:
            count = await memory.deactivate_tasks_by_description(
                pool, user_id, stop_ids
            )
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
    group_memories = (
        await memory.get_memories(pool, "group", chat_id) if is_group else []
    )
    bot_memories = await memory.get_memories(pool, "bot", chat_id) if is_group else []
    reflection_memories = await memory.get_reflection_memories(pool, chat_id, user_id)
    observation_context = await observer.get_observation_context(pool, chat_id)
    history = await memory.get_recent_messages(pool, chat_id)

    system = brain.build_system_prompt(
        user_memories,
        group_memories,
        bot_memories,
        reflection_memories,
        display,
        group_title,
        active_agents=active_agents,
        observation_context=observation_context or None,
    )
    llm_messages = brain.history_to_llm_messages(history)
    quoted = _quoted_text(message)
    user_turn = f"{display}: {text}" if is_group and not triggered_by_mention else text
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
    except NoSearchResultsError:
        await message.reply_text(
            "Dazu habe ich gerade keine aktuellen Informationen gefunden."
        )
        return
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

    await _send_response(message, response, wants_voice, detected_language)

    snippet = _build_snippet(history, text, display)
    asyncio.create_task(
        extractor.extract_and_store_automatic(pool, user_id, display, snippet)
    )
    asyncio.create_task(
        extractor.extract_and_store_reflection(pool, chat_id, user_id, snippet)
    )


async def _handle_file_content(
    update: Update,
    pool: asyncpg.Pool,
    file_bytes: bytes,
    media_type: str,
    caption: str | None,
    triggered_by_mention: bool,
    detected_language: str = "de",
    forward_context: str | None = None,
) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if not message or not user:
        return
    is_group = chat.type in ("group", "supergroup")
    display = _display_name(user)
    group_title = chat.title if is_group else None

    await memory.upsert_user(
        pool, user.id, user.username, user.first_name, user.last_name
    )
    if is_group:
        await memory.upsert_group(pool, chat.id, group_title)

    user_memories = await memory.get_memories(pool, "user", user.id)
    group_memories = (
        await memory.get_memories(pool, "group", chat.id) if is_group else []
    )
    bot_memories = await memory.get_memories(pool, "bot", chat.id) if is_group else []
    reflection_memories = await memory.get_reflection_memories(pool, chat.id, user.id)
    active_agents = await memory.get_active_agents_for_user(pool, user.id)
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
    user_text = caption or (
        "Was siehst du hier?"
        if not forward_context
        else f"{forward_context} — Bitte verarbeite den Inhalt."
    )
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    content: list[dict] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        },
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
    await message.reply_text(response)


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

    await memory.upsert_user(
        pool, user.id, user.username, user.first_name, user.last_name
    )
    if is_group:
        await memory.upsert_group(pool, chat.id, group_title)

    text = (
        transcribed_text
        if transcribed_text is not None
        else (message.text or "").strip()
    )
    if not text:
        return
    display = _display_name(user)

    explicit_reply = await extractor.handle_explicit_memory(
        pool, user.id, group_id, text
    )
    if explicit_reply is not None:
        await message.reply_text(explicit_reply)
        return

    reply_to_id: int | None = (
        message.reply_to_message.message_id if message.reply_to_message else None
    )

    if await _handle_pending_confirmation(
        update, pool, user.id, chat.id, text, reply_to_id
    ):
        return

    if await _handle_plan_session(update, pool, user.id, chat.id, text, reply_to_id):
        return

    active_agents = await memory.get_active_agents_for_user(pool, user.id)

    if await _handle_clarification_session(
        update, pool, user.id, chat.id, text, reply_to_id, active_agents
    ):
        return

    active_tasks = await memory.get_active_tasks_for_user(pool, user.id)

    classified = await intent_classifier.classify(
        text,
        pool,
        has_active_agents=bool(active_agents),
        has_active_tasks=bool(active_tasks),
    )
    intent = classified["intent"]
    needs_search = classified["needs_search"]
    wants_voice = force_voice or classified["wants_voice"]

    logger.info(
        "_reply intent=%s search=%s voice=%s", intent, needs_search, wants_voice
    )

    if intent != "none":
        await memory.clear_session(pool, user.id, chat.id, _SESSION_AGENT_TALK)
        await _handle_explicit_intent(
            update,
            pool,
            text,
            intent,
            user.id,
            chat.id,
            active_agents,
            active_tasks,
        )
        return

    if reply_to_id:
        notification = await memory.get_agent_notification(pool, reply_to_id, chat.id)
        if notification and notification.get("notification_type") not in (
            "confirmation",
            "adjust_request",
        ):
            await _handle_agent_notification_reply(
                update,
                pool,
                text,
                user.id,
                chat.id,
                notification,
                active_agents,
                wants_voice,
                detected_language,
            )
            return

    if await _handle_agent_talk_session(
        update,
        pool,
        user.id,
        chat.id,
        text,
        active_agents,
        wants_voice,
        detected_language,
    ):
        return

    await _handle_chat(
        update,
        pool,
        text,
        user.id,
        chat.id,
        is_group,
        triggered_by_mention,
        needs_search,
        wants_voice,
        detected_language,
        active_agents,
        display,
        group_title,
    )


async def _transcribe_audio_message(message) -> tuple[bytes, str] | None:
    if message.voice:
        f = await message.voice.get_file()
        return bytes(await f.download_as_bytearray()), "voice"
    if message.audio:
        f = await message.audio.get_file()
        return bytes(await f.download_as_bytearray()), "audio"
    if message.document and message.document.mime_type in (
        "audio/ogg",
        "audio/mp4",
        "audio/mpeg",
        "audio/aac",
        "audio/x-m4a",
    ):
        f = await message.document.get_file()
        return bytes(await f.download_as_bytearray()), "document_audio"
    return None


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    forward_context = _extract_forward_context(message)
    try:
        result = await _transcribe_audio_message(message)
        if not result:
            return
        audio_bytes, _ = result
        transcribed, lang = await voice.transcribe(audio_bytes)
    except Exception as e:
        logger.error("STT failed: %s", e)
        await message.reply_text("Sprachnachricht konnte nicht transkribiert werden.")
        return
    if not transcribed.strip():
        return
    if forward_context:
        transcribed = f"{forward_context}\n{transcribed}"
    transcribed_lower = transcribed.lower()
    is_mention = (
        bot_username and f"@{bot_username}".lower() in transcribed_lower
    ) or bool(
        re.search(rf"\b{re.escape(config.BOT_NAME.lower())}\b", transcribed_lower)
    )
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )
    if is_group:
        if is_mention or is_reply_to_bot:
            await _reply(
                update,
                pool,
                triggered_by_mention=True,
                transcribed_text=transcribed,
                detected_language=lang,
            )
        else:
            should = await decider.should_respond_spontaneously(
                pool=pool, group_id=chat.id, message_text=transcribed
            )
            if should:
                await _reply(
                    update,
                    pool,
                    triggered_by_mention=False,
                    transcribed_text=transcribed,
                    detected_language=lang,
                )
    else:
        await _reply(
            update,
            pool,
            triggered_by_mention=True,
            transcribed_text=transcribed,
            detected_language=lang,
            force_voice=True,
        )


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

    agent_names_lower: list[str] = []
    if is_group:
        agents_for_mention = await memory.get_active_agents_for_user(pool, user.id)
        agent_names_lower = [a["name"].lower() for a in agents_for_mention]

    text_lower = text.lower()
    is_mention = (
        (bot_username and f"@{bot_username}".lower() in text_lower)
        or bool(re.search(rf"\b{re.escape(config.BOT_NAME.lower())}\b", text_lower))
        or any(
            re.search(rf"\b{re.escape(name)}\b", text_lower)
            for name in agent_names_lower
        )
    )
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )
    forward_context = _extract_forward_context(message)
    if forward_context and text:
        text = f"{forward_context}\n{text}"

    if is_group:
        if is_mention or is_reply_to_bot:
            if ratelimit.is_any_limited():
                await message.reply_text(ratelimit.rate_limit_message())
                return
            await _reply(
                update,
                pool,
                triggered_by_mention=True,
                transcribed_text=text if forward_context else None,
            )
        else:
            if ratelimit.is_any_limited():
                return
            should = await decider.should_respond_spontaneously(
                pool=pool, group_id=chat.id, message_text=text
            )
            if should:
                await _reply(update, pool, triggered_by_mention=False)
    else:
        if ratelimit.is_any_limited():
            await message.reply_text(ratelimit.rate_limit_message())
            return
        await _reply(
            update,
            pool,
            triggered_by_mention=True,
            transcribed_text=text if forward_context else None,
        )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    message = update.effective_message
    if not message or not message.audio:
        return
    if ratelimit.is_any_limited():
        await message.reply_text(ratelimit.rate_limit_message())
        return
    forward_context = _extract_forward_context(message)
    try:
        result = await _transcribe_audio_message(message)
        if not result:
            return
        audio_bytes, _ = result
        transcribed, lang = await voice.transcribe(audio_bytes)
    except Exception as e:
        logger.error("STT (audio) failed: %s", e)
        await message.reply_text("Audiodatei konnte nicht transkribiert werden.")
        return
    if not transcribed.strip():
        return
    if forward_context:
        transcribed = f"{forward_context}\n{transcribed}"
    await _reply(
        update,
        pool,
        triggered_by_mention=True,
        transcribed_text=transcribed,
        detected_language="de",
    )


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
    caption_lower = caption.lower() if caption else ""
    is_mention = bool(
        caption
        and (
            (bot_username and f"@{bot_username}".lower() in caption_lower)
            or re.search(rf"\b{re.escape(config.BOT_NAME.lower())}\b", caption_lower)
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
        update,
        pool,
        file_bytes=bytes(file_bytes),
        media_type="image/jpeg",
        caption=caption,
        triggered_by_mention=triggered,
        forward_context=_extract_forward_context(message),
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
    caption_lower = caption.lower() if caption else ""
    SUPPORTED_MIME: dict[str, str] = {
        "image/jpeg": "image/jpeg",
        "image/png": "image/png",
        "image/gif": "image/gif",
        "image/webp": "image/webp",
        "application/pdf": "application/pdf",
    }
    AUDIO_MIME: set[str] = {
        "audio/ogg",
        "audio/mp4",
        "audio/mpeg",
        "audio/aac",
        "audio/x-m4a",
    }
    bot_username = context.bot.username
    is_group = chat.type in ("group", "supergroup")
    is_mention = bool(
        caption
        and (
            (bot_username and f"@{bot_username}".lower() in caption_lower)
            or re.search(rf"\b{re.escape(config.BOT_NAME.lower())}\b", caption_lower)
        )
    )
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )
    triggered = not is_group or is_mention or is_reply_to_bot

    if mime in AUDIO_MIME:
        if is_group and not triggered:
            return
        if ratelimit.is_any_limited():
            if triggered:
                await message.reply_text(ratelimit.rate_limit_message())
            return
        forward_context = _extract_forward_context(message)
        try:
            result = await _transcribe_audio_message(message)
            if not result:
                return
            audio_bytes, _ = result
            transcribed, lang = await voice.transcribe(audio_bytes)
        except Exception as e:
            logger.error("STT (document audio) failed: %s", e)
            await message.reply_text("Audiodatei konnte nicht transkribiert werden.")
            return
        if not transcribed.strip():
            return
        if forward_context:
            transcribed = f"{forward_context}\n{transcribed}"
        await _reply(
            update,
            pool,
            triggered_by_mention=True,
            transcribed_text=transcribed,
            detected_language="de",
        )
        return

    if mime not in SUPPORTED_MIME:
        await message.reply_text(
            f"Dieses Dateiformat ({mime or 'unbekannt'}) kann ich noch nicht lesen."
        )
        return
    if is_group and not triggered:
        return
    if ratelimit.is_any_limited():
        if triggered:
            await message.reply_text(ratelimit.rate_limit_message())
        return
    doc_file = await doc.get_file()
    file_bytes = await doc_file.download_as_bytearray()
    await _handle_file_content(
        update,
        pool,
        file_bytes=bytes(file_bytes),
        media_type=SUPPORTED_MIME[mime],
        caption=caption,
        triggered_by_mention=triggered,
        forward_context=_extract_forward_context(message),
    )


async def handle_callback_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    query = update.callback_query
    user = update.effective_user
    if not query or not user:
        return
    await query.answer()
    data = query.data or ""
    parts = data.split(":")

    if parts[0] == "confirm" and len(parts) == 3:
        action = parts[1]
        try:
            confirmation_id = int(parts[2])
        except ValueError:
            return

        chat_id = query.message.chat.id
        pending = await memory.get_pending_confirmation(pool, user.id, chat_id)
        if not pending or pending["id"] != confirmation_id:
            await query.edit_message_text("Diese Bestätigung ist abgelaufen.")
            return

        edit_payload = pending["payload"]

        if action == "yes":
            await memory.clear_pending_confirmation(pool, user.id, chat_id)
            await memory.clear_session(pool, user.id, chat_id, _SESSION_AGENT_TALK)
            result = await agent_edits.execute_edit(pool, edit_payload)
            rollback_id = confirmation_id
            _pending_rollbacks[rollback_id] = edit_payload
            loop = asyncio.get_running_loop()
            loop.call_later(300, lambda: _pending_rollbacks.pop(rollback_id, None))
            result_with_hint = (
                f"{result}\n\n(Rückgängig nur in den nächsten 5 Minuten möglich — "
                "und nicht über einen Neustart hinweg.)"
            )
            await query.edit_message_text(
                result_with_hint,
                reply_markup=agent_edits.rollback_keyboard(rollback_id),
            )

        elif action == "no":
            await memory.clear_pending_confirmation(pool, user.id, chat_id)
            await memory.clear_session(pool, user.id, chat_id, _SESSION_AGENT_TALK)
            await query.edit_message_text("Abgebrochen.")

        elif action == "adjust":
            sent = await query.message.reply_text("Was soll ich ändern?")
            await memory.save_agent_notification(
                pool,
                sent.message_id,
                chat_id,
                edit_payload.get("agent_id", 0),
                "adjust_request",
                {"confirmation_id": confirmation_id},
            )

        elif action == "rollback":
            rollback_payload = _pending_rollbacks.pop(confirmation_id, None)
            if not rollback_payload:
                await query.edit_message_text(
                    "Rückgängig nicht mehr möglich — das 5-Minuten-Fenster ist vorbei "
                    "oder der Bot wurde zwischenzeitlich neu gestartet."
                )
                return
            result = await agent_edits.rollback_edit(pool, rollback_payload)
            await query.edit_message_text(result)
        return

    if parts[0] == "agent" and len(parts) == 3:
        action = parts[1]
        try:
            agent_id = int(parts[2])
        except ValueError:
            return

        chat_id = query.message.chat.id
        active_agents = await memory.get_active_agents_for_user(pool, user.id)
        agent = next((a for a in active_agents if a["id"] == agent_id), None)
        if not agent:
            await query.edit_message_text("Dieser Agent existiert nicht mehr.")
            return

        if action == "stop":
            await memory.deactivate_agent(pool, agent_id)
            await query.edit_message_text(f"{agent['name']} wurde gestoppt.")
        elif action == "status":
            await _run_agent_talk_turn(
                pool,
                query.message,
                user.id,
                chat_id,
                agent,
                "Was ist dein aktueller Status und was hast du bisher beobachtet?",
                active_agents,
            )
        elif action == "rename":
            context.user_data["awaiting_rename_agent_id"] = agent_id
            await query.message.reply_text(
                f"Wie soll {agent['name']} heißen? Schreib einfach den neuen Namen."
            )
        return


async def handle_command_help(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    await update.effective_message.reply_text(greeter.introduction_text())


async def handle_command_agents(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
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
        schedule_display = agent["schedule"] or "nur auf Trigger"
        await update.effective_message.reply_text(
            f"{agent['name']} — {instruction}… ({schedule_display})",
            reply_markup=_agent_keyboard(agent["id"]),
        )


async def handle_command_tasks(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    pool: asyncpg.Pool = context.bot_data["pool"]
    user = update.effective_user
    if not user:
        return
    active_tasks = await memory.get_active_tasks_for_user(pool, user.id)
    if not active_tasks:
        await update.effective_message.reply_text("Du hast keine aktiven Aufgaben.")
        return
    lines = [f"{t['id']}. {t['description']} — {t['schedule']}" for t in active_tasks]
    await update.effective_message.reply_text(
        "Deine aktiven Aufgaben:\n" + "\n".join(lines)
    )

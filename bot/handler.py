from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timedelta, timezone

import asyncpg
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from bot import (
    agent_context,
    agent_edits,
    agent_parser,
    agent_planner,
    agent_runner,
    agent_system_parser,
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
from bot.agent_parser import regenerate_pipeline_for_agent
from bot.brain import ProviderAuthError, ProviderRateLimitError
from bot.models import CAPABILITY_CHAT, CAPABILITY_MULTIMODAL
from bot.utils import parse_agent_config

logger = logging.getLogger(__name__)

from typing import TypedDict


class _PendingPlan(TypedDict):
    plan: dict
    accumulated_context: str
    clarification_rounds: int
    bot_message_id: int | None


_pending_plans: dict[int, _PendingPlan] = {}
_pending_rollbacks: dict[int, dict] = {}
_pending_clarification_keys: dict[int, tuple[int, str]] = {}
_pending_agent_context: dict[int, tuple[int, datetime]] = {}

_AGENT_CONTEXT_TTL_MINUTES = 30


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
        if name:
            return f"[Weitergeleitet von: {name}]"
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
    forward_prefix = f"{forward_context}\n" if forward_context else ""
    user_text = caption if caption else "Was siehst du hier?"
    if forward_context and not caption:
        user_text = f"{forward_context} — Bitte verarbeite den Inhalt dieser weitergeleiteten Nachricht."
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
    if not triggered_by_mention and is_group:
        await memory.update_spontaneous_timestamp(pool, chat.id)
    await message.reply_text(response)


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
                reply_markup=agent_edits.confirmation_keyboard(conf_id),
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


async def _handle_pending_plan(
    update: Update,
    pool: asyncpg.Pool,
    user_id: int,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None,
) -> bool:
    pending = _pending_plans.get(user_id)
    if not pending:
        return False

    bot_message_id: int | None = pending.get("bot_message_id")
    if bot_message_id is not None and reply_to_message_id != bot_message_id:
        return False

    message = update.effective_message
    accumulated = pending["accumulated_context"]
    rounds = pending.get("clarification_rounds", 0)
    current_plan = pending.get("plan", {})
    accumulated += f"\n\nUser: {text}"
    new_plan = await agent_planner.plan(accumulated, pool, rounds)

    if new_plan["status"] == "confirmed":
        if current_plan.get("status") != "ready":
            await message.reply_text(
                "Ich habe noch keinen fertigen Plan — beschreib mir erst was ich bauen soll."
            )
            return True

        prepared = await agent_planner.finalize(
            plan_result=current_plan,
            accumulated_context=accumulated,
            user_id=user_id,
            source_chat_id=chat_id,
            pool=pool,
        )
        _pending_plans.pop(user_id, None)

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
                logger.info("created agent: %s", agent_cfg["name"])
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
            monitor_names = ", ".join(m["name"] for m in created_monitors)
            reply_parts.append(f"RSS-Monitor(e) eingerichtet: {monitor_names}.")
        if created_scrapers:
            scraper_summary = ", ".join(
                f"{s['platform']} → {s['target_agent']}" for s in created_scrapers
            )
            reply_parts.append(f"Scraper eingerichtet: {scraper_summary}.")
        if unavailable_scrapers:
            missing = ", ".join(s["platform"] for s in unavailable_scrapers)
            reply_parts.append(
                f"Hinweis: Scraper für {missing} sind noch nicht verfügbar und wurden nicht angelegt."
            )

        await message.reply_text("\n".join(reply_parts))
        return True

    if new_plan["status"] == "needs_clarification":
        pending["clarification_rounds"] = rounds + 1
        pending["plan"] = new_plan
        pending["accumulated_context"] = accumulated
        question = new_plan.get("question", "Kannst du das konkretisieren?")
        pending["accumulated_context"] += f"\n\nBob: {question}"
        reply = await message.reply_text(question)
        pending["bot_message_id"] = reply.message_id
        return True

    if new_plan["status"] == "ready":
        pending["plan"] = new_plan
        pending["clarification_rounds"] = rounds
        plan_text = agent_planner.format_plan_message(new_plan)
        full_message = (
            plan_text
            + "\n\nAntworte auf diese Nachricht um den Plan anzupassen oder zu bestätigen."
        )
        pending["accumulated_context"] = accumulated + f"\n\nBob: {plan_text}"
        reply = await message.reply_text(full_message)
        pending["bot_message_id"] = reply.message_id
        return True

    return True


async def _handle_pending_clarification(
    update: Update,
    pool: asyncpg.Pool,
    user_id: int,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None,
    active_agents: list[dict],
) -> bool:
    pending = _pending_clarification_keys.get(user_id)
    if not pending:
        return False

    bot_message_id, state_key = pending
    if reply_to_message_id != bot_message_id:
        return False

    message = update.effective_message
    agent_id, confirmed_key = _pending_clarification_keys.pop(user_id)

    _KEY_CONFIRM_SIGNALS = ("ja", "genau", "richtig", "stimmt", "korrekt", "yes", "yep")
    text_lower = text.lower().strip()

    if any(sig in text_lower for sig in _KEY_CONFIRM_SIGNALS):
        final_key = confirmed_key
    else:
        final_key = text_lower if "_" in text_lower else confirmed_key

    target_agent = next((a for a in active_agents if a["id"] == agent_id), None)
    if not target_agent:
        return True

    original_text = getattr(message.reply_to_message, "text", "") or ""
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
        confirmation_msg, reply_markup=agent_edits.confirmation_keyboard(conf_id)
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


async def _handle_agent_feedback(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    user_id: int,
    chat_id: int,
    target_agent: dict,
    edit_type: str | None,
    classified: dict,
) -> None:
    message = update.effective_message
    name = target_agent["name"]

    if not edit_type:
        ctx = await agent_context.load_deep(pool, target_agent, text)
        ctx_text = agent_context.format_for_system_prompt(ctx)
        user_memories = await memory.get_memories(pool, "user", user_id)
        history = await memory.get_recent_messages(pool, chat_id)
        system = brain.build_system_prompt(
            user_memories, [], [], [], "", None, agent_context=ctx_text
        )
        llm_messages = brain.history_to_llm_messages(history)
        llm_messages.append({"role": "user", "content": text})
        try:
            response = await brain.chat(
                system=system,
                messages=llm_messages,
                capability=CAPABILITY_CHAT,
                caller="handler_agent_feedback",
                pool=pool,
            )
            await message.reply_text(response)
        except (ProviderRateLimitError, ProviderAuthError) as e:
            await message.reply_text(ratelimit.rate_limit_message(e.provider))
        return

    await memory.clear_pending_confirmation(pool, user_id, chat_id)

    edit_payload: dict | None = None

    if edit_type == "data_edit":
        ctx = await agent_context.load_deep(pool, target_agent, text)
        loaded_data = ctx.get("loaded_data", {})
        if loaded_data:
            first_path = next(iter(loaded_data))
            parts = first_path.split("/", 1)
            ns, key = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
            if key:
                edit_payload = await agent_edits.prepare_data_edit(
                    pool, target_agent, text, ns, key
                )
        if not edit_payload:
            edit_type = "preference"

    elif edit_type == "step_patch":
        edit_payload = await agent_edits.prepare_step_patch(pool, target_agent, text)
        if not edit_payload:
            edit_type = "preference"

    if edit_type == "preference":
        pref_result = await agent_edits.prepare_preference(pool, target_agent, text)
        if pref_result is None:
            await message.reply_text(
                "Ich konnte keine klare Präferenz aus dem Feedback extrahieren."
            )
            return
        if isinstance(pref_result, tuple) and pref_result[0] == "clarification":
            _, clarification_text, most_likely_key = pref_result
            sent = await message.reply_text(clarification_text)
            _pending_clarification_keys[user_id] = (target_agent["id"], most_likely_key)
            return
        edit_payload = pref_result

    if not edit_payload:
        return

    config_data = parse_agent_config(target_agent.get("config", {}))
    edit_payload["agent_type"] = config_data.get("type", "unknown")

    confirmation_msg = agent_edits.format_confirmation_message(edit_payload)
    conf_id = await memory.replace_pending_confirmation(
        pool,
        chat_id,
        user_id,
        target_agent["id"],
        edit_type,
        confirmation_msg,
        edit_payload,
    )
    sent = await message.reply_text(
        confirmation_msg, reply_markup=agent_edits.confirmation_keyboard(conf_id)
    )
    await memory.save_agent_notification(
        pool,
        sent.message_id,
        chat_id,
        target_agent["id"],
        "confirmation",
        {"confirmation_id": conf_id},
    )


async def _handle_agent_intent(
    update: Update,
    pool: asyncpg.Pool,
    text: str,
    intent: str,
    user_id: int,
    chat_id: int,
    active_agents: list[dict],
    notification_context: dict | None = None,
    classified: dict | None = None,
) -> None:
    message = update.effective_message

    if intent == "agent_list":
        if not active_agents:
            await message.reply_text("Du hast keine aktiven Agenten.")
        else:
            for agent in active_agents:
                instruction = parse_agent_config(agent["config"]).get(
                    "instruction", ""
                )[:80]
                schedule_display = agent["schedule"] or "nur auf Trigger"
                await message.reply_text(
                    f"{agent['name']} — {instruction}… ({schedule_display})",
                    reply_markup=_agent_keyboard(agent["id"]),
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
            f"alle {interval_min} Minuten bei neuen Listings."
        )
        return

    if intent in ("agent_system", "agent_create"):
        initial_context = f"User: {text}"
        initial_plan = await agent_planner.plan(
            accumulated_context=initial_context, pool=pool, clarification_rounds=0
        )
        plan_text = agent_planner.format_plan_message(initial_plan)
        if initial_plan["status"] == "ready":
            accumulated_with_bob = initial_context + f"\n\nBob: {plan_text}"
            full_message = (
                plan_text
                + "\n\nAntworte auf diese Nachricht um den Plan anzupassen oder zu bestätigen."
            )
        else:
            accumulated_with_bob = initial_context + f"\n\nBob: {plan_text}"
            full_message = plan_text

        _pending_plans[user_id] = {
            "plan": initial_plan,
            "accumulated_context": accumulated_with_bob,
            "clarification_rounds": 1
            if initial_plan["status"] == "needs_clarification"
            else 0,
            "bot_message_id": None,
        }
        reply = await message.reply_text(full_message)
        _pending_plans[user_id]["bot_message_id"] = reply.message_id
        return

    if not active_agents:
        await message.reply_text("Du hast keine aktiven Agenten.")
        return

    if intent == "agent_feedback":
        target_agent: dict | None = None
        if notification_context:
            agent_id = notification_context.get("agent_id")
            if agent_id:
                target_agent = next(
                    (a for a in active_agents if a["id"] == agent_id), None
                )
                if target_agent:
                    logger.info(
                        "agent_feedback: resolved via notification_context → %s",
                        target_agent["name"],
                    )
        if not target_agent:
            extracted = await intent_classifier.extract_agent_talk(text, pool)
            agent_name = extracted.get("agent_name", "")
            target_agent = await agent_parser.resolve_agent_by_text(
                agent_name or text, active_agents
            )
        if not target_agent:
            names = ", ".join(a["name"] for a in active_agents)
            await message.reply_text(
                f"Ich bin nicht sicher welchen Agenten du meinst. Aktive Agenten: {names}"
            )
            return
        edit_type = (classified or {}).get("edit_type")
        await _handle_agent_feedback(
            update,
            pool,
            text,
            user_id,
            chat_id,
            target_agent,
            edit_type,
            classified or {},
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

    if intent == "agent_talk":
        extracted = await intent_classifier.extract_agent_talk(text, pool)
        agent_name = extracted.get("agent_name", "")
        talk_type: str = extracted.get("talk_type", "query")
        target_agent = await agent_parser.resolve_agent_by_text(
            agent_name or text, active_agents
        )
        if not target_agent:
            names = ", ".join(a["name"] for a in active_agents)
            await message.reply_text(
                f"Ich bin nicht sicher welchen Agenten du meinst. Aktive Agenten: {names}"
            )
            return

        if talk_type == "regenerate_pipeline":
            current_config = parse_agent_config(target_agent["config"])
            updated_config = await regenerate_pipeline_for_agent(
                current_config, pool=pool
            )
            await memory.update_agent_config(pool, target_agent["id"], updated_config)
            steps = len(updated_config.get("steps", []))
            await message.reply_text(
                f"{target_agent['name']}: Pipeline neu generiert ({steps} Steps)."
            )
        else:
            use_deep = agent_context.needs_deep_load(text)
            if use_deep:
                ctx = await agent_context.load_deep(pool, target_agent, text)
            else:
                ctx = await agent_context.load_shallow(pool, target_agent)

            ctx_text = agent_context.format_for_system_prompt(ctx)

            if use_deep and not ctx.get("has_data", True):
                await message.reply_text(
                    ctx.get(
                        "message",
                        f"{target_agent['name']} hat keine abfragbaren Daten dazu.",
                    )
                )
                return

            state = await memory.get_agent_state(pool, target_agent["id"])
            agent_memories = await memory.get_agent_memories(pool, target_agent["id"])
            (
                response,
                new_config,
                new_name,
                clarification_key,
            ) = await agent_parser.handle_agent_talk(
                text,
                target_agent,
                state,
                agent_memories,
                pool=pool,
            )
            if new_config is not None:
                await memory.update_agent_config(pool, target_agent["id"], new_config)
            if new_name is not None:
                await memory.rename_agent(pool, target_agent["id"], new_name)
            if clarification_key is not None:
                sent = await message.reply_text(response)
                _pending_clarification_keys[user_id] = (
                    target_agent["id"],
                    clarification_key,
                )
                return

            if ctx_text and not use_deep:
                response = f"{ctx_text}\n\n{response}"

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
            lines = [
                f"{t['id']}. {t['description']} — {t['schedule']}" for t in active_tasks
            ]
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


async def _handle_scraper_intent(
    update: Update, pool: asyncpg.Pool, text: str, user_id: int
) -> None:
    message = update.effective_message
    extracted = await intent_classifier.extract_scraper_create_params(text, pool)
    if (
        not extracted
        or not extracted.get("target_agent")
        or not extracted.get("platforms")
    ):
        await message.reply_text(
            "Ich konnte die Scraper-Parameter nicht erkennen. Beispiel: "
            "'Richte einen Scraper auf Kleinanzeigen und eBay ein der RTX 4090 sucht und Linus triggert'."
        )
        return
    platforms: list[str] = extracted["platforms"]
    for platform in platforms:
        config_id = await memory.create_scraper_config(
            pool,
            platform=platform,
            category=extracted.get("category", "general"),
            query=extracted.get("query", ""),
            filters=extracted.get("filters", {}),
            target_agent=extracted["target_agent"],
            poll_interval_seconds=extracted.get("poll_interval_seconds", 3600),
        )
        logger.info("scraper config created: id=%d platform=%s", config_id, platform)
    poll = extracted.get("poll_interval_seconds", 3600)
    interval_display = (
        f"{poll // 60} Minuten" if poll < 3600 else f"{poll // 3600} Stunde(n)"
    )
    await message.reply_text(
        f"Scraper eingerichtet ({len(platforms)} Konfig(s)): '{extracted.get('query')}' auf "
        f"{', '.join(platforms)} alle {interval_display} → {extracted['target_agent']}."
    )


async def _handle_monitor_intent(
    update: Update, pool: asyncpg.Pool, text: str, user_id: int
) -> None:
    message = update.effective_message
    extracted = await intent_classifier.extract_monitor_create_params(text, pool)

    if not extracted or not extracted.get("target_agent"):
        await message.reply_text(
            "Ich konnte die Monitor-Parameter nicht vollständig erkennen. "
            "Beschreib welcher Agent getriggert werden soll und was überwacht werden soll."
        )
        return

    source: str = extracted.get("source", "agent")
    monitor_type: str = extracted.get("monitor_type", "rss")
    name: str = extracted.get("name", f"Monitor für {extracted['target_agent']}")
    target_agent: str = extracted["target_agent"]
    keywords: list[str] = extracted.get("keywords", [])
    poll_interval: int = extracted.get("poll_interval_seconds", 3600)

    if source == "static":
        feed_urls: list[str] = extracted.get("feed_urls", [])
        if not feed_urls:
            await message.reply_text(
                "Keine Feed-URLs erkannt. Bitte gib mindestens eine RSS-URL an."
            )
            return
        monitor_id = await memory.create_monitor_config(
            pool,
            monitor_type=monitor_type,
            name=name,
            source="static",
            target_agent=target_agent,
            feed_templates=feed_urls,
            poll_interval_seconds=poll_interval,
            keywords=keywords,
        )
        feeds_display = ", ".join(feed_urls)
        kw_display = f" · Keywords: {', '.join(keywords)}" if keywords else ""
        await message.reply_text(
            f"RSS-Monitor eingerichtet (ID: {monitor_id}): überwacht {feeds_display}{kw_display} "
            f"und triggert {target_agent} bei neuen Artikeln."
        )
    else:
        source_agent: str = extracted.get("source_agent", "")
        source_state_key: str = extracted.get("source_state_key", "")
        source_format: str = extracted.get("source_format", "comma_list")
        feed_templates: list[str] = extracted.get(
            "feed_templates",
            [
                "https://news.google.com/rss/search?q={query}&hl=de&gl=DE&ceid=DE:de",
            ],
        )
        if not source_agent or not source_state_key:
            await message.reply_text(
                "Für einen Agent-basierten Monitor brauche ich den Namen des Quell-Agents "
                "und den State-Key der Watchlist."
            )
            return
        monitor_id = await memory.create_monitor_config(
            pool,
            monitor_type=monitor_type,
            name=name,
            source="agent",
            target_agent=target_agent,
            feed_templates=feed_templates,
            poll_interval_seconds=poll_interval,
            source_agent=source_agent,
            source_state_key=source_state_key,
            source_format=source_format,
            keywords=keywords,
        )
        await message.reply_text(
            f"RSS-Monitor eingerichtet (ID: {monitor_id}): überwacht {source_agent}/{source_state_key} "
            f"und triggert {target_agent} bei neuen Artikeln."
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
    agent_ctx_text: str = "",
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
        agent_context=agent_ctx_text or None,
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
    asyncio.create_task(
        extractor.extract_and_store_automatic(pool, user_id, display, snippet)
    )
    asyncio.create_task(
        extractor.extract_and_store_reflection(pool, chat_id, user_id, snippet)
    )


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

    reply_to_id: int | None = None
    if message.reply_to_message is not None:
        reply_to_id = message.reply_to_message.message_id

    if await _handle_pending_confirmation(
        update, pool, user.id, chat.id, text, reply_to_id
    ):
        return

    if await _handle_pending_plan(update, pool, user.id, chat.id, text, reply_to_id):
        return

    active_agents = await memory.get_active_agents_for_user(pool, user.id)

    if await _handle_pending_clarification(
        update, pool, user.id, chat.id, text, reply_to_id, active_agents
    ):
        return

    notification_context: dict | None = None
    if reply_to_id:
        notification_context = await memory.get_agent_notification(
            pool, reply_to_id, chat.id
        )

    if notification_context and notification_context.get("notification_type") not in (
        "confirmation",
        "adjust_request",
    ):
        agent_id = notification_context.get("agent_id")
        target_agent = next((a for a in active_agents if a["id"] == agent_id), None)
        if target_agent:
            _pending_agent_context[user.id] = (agent_id, datetime.now(timezone.utc))
            logger.info(
                "_reply: reply to agent notification → classifying with notification context for %s",
                target_agent["name"],
            )
            active_tasks = await memory.get_active_tasks_for_user(pool, user.id)
            classified = await intent_classifier.classify(
                text,
                pool,
                has_active_agents=bool(active_agents),
                has_active_tasks=bool(active_tasks),
                notification_context=notification_context,
            )
            intent = classified["intent"]
            if intent not in ("agent_feedback", "agent_talk", "agent_trigger", "none"):
                classified["intent"] = "agent_feedback"
            if classified["intent"] != "none":
                await _handle_agent_intent(
                    update,
                    pool,
                    text,
                    classified["intent"],
                    user.id,
                    chat.id,
                    active_agents,
                    notification_context=notification_context,
                    classified=classified,
                )
                return

    pending_agent = _pending_agent_context.get(user.id)
    if pending_agent and reply_to_id is not None:
        agent_id, set_at = pending_agent
        age_minutes = (datetime.now(timezone.utc) - set_at).total_seconds() / 60
        if age_minutes > _AGENT_CONTEXT_TTL_MINUTES:
            _pending_agent_context.pop(user.id, None)
            logger.info("_reply: agent context expired for user %d", user.id)
        else:
            target_agent = next((a for a in active_agents if a["id"] == agent_id), None)
            if target_agent:
                replied_to_notification = await memory.get_agent_notification(
                    pool, reply_to_id, chat.id
                )
                if (
                    replied_to_notification
                    and replied_to_notification.get("agent_id") == agent_id
                ):
                    logger.info(
                        "_reply: continuing agent feedback context for %s",
                        target_agent["name"],
                    )
                    active_tasks = await memory.get_active_tasks_for_user(pool, user.id)
                    classified = await intent_classifier.classify(
                        text,
                        pool,
                        has_active_agents=bool(active_agents),
                        has_active_tasks=bool(active_tasks),
                        notification_context=replied_to_notification,
                    )
                    intent = classified["intent"]
                    if intent not in (
                        "agent_feedback",
                        "agent_talk",
                        "agent_trigger",
                        "none",
                    ):
                        classified["intent"] = "agent_feedback"
                    if classified["intent"] != "none":
                        await _handle_agent_intent(
                            update,
                            pool,
                            text,
                            classified["intent"],
                            user.id,
                            chat.id,
                            active_agents,
                            notification_context=replied_to_notification,
                            classified=classified,
                        )
                        return

    active_tasks = await memory.get_active_tasks_for_user(pool, user.id)

    classified = await intent_classifier.classify(
        text,
        pool,
        has_active_agents=bool(active_agents),
        has_active_tasks=bool(active_tasks),
        notification_context=notification_context,
    )
    intent = classified["intent"]
    needs_search = classified["needs_search"]
    wants_voice = force_voice or classified["wants_voice"]

    logger.debug(
        "_reply intent=%s search=%s voice=%s", intent, needs_search, wants_voice
    )

    if intent in (
        "agent_system",
        "agent_create",
        "agent_trigger",
        "agent_talk",
        "agent_list",
        "agent_feedback",
    ):
        await _handle_agent_intent(
            update,
            pool,
            text,
            intent,
            user.id,
            chat.id,
            active_agents,
            notification_context=notification_context,
            classified=classified,
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

    if intent == "monitor_create":
        await _handle_monitor_intent(update, pool, text, user.id)
        return

    agent_ctx_text = ""
    if notification_context and active_agents:
        agent_id = notification_context.get("agent_id")
        target_agent = next((a for a in active_agents if a["id"] == agent_id), None)
        if target_agent:
            ctx = await agent_context.load_shallow(pool, target_agent)
            agent_ctx_text = agent_context.format_for_system_prompt(ctx)

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
        agent_ctx_text=agent_ctx_text,
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

    is_mention = (
        bot_username and f"@{bot_username}".lower() in transcribed.lower()
    ) or config.BOT_NAME.lower() in transcribed.lower()
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
        caption
        and (
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
    forward_context = _extract_forward_context(message)
    await _handle_file_content(
        update,
        pool,
        file_bytes=bytes(file_bytes),
        media_type="image/jpeg",
        caption=caption,
        triggered_by_mention=triggered,
        forward_context=forward_context,
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
    forward_context = _extract_forward_context(message)
    doc_file = await doc.get_file()
    file_bytes = await doc_file.download_as_bytearray()
    await _handle_file_content(
        update,
        pool,
        file_bytes=bytes(file_bytes),
        media_type=SUPPORTED_MIME[mime],
        caption=caption,
        triggered_by_mention=triggered,
        forward_context=forward_context,
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

        pending = await memory.get_pending_confirmation(
            pool, user.id, query.message.chat.id
        )
        if not pending or pending["id"] != confirmation_id:
            await query.edit_message_text("Diese Bestätigung ist abgelaufen.")
            return

        edit_payload = pending["payload"]

        if action == "yes":
            await memory.clear_pending_confirmation(
                pool, user.id, query.message.chat.id
            )
            _pending_agent_context.pop(user.id, None)
            result = await agent_edits.execute_edit(pool, edit_payload)
            rollback_id = confirmation_id
            _pending_rollbacks[rollback_id] = edit_payload
            asyncio.get_event_loop().call_later(
                300, lambda: _pending_rollbacks.pop(rollback_id, None)
            )
            await query.edit_message_text(
                result, reply_markup=agent_edits.rollback_keyboard(rollback_id)
            )

        elif action == "no":
            await memory.clear_pending_confirmation(
                pool, user.id, query.message.chat.id
            )
            _pending_agent_context.pop(user.id, None)
            await query.edit_message_text("Abgebrochen.")

        elif action == "adjust":
            sent = await query.message.reply_text("Was soll ich ändern?")
            await memory.save_agent_notification(
                pool,
                sent.message_id,
                query.message.chat.id,
                edit_payload.get("agent_id", 0),
                "adjust_request",
                {"confirmation_id": confirmation_id},
            )

        elif action == "rollback":
            rollback_payload = _pending_rollbacks.pop(confirmation_id, None)
            if not rollback_payload:
                await query.edit_message_text(
                    "Rückgängig nicht mehr möglich — Zeit abgelaufen."
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

        active_agents = await memory.get_active_agents_for_user(pool, user.id)
        agent = next((a for a in active_agents if a["id"] == agent_id), None)
        if not agent:
            await query.edit_message_text("Dieser Agent existiert nicht mehr.")
            return

        if action == "stop":
            await memory.deactivate_agent(pool, agent_id)
            await query.edit_message_text(f"{agent['name']} wurde gestoppt.")
        elif action == "status":
            state = await memory.get_agent_state(pool, agent_id)
            agent_memories = await memory.get_agent_memories(pool, agent_id)
            status_text, _, _, _ = await agent_parser.handle_agent_talk(
                "Was ist dein aktueller Status und was hast du bisher beobachtet?",
                agent,
                state,
                agent_memories,
                pool=pool,
            )
            await query.message.reply_text(
                f"{agent['name']} — Status:\n\n{status_text}",
                reply_markup=_agent_keyboard(agent_id),
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

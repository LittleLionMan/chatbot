import random
import asyncpg
from bot import config, brain, memory


DECIDER_SYSTEM = """Du entscheidest ob ein Bot mit einem bestimmten Charakter auf eine Gruppennachricht spontan reagieren soll.
Antworte NUR mit 'ja' oder 'nein'.
Reagiere mit 'ja' wenn: die Nachricht eine klare Meinung, eine Behauptung, eine Frage an alle, eine Neuigkeit oder etwas Humorvolles enthält — und der Charakter des Bots dazu etwas Passendes beitragen könnte.
Reagiere mit 'nein' wenn: die Nachricht reine Logistik ist, Small Talk ohne Substanz, oder der Bot kürzlich schon gesprochen hat."""


async def should_respond_spontaneously(
    pool: asyncpg.Pool,
    group_id: int,
    message_text: str,
    bot_character: str,
) -> bool:
    seconds_since = await memory.get_cooldown_seconds_since_last_spontaneous(pool, group_id)
    if seconds_since < config.BOT_SPONTANEOUS_COOLDOWN_SECONDS:
        return False

    if random.random() > config.BOT_SPONTANEOUS_PROBABILITY:
        return False

    decision = await brain.chat(
        system=DECIDER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Charakter des Bots: {bot_character}\n\nNachricht: {message_text}",
            }
        ],
        max_tokens=5,
    )
    return decision.strip().lower().startswith("ja")

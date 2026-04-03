import asyncpg
from bot import config, brain, memory
from bot.soul import SOUL


DECIDER_SYSTEM = """Entscheide ob ein Bot spontan auf eine Gruppennachricht reagieren soll.
Antworte ausschließlich mit dem Wort 'ja' oder dem Wort 'nein'. Keine anderen Wörter, keine Erklärungen.
Reagiere mit 'ja' wenn: die Nachricht eine klare Meinung, Behauptung, Frage an alle, Neuigkeit oder etwas Humorvolles enthält — und der Charakter des Bots dazu etwas Passendes beitragen könnte.
Reagiere mit 'nein' wenn: die Nachricht reine Logistik ist, Small Talk ohne Substanz, oder der Bot kürzlich schon gesprochen hat.
Beispiele: "Was haltet ihr von KI?" → ja, "Bin um 18 Uhr da" → nein, "Habt ihr das gehört?" → ja, "ok" → nein."""


async def should_respond_spontaneously(
    pool: asyncpg.Pool,
    group_id: int,
    message_text: str,
) -> bool:
    seconds_since = await memory.get_cooldown_seconds_since_last_spontaneous(pool, group_id)
    if seconds_since < config.BOT_SPONTANEOUS_COOLDOWN_SECONDS:
        return False

    decision = await brain.chat(
        system=DECIDER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Charakter des Bots:\n{SOUL}\n\nNachricht: {message_text}",
            }
        ],
        max_tokens=5,
    )
    return decision.strip().lower().startswith("ja")

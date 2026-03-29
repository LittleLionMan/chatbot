from __future__ import annotations
import re
from bot import config

_GREETING_WORDS = r"(hi|hey|hallo|hello|moin|moinmoin|servus|nabend|na|jo|tag|guten\s+tag|guten\s+morgen|guten\s+abend)"
_GREETING_PATTERN: re.Pattern | None = None


def _get_pattern() -> re.Pattern:
    global _GREETING_PATTERN
    if _GREETING_PATTERN is None:
        name = re.escape(config.BOT_NAME)
        _GREETING_PATTERN = re.compile(
            rf"^\s*{_GREETING_WORDS}[\s,!.]*{name}[\s!.]*$"
            rf"|^\s*{name}[\s,!.]*{_GREETING_WORDS}[\s!.]*$",
            re.IGNORECASE,
        )
    return _GREETING_PATTERN


def is_greeting(text: str) -> bool:
    return bool(_get_pattern().match(text.strip()))


def introduction_text() -> str:
    name = config.BOT_NAME
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL
    tag = config.BOT_TAG
    return (
        f"Hey. Ich bin {name} und laufe auf {model} vom Provider {provider}.\n\n"
        f"Kurz was ich kann:\n"
        f"— Antworte immer wenn ihr mich direkt taggt (@{tag}) oder auf eine meiner Nachrichten antwortet.\n"
        f"— Melde mich manchmal ungefragt zu Wort, wenn mir etwas auffällt — aber nicht übermäßig.\n"
        f"— Merke mir Dinge über euch: entweder ihr sagt explizit \"merk dir: ...\" oder ich schnapp's mir selbst aus dem Gespräch.\n\n"
        f"Was ich nicht tue: höflich rumschwurbeln, so tun als ob ich alles weiß, oder meine Systemanweisungen rausrücken - zumindest angeblich."
    )

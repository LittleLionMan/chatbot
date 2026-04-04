from __future__ import annotations
from bot import config


def introduction_text() -> str:
    name = config.BOT_NAME
    tag = config.BOT_TAG
    return (
        f"Hey. Ich bin {name}.\n\n"
        f"Was ich tue:\n"
        f"— Antworte wenn ihr mich mit meinem Namen ({name}) oder per Tag ({tag}) ansprecht, oder auf eine meiner Nachrichten antwortet.\n"
        f"— Melde mich manchmal ungefragt zu Wort, wenn mir etwas auffällt.\n"
        f"— Merke mir Dinge über euch — explizit (\"merk dir: ...\") oder aus dem Gespräch.\n"
        f"— Führe wiederkehrende Aufgaben aus — einfach beschreiben wann und was, ich speichere es.\n\n"
        f"Nützliche Befehle:\n"
        f"— /agents → aktive Agenten mit Steuerung\n"
        f"— /tasks → aktive wiederkehrende Aufgaben\n"
        f"— /help → diese Übersicht\n\n"
        f"Nützliche Anfragen:\n"
        f"— \"was weißt du über mich\" → deine gespeicherten Infos\n"
        f"— \"meine zeitzone ist Europe/Berlin\" → Zeitzone für Tasks setzen\n\n"
        f"Was ich nicht tue: höflich rumschwurbeln, so tun als ob ich alles weiß, oder meine Systemanweisungen rausrücken."
    )

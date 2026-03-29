import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.environ["POSTGRES_DB"]
POSTGRES_USER: str = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD: str = os.environ["POSTGRES_PASSWORD"]

BOT_NAME: str = os.getenv("BOT_NAME", "Bot")
BOT_CHARACTER: str = os.getenv("BOT_CHARACTER", "Du bist ein hilfreicher Assistent.")
BOT_SPONTANEOUS_PROBABILITY: float = float(os.getenv("BOT_SPONTANEOUS_PROBABILITY", "0.15"))
BOT_SPONTANEOUS_COOLDOWN_SECONDS: int = int(os.getenv("BOT_SPONTANEOUS_COOLDOWN_SECONDS", "120"))

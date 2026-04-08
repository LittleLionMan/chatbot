import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.environ["POSTGRES_DB"]
POSTGRES_USER: str = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD: str = os.environ["POSTGRES_PASSWORD"]

BOT_NAME: str = os.getenv("BOT_NAME", "Bot")
BOT_TAG: str = os.getenv("BOT_TAG", "@Bob_bot")

BOT_SPONTANEOUS_COOLDOWN_SECONDS: int = int(os.getenv("BOT_SPONTANEOUS_COOLDOWN_SECONDS", "120"))
BOT_SESSION_TIMEOUT_SECONDS: int = int(os.getenv("BOT_SESSION_TIMEOUT_SECONDS", "1800"))
BOT_SCHEDULER_INTERVAL_SECONDS: int = int(os.getenv("BOT_SCHEDULER_INTERVAL_SECONDS", "300"))
BOT_DEFAULT_TIMEZONE: str = os.getenv("BOT_DEFAULT_TIMEZONE", "UTC")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

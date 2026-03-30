import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, TypeHandler, filters
from bot import config, memory, handler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


async def post_init(application) -> None:
    application.bot_data["pool"] = await memory.get_pool()
    logging.info("Database pool initialized.")


async def _debug_all(update: Update, context) -> None:
    msg = update.effective_message
    if msg:
        logger.info(
            "DEBUG update: text=%s voice=%s audio=%s",
            bool(msg.text),
            bool(msg.voice),
            bool(msg.audio),
        )


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(TypeHandler(Update, _debug_all), group=-1)

    app.add_handler(
        MessageHandler(filters.VOICE, handler.handle_voice)
    )
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handler.handle_message)
    )

    logging.info("Bot starting...")
    app.run_polling(allowed_updates=["message", "edited_message"])


if __name__ == "__main__":
    main()

import logging
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from bot import config, memory, handler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


async def post_init(application) -> None:
    application.bot_data["pool"] = await memory.get_pool()
    logging.info("Database pool initialized.")


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

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

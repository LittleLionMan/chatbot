import asyncio
import logging
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from bot import config, memory, handler, scheduler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


async def post_init(application) -> None:
    pool = await memory.get_pool()
    application.bot_data["pool"] = pool
    logging.info("Database pool initialized.")
    asyncio.create_task(scheduler.run(pool, application.bot))
    logging.info("Scheduler started.")


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
    app.run_polling()


if __name__ == "__main__":
    main()

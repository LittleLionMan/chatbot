import asyncio
import logging
from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
from bot import config, memory, handler, scheduler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(context.error, (NetworkError, TimedOut)):
        logger.debug("Transient network error (ignored): %s", context.error)
        return
    logger.error("Unhandled exception", exc_info=context.error)


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

    app.add_error_handler(error_handler)

    app.add_handler(CommandHandler("help", handler.handle_command_help))
    app.add_handler(CommandHandler("agents", handler.handle_command_agents))
    app.add_handler(CommandHandler("tasks", handler.handle_command_tasks))

    app.add_handler(CallbackQueryHandler(handler.handle_callback_query, pattern=r"^agent:"))

    app.add_handler(MessageHandler(filters.VOICE, handler.handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handler.handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handler.handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler.handle_message))

    logging.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()

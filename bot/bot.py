import os
import requests
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()

logger = logging.getLogger(__name__)


BOT_TOKEN = os.getenv("BOT_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000")


def format_reply(data: dict) -> str:
    label = data.get("label")
    text = data.get("text", "")
    if label == "positive":
        return f"Тональность: ПОЛОЖИТЕЛЬНАЯ\n\nРаспознанный текст: {text}"
    if label == "negative":
        return f"Тональность: ОТРИЦАТЕЛЬНАЯ\n\nРаспознанный текст: {text}"
    return "Не удалось определить тональность."


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None or update.message.voice is None:
        return

    file = await context.bot.get_file(update.message.voice.file_id)
    voice_bytes = await file.download_as_bytearray()

    try:
        resp = requests.post(
            f"{SERVER_URL}/predict",
            files={"file": ("voice.ogg", bytes(voice_bytes), "audio/ogg")},
            timeout=300,
        )
        data = resp.json()
    except Exception as err:
        logger.exception(f"{err.args}")
        await update.message.reply_text("Ошибка при обращении к серверу.")
        return

    if "error" in data:
        await update.message.reply_text(f"Ошибка: {data['error']}")
        return

    await update.message.reply_text(format_reply(data))


def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set. Put it into .env")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.run_polling()


if __name__ == "__main__":
    main()

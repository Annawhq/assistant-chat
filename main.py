import random
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from filter import text_filter
from secret_token import TOKEN
from speech_recognition import Speech
from vectorize import vectorizer, model, INTENTS


def get_intent_ml(user_text):
    user_text = text_filter(user_text)
    print(user_text)
    vec_text = vectorizer.transform([user_text])
    print(vec_text)
    if vec_text.nnz != 0:
        intent = model.predict(vec_text)[0]
        print(intent)
        return intent
    return "not_that"


def get_random_response(intent):
    return random.choice(INTENTS[intent]["responses"])


def bot(user_text):
    intent = get_intent_ml(user_text)
    if intent is None:
        return get_random_response("not_that")
    return get_random_response(intent)


app = ApplicationBuilder().token(TOKEN).build()


async def telegram_reply(upd: Update, ctx):
    name = upd.message.from_user.full_name
    user_text = upd.message.text
    print(f"{name}: {user_text}")
    reply = bot(user_text)
    print(f"BOT: {reply}")
    await upd.message.reply_text(reply)


async def telegram_reply_voice(upd: Update, ctx):
    name = upd.message.from_user.full_name
    speech = Speech()
    user_voice = await upd.message.voice.get_file()
    user_voice = user_voice.file_path
    user_voice = speech.recognize(user_voice)
    print(f"{name}: {user_voice}")
    reply = bot(user_voice)
    print(f"BOT: {reply}")
    await upd.message.reply_text(reply)

handler_text = MessageHandler(filters.TEXT, telegram_reply)  # Создаем обработчик текстовых сообщений
app.add_handler(handler_text)  # Добавляем обработчик в приложение
handler_voice = MessageHandler(filters.VOICE, telegram_reply_voice)  # Создаем обработчик голосовых сообщений
app.add_handler(handler_voice)  # Добавляем обработчик в приложение
app.run_polling()

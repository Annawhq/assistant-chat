import random
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters
from filter import text_filter
from secret_token import TOKEN
from vectorize import vectorizer, model, INTENTS


def get_intent_ml(user_text):
    user_text = text_filter(user_text)
    print(user_text)
    vec_text = vectorizer.transform([user_text])
    if vec_text.nnz == 0:
        intent = model.predict(vec_text)[0]
        print(intent)
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

handler = MessageHandler(filters.TEXT, telegram_reply)  # Создаем обработчик текстовых сообщений
app.add_handler(handler)  # Добавляем обработчик в приложение

app.run_polling()

from pymorphy2 import MorphAnalyzer
import re
import nltk
morph = MorphAnalyzer()


def lemmatize(text):
    tokens = []
    for token in text.split():
        token = token.strip()
        token = morph.normal_forms(token)[0]
        tokens.append(token)
    text = " ".join(tokens).strip()
    return text


def text_filter(text):
    text = text.lower().strip() # strip - вырезать пробелы с начала и конца строки
    pattern = r"[^\w\s]+|[\d]+"
    text = re.sub(pattern, "", text) # Из переменной text вырезаем "Все что не слово и не пробел"
    return lemmatize(text)


def text_match(user_text, example):
    user_text = text_filter(user_text)
    example = text_filter(example)
    if user_text == example:
        return True
    if user_text.find(example) != -1:
        return True

    distance = nltk.edit_distance(user_text, example)
    # Отношение кол-ва ошибок к длине слова, 1.0 - слово целиком другое, 0 - слова полностью совпадают
    ratio = distance / len(example)
    if ratio < 0.40:
        return True
    return False

import pymorphy2
import re
import nltk


def lemmatize(text):
    res = []
    morph = pymorphy2.MorphAnalyzer()
    for line in text:
        res_line = []
        for word in line:
            res_line.append(morph.parse(word)[0].normal_form)
        res.append(' '.join(res_line))
    text = "".join(res).strip()
    return text


def text_filter(text):
    text = text.lower()
    text = text.strip()  # strip - вырезать пробелы с начала и конца строки
    pattern = r"[^\w\s]"
    text = re.sub(pattern, "", text)  # Из переменной text вырезаем "Все что не слово и не пробел"
    text = lemmatize(text)
    return text


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

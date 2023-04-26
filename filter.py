import re
import string
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
russian_stopwords = stopwords.words("russian")
morph = MorphAnalyzer()


def lemmatize(text):
    tokens = []
    for token in text.split():
        token = token.strip()
        token = morph.normal_forms(token)[0]
        if token not in (russian_stopwords and string.punctuation):
            tokens.append(token)
    text = " ".join(tokens).strip()
    return text


def text_filter(text):
    text = text.lower().strip()
    pattern = r"[^\w\s]+|[\d]+"
    text = re.sub(pattern, "", text)
    return lemmatize(text)

